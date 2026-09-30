"""Post hoc 4-pixel near/far aggregates for existing A1/E0 500k evaluations, no game.

Uses the original coverage distance and fitted trajectory split. Refuses input
pin or episode identity mismatches. Never reads fresh v2 starts. See the private
near-far protocol for interpretation limits. Output contains aggregates only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from celeste_rl.cloning import OBS_KEYS, Demonstrations, split_by_trajectory
from celeste_rl.texthash import matches_text_hash, text_sha256
from describe_coverage_ceiling import CANONICAL, distances, pixels

STARTS = 'config/heldout_starts.json'
FINAL = 'step_000501760'
BATCHES = {
    'pilot': ('config/campaign-ppo-anchor-eval.json', 'runs/campaign/20260929-165751-ppo-anchor-pilot-eval-v1/summary.json', 'room1-', 'ppo-anchor-pilot'),
    'new': ('config/campaign-ppo-tiebreak-eval.json', 'runs/campaign/20260930-054156-ppo-anchor-tiebreak-eval-v1/summary.json', 'tb-room1-', 'ppo-tiebreak'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(name):
    path = (REPO / name).resolve()
    require(path.is_relative_to(REPO) and 'v2' not in path.name, f'Forbidden input: {name}')
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(episodes):
    by_route = {}
    for e in episodes:
        by_route.setdefault(e['route'], []).append(e['ending'] == 'success')
    return {'successes': sum(e['ending'] == 'success' for e in episodes),
            'episodes': len(episodes),
            'success_rate': sum(e['ending'] == 'success' for e in episodes) / len(episodes) if episodes else None,
            'represented_routes': len(by_route),
            'route_macro': statistics.mean(statistics.mean(v) for v in by_route.values()) if by_route else None}


def validate_episodes(episodes, entries, checkpoint_sha):
    expected = {e['state_id']: e for e in entries}
    require(len(expected) == len(entries), 'Duplicate starts')
    require(len(episodes) == len(entries), 'Wrong episode count')
    seen = set()
    for episode in episodes:
        sid = episode['state_id']
        require(sid in expected and sid not in seen, 'Unknown or duplicate episode state')
        seen.add(sid)
        entry = expected[sid]
        require(episode['repeat'] == 0 and episode['problem'] is None, 'Repeated or faulty evaluation')
        require(episode['checkpoint_sha256'] == checkpoint_sha, 'Episode checkpoint mismatch')
        require(episode['route'] == entry['route'] and episode['start_position'] == entry['position']
                and episode['start_dashes'] == entry['dashes'] and episode['start_frames'] == entry['frames']
                and episode['start_room'] == entry['room'], 'Episode start metadata mismatch')
    return {e['state_id']: e for e in episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists')
    inputs = {}

    def read(name, expected=None):
        path = safe_path(name)
        digest = sha(path)
        require(expected is None or digest == expected, f'Changed input: {name}')
        inputs[path.relative_to(REPO).as_posix()] = digest
        return json.loads(path.read_text(encoding='utf-8'))

    held = read(STARTS)
    entries = held['entries']
    require(len(entries) == 200, 'Expected 200 v1 starts')
    starts = np.array([[e['position'][0], e['position'][1], e['dashes']] for e in entries])
    pilot = read('config/ppo-anchor-pilot.json')
    published = {'pilot': read('docs/results/ppo-anchor-pilot.json'), 'new': read('docs/results/ppo-anchor-tiebreak.json')}
    coverage = read('docs/results/coverage-ceiling-descriptive.json')['coverage']
    for name in ('scripts/describe_coverage_ceiling.py', 'celeste_rl/cloning.py', 'celeste_rl/schema.py'):
        inputs[name] = sha(safe_path(name))
    masks, groups, records = {}, {}, {}
    for k in range(4):
        recording = pilot['anchor_data']['recordings'][str(3+k)]
        path = safe_path(recording['play'] + '/dataset.npz')
        require(sha(path) == recording['dataset_sha256'], 'Recording pin mismatch')
        inputs[path.relative_to(REPO).as_posix()] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            demos = Demonstrations({key: data[f'obs_{key}'] for key in OBS_KEYS}, data['actions'], data['trajectory'])
        require(len(demos) == recording['frames'], 'Recording frame count mismatch')
        fitted, held_frames = split_by_trajectory(demos, .25, k)
        require(len(fitted.trajectories) == 19 and len(held_frames.trajectories) == 6, 'Wrong fitted split')
        first = np.r_[0, np.flatnonzero(np.diff(demos.trajectory)) + 1]
        require({tuple(row) for row in pixels(demos)[first]} == {CANONICAL}, 'Pixel conversion check failed')
        d, fallbacks = distances(pixels(fitted), starts)
        require(fallbacks == 0, 'Missing same-dash fitted frame')
        near = d <= 4
        previous = coverage[f'room1-S-play-k{k}']
        require(len(fitted) == previous['fitted_frames'] and float(near.mean()) == previous['within_4px'], 'Prior coverage mismatch')
        masks[k] = {entry['state_id']: bool(hit) for entry, hit in zip(entries, near)}
        groups[f'k{k}'] = {'donor': 3+k, 'fitted_frames': len(fitted), 'near_starts': int(near.sum()),
                            'far_starts': int((~near).sum()), 'same_dash_fallbacks': fallbacks}
        records[k] = (recording, fitted, held_frames)

    runs = {}
    for batch, (plan_name, summary_name, prefix, folder) in BATCHES.items():
        plan, summary = read(plan_name), read(summary_name)
        require(matches_text_hash(safe_path(plan_name), summary['plan_sha256']), 'Summary plan mismatch')
        planned = {r['id']: r for r in plan['runs']}
        actual = {r['id']: r for r in summary['results']}
        require(len(actual) == len(summary['results']) and set(actual) == set(planned), 'Summary ids mismatch')
        for k in range(4):
            manifest = read(f'runs/train/{folder}-A1-k{k}/manifest.json')
            for session in manifest['sessions']:
                anchor = session['provenance']['anchor']
                recording, fitted, held_frames = records[k]
                require(anchor['play']['dataset_sha256'] == recording['dataset_sha256'] and anchor['split_seed'] == k
                        and anchor['holdout'] == .25 and anchor['fitted_frames'] == len(fitted)
                        and anchor['held_frames'] == len(held_frames), 'Anchor training split mismatch')
            for arm in ('A1', 'E0'):
                rid = f'{prefix}{arm}-k{k}-{FINAL}'
                item, spec = actual[rid], planned[rid]
                require(item['status'] == 'ok' and STARTS in spec['command'], 'Wrong evaluation or starts')
                artifact = item['artifact']
                result = read(artifact['result_file'], artifact['result_sha256'])
                pin = spec['checkpoint_sha256']
                require(sha(safe_path(spec['checkpoint'])) == pin, 'Checkpoint changed')
                require(result['attributable'] and result['checkpoint_sha256'] == pin and result['heldout_sha256'] == held['sha256']
                        and result['heldout_file_sha256'] == inputs[STARTS] and result['repeats'] == 1, 'Result provenance mismatch')
                ep_path = safe_path(artifact['episodes_file'])
                require(sha(ep_path) == artifact['episodes_sha256'], 'Episodes changed')
                inputs[ep_path.relative_to(REPO).as_posix()] = sha(ep_path)
                episodes = [json.loads(line) for line in ep_path.read_text().splitlines()]
                indexed = validate_episodes(episodes, entries, pin)
                full = aggregate(episodes)
                require(full['successes'] == result['successes'] and abs(full['route_macro'] - result['route_macro_success_rate']) < 1e-12, 'Full outcome reconciliation failed')
                if batch == 'pilot':
                    saved = published[batch]['runs'][f'{arm}-k{k}']['room1_curve'][-1]
                    require(full['successes'] == saved['successes'], 'Published pilot success count mismatch')
                    saved_macro = saved['route_macro']
                else:
                    saved_macro = published[batch]['new_runs'][f'{arm}-k{k}']['room1_route_macro_clone_then_checkpoints'][-1]
                require(round(full['route_macro'], 4) == saved_macro, 'Published final route macro mismatch')
                row = {group: aggregate([indexed[sid] for sid, is_near in masks[k].items() if is_near == (group == 'near')]) for group in ('near','far')}
                require(sum(r['episodes'] for r in row.values()) == 200 and sum(r['successes'] for r in row.values()) == full['successes'], 'Split reconciliation failed')
                runs[f'{batch}-{arm}-k{k}'] = row

    pooled, pairs = {}, {}
    for group in ('near','far'):
        pooled[group] = {}
        for arm in ('A1','E0'):
            rows = [runs[f'{b}-{arm}-k{k}'][group] for b in BATCHES for k in range(4)]
            n, successes = sum(r['episodes'] for r in rows), sum(r['successes'] for r in rows)
            pooled[group][arm] = {'successes': successes, 'episodes': n, 'success_rate': successes/n,
                                  'mean_run_route_macro': statistics.mean(r['route_macro'] for r in rows)}
        diffs = [runs[f'{b}-A1-k{k}'][group]['route_macro'] - runs[f'{b}-E0-k{k}'][group]['route_macro'] for b in BATCHES for k in range(4)]
        pairs[group] = {'mean_route_macro_difference_a1_minus_e0': statistics.mean(diffs), 'a1_higher_of_8': sum(v > 0 for v in diffs),
                        'by_batch_and_clone': {f'{b}-k{k}': runs[f'{b}-A1-k{k}'][group]['route_macro'] - runs[f'{b}-E0-k{k}'][group]['route_macro'] for b in BATCHES for k in range(4)}}
    batch_pooled = {}
    for batch in BATCHES:
        batch_pooled[batch] = {}
        for group in ('near','far'):
            batch_pooled[batch][group] = {}
            for arm in ('A1','E0'):
                rows = [runs[f'{batch}-{arm}-k{k}'][group] for k in range(4)]
                n, successes = sum(r['episodes'] for r in rows), sum(r['successes'] for r in rows)
                batch_pooled[batch][group][arm] = {'successes': successes, 'episodes': n, 'success_rate': successes/n,
                                                 'mean_run_route_macro': statistics.mean(r['route_macro'] for r in rows)}
    out = {'label': 'post hoc descriptive near/far check; aggregates only; no causal or confirmatory claim',
           'analysis_code_sha256': text_sha256(Path(__file__)), 'inputs_sha256': inputs,
           'method': 'near <=4 px Euclidean to fitted donor frame with same dash count; far >4 px; position/dashes only',
           'groups': groups, 'runs': runs, 'pooled': pooled, 'by_batch': batch_pooled, 'paired': pairs,
           'record_checks': 'passed: pins, fitted splits, prior coverage, episode identity/metadata and full/split totals',
           'limits': ['Post hoc, no threshold sweep.', 'Four shared clones, two PPO seeds per arm per clone; correlated repeats.',
                      'Within-group route macro averages represented routes only, then runs equally.', 'Start distance omits speed, timers and later trajectory.',
                      'Near/far groups differ in composition and difficulty; cannot separate retention from rehearsal causally.', 'Neither fresh v2 set read; declared selection unchanged.']}
    args.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pooled': pooled, 'paired': pairs}, indent=2))


if __name__ == '__main__':
    main()

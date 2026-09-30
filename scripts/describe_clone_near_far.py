"""Post hoc starting-clone near/far scores and signed percentage-point decay.

Only four existing Room 1 v1 clone evaluations are added to the reviewed final
near/far report. No games or fresh v2 reads; aggregates only. Positive loss is
clone success minus final success. Shared baselines are not independent samples.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from celeste_rl.cloning import OBS_KEYS, Demonstrations, split_by_trajectory
from celeste_rl.texthash import matches_text_hash, text_sha256
from describe_tiebreak_near_far import STARTS, aggregate, distances, pixels, require, safe_path, sha, input_sha, validate_episodes

PLAN = 'config/campaign-mixed-self-distillation-clone-eval.json'
SUMMARY = 'runs/campaign/20260928-012535-mixed-self-distillation-clone-eval-v1/summary.json'
FINALS = 'docs/results/ppo-tiebreak-near-far.json'


def loss_row(baseline, final):
    """Signed differences in success rates, not normalized donor-margin units."""
    return {f'{metric}_loss_percentage_points': 100*(baseline[metric]-final[metric]) for metric in ('success_rate','route_macro')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists')
    inputs = {}
    def read(name, expected=None):
        path = safe_path(name)
        require(expected is None or sha(path) == expected, f'Changed artifact: {name}')
        inputs[name] = input_sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    finals = read(FINALS)
    for name, pin in finals['inputs_sha256'].items():
        require(input_sha(safe_path(name)) == pin, f'Prior near/far input changed: {name}')
    for name in ('scripts/describe_tiebreak_near_far.py','scripts/describe_coverage_ceiling.py','celeste_rl/cloning.py','celeste_rl/schema.py'):
        inputs[name] = input_sha(safe_path(name))
    require(text_sha256(safe_path('scripts/describe_tiebreak_near_far.py')) == finals['analysis_code_sha256'], 'Finals analyzer code mismatch')
    held = read(STARTS)
    entries = held['entries']
    starts = np.array([[e['position'][0],e['position'][1],e['dashes']] for e in entries])
    pilot = read('config/ppo-anchor-pilot.json')
    published = read('docs/results/mixed-self-distillation-clone.json')['clones']
    plan, summary = read(PLAN), read(SUMMARY)
    require(matches_text_hash(safe_path(PLAN), summary['plan_sha256']), 'Clone summary plan mismatch')
    actual = {r['id']:r for r in summary['results']}
    planned = {r['id']:r for r in plan['runs']}
    require(len(actual) == len(summary['results']) and set(actual) == set(planned), 'Clone summary ids mismatch')
    training = [read(name) for name in ('config/campaign-ppo-anchor-train.json','config/campaign-ppo-tiebreak-train.json')]
    baselines, losses = {}, {}
    for k in range(4):
        recording = pilot['anchor_data']['recordings'][str(k+3)]
        path = safe_path(recording['play']+'/dataset.npz')
        require(sha(path) == recording['dataset_sha256'], 'Recording pin mismatch')
        inputs[path.relative_to(REPO).as_posix()] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            demos = Demonstrations({key:data[f'obs_{key}'] for key in OBS_KEYS},data['actions'],data['trajectory'])
        fitted, _ = split_by_trajectory(demos,.25,k)
        d, fallback = distances(pixels(fitted),starts)
        mask = {e['state_id']:bool(v <=4) for e,v in zip(entries,d)}
        require(fallback == 0 and len(fitted) == finals['groups'][f'k{k}']['fitted_frames']
                and sum(mask.values()) == finals['groups'][f'k{k}']['near_starts'], 'Prior grouping mismatch')
        rid = f'room1-SD-k{k}-clone'
        item,spec = actual[rid],planned[rid]
        pin = spec['checkpoint_sha256']
        require(item['status'] == 'ok' and STARTS in spec['command'], 'Wrong clone evaluation')
        require(sha(safe_path(spec['checkpoint'])) == pin and pin == plan['clones'][f'SD-k{k}']['sha256'], 'Clone checkpoint pin mismatch')
        for tp in training:
            for arm in ('A1','E0'):
                run = next(r for r in tp['runs'] if r['arm'] == arm and r['k'] == k)
                require(run['init_from'] == spec['checkpoint'] and run['clone_sha256'] == pin, 'Starting clone mismatch')
        artifact = item['artifact']
        result = read(artifact['result_file'],artifact['result_sha256'])
        require(result['attributable'] and result['checkpoint_sha256'] == pin and result['heldout_sha256'] == held['sha256']
                and matches_text_hash(safe_path(STARTS),result['heldout_file_sha256']) and result['repeats'] == 1,'Clone result provenance mismatch')
        ep_path = safe_path(artifact['episodes_file'])
        require(sha(ep_path) == artifact['episodes_sha256'],'Clone episode hash mismatch')
        inputs[artifact['episodes_file']] = sha(ep_path)
        episodes = [json.loads(line) for line in ep_path.read_text().splitlines()]
        indexed = validate_episodes(episodes,entries,pin)
        full = aggregate(episodes)
        saved = published[f'SD-k{k}']['room1']
        require(full['successes'] == result['successes'] == saved['successes']
                and abs(full['route_macro']-result['route_macro_success_rate']) < 1e-12
                and round(full['route_macro'],4) == saved['route_macro'],'Clone full-set mismatch')
        row = {group:aggregate([indexed[sid] for sid,near in mask.items() if near == (group == 'near')]) for group in ('near','far')}
        require(sum(r['successes'] for r in row.values()) == full['successes'] and sum(r['episodes'] for r in row.values()) == 200,'Clone split reconciliation failed')
        require(all(r['represented_routes'] == 11 for r in row.values()),'Missing clone subgroup routes')
        baselines[f'k{k}'] = row
        for batch in ('pilot','new'):
            for arm in ('A1','E0'):
                name = f'{batch}-{arm}-k{k}'
                losses[name] = {}
                for group in ('near','far'):
                    final = finals['runs'][name][group]
                    require(row[group]['episodes'] == final['episodes'],'Baseline/final subgroup count mismatch')
                    losses[name][group] = loss_row(row[group],final)
    means = {}
    for group in ('near','far'):
        rows = [b[group] for b in baselines.values()]
        n,s = sum(r['episodes'] for r in rows),sum(r['successes'] for r in rows)
        baseline_macro = statistics.mean(r['route_macro'] for r in rows)
        means[group] = {'baseline_successes_once':s,'baseline_episodes_once':n,'baseline_success_rate':s/n,
                        'baseline_mean_clone_route_macro':baseline_macro,'arms':{}}
        for arm in ('A1','E0'):
            final = finals['pooled'][group][arm]
            require(final['episodes'] == 2*n,'Pooled subgroup count mismatch')
            means[group]['arms'][arm] = {'final_success_rate':final['success_rate'], 'final_mean_run_route_macro':final['mean_run_route_macro'],
                'state_weighted_loss_percentage_points':100*(s/n-final['success_rate']),
                'mean_route_macro_loss_percentage_points':100*(baseline_macro-final['mean_run_route_macro'])}
    by_batch = {batch:{group:{arm:{metric:statistics.mean(losses[f'{batch}-{arm}-k{k}'][group][metric] for k in range(4))
        for metric in ('success_rate_loss_percentage_points','route_macro_loss_percentage_points')} for arm in ('A1','E0')} for group in ('near','far')} for batch in ('pilot','new')}
    out = {'label':'post hoc clone near/far baselines; signed decay in success percentage points; aggregates only',
           'analysis_code_sha256':text_sha256(Path(__file__)), 'inputs_sha256':inputs,'groups':finals['groups'],
           'clone_baselines':baselines,'signed_losses':losses,'means':means,'mean_losses_by_batch':by_batch,
           'record_checks':'passed: prior final input pins, masks, shared initial clone pins, attributable clone evaluations, episode identity, full/split outcome totals',
           'limits':['Four baselines measured once, shared across both arms and PPO batches; correlated noise.',
                     'Positive loss means decline; negative means improvement; not normalized donor-margin loss.',
                     'Thin far routes, unequal clone composition and stochastic outcomes remain.',
                     'Historical clone and final evaluations occurred at different times.',
                     'Neither subgroup difficulty nor subsequent trajectory overlap is controlled; no causal or significance claim.',
                     'Donor subgroup baselines not computed; no fresh v2 reads; declared A1 selection unchanged.']}
    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(means,indent=2))


if __name__ == '__main__':
    main()

"""Post hoc near/far decay of the four original SD-k controls, existing v1 outcomes only."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from celeste_rl.cloning import OBS_KEYS, Demonstrations, split_by_trajectory
from celeste_rl.texthash import matches_text_hash, text_sha256
from describe_tiebreak_near_far import STARTS, aggregate, distances, pixels, require, safe_path, sha, input_sha, validate_episodes
from describe_clone_near_far import loss_row

PLAN = 'config/campaign-mixed-self-distillation-ppo-eval.json'
SUMMARY = 'runs/campaign/20260928-035401-mixed-self-distillation-ppo-eval-v1/summary.json'
BASELINES = 'docs/results/clone-near-far.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(),'Output already exists')
    inputs = {}
    def read(name,expected=None):
        path = safe_path(name)
        require(expected is None or sha(path) == expected,f'Changed artifact: {name}')
        inputs[name] = input_sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    baselines = read(BASELINES)
    for name,pin in baselines['inputs_sha256'].items():
        require(input_sha(safe_path(name)) == pin,f'Baseline input changed: {name}')
    require(text_sha256(safe_path('scripts/describe_clone_near_far.py')) == baselines['analysis_code_sha256'],'Baseline code changed')
    for name in ('scripts/describe_clone_near_far.py','scripts/describe_tiebreak_near_far.py','scripts/describe_coverage_ceiling.py','celeste_rl/cloning.py','celeste_rl/schema.py'):
        inputs[name] = input_sha(safe_path(name))
    held = read(STARTS)
    entries = held['entries']
    starts = np.array([[e['position'][0],e['position'][1],e['dashes']] for e in entries])
    pilot = read('config/ppo-anchor-pilot.json')
    published = read('docs/results/mixed-self-distillation-ppo.json')
    clone_plan = read('config/campaign-mixed-self-distillation-clone-eval.json')
    plan,summary = read(PLAN),read(SUMMARY)
    require(matches_text_hash(safe_path(PLAN),summary['plan_sha256']),'Control summary plan mismatch')
    require(matches_text_hash(safe_path(PLAN),published['evaluation_plan_sha256']),'Published control plan mismatch')
    planned = {r['id']:r for r in plan['runs']}
    actual = {r['id']:r for r in summary['results']}
    require(len(actual) == len(summary['results']) and set(actual) == set(planned),'Summary ids mismatch')
    runs,losses = {},{}
    for k in range(4):
        recording = pilot['anchor_data']['recordings'][str(k+3)]
        path = safe_path(recording['play']+'/dataset.npz')
        require(sha(path) == recording['dataset_sha256'],'Recording pin mismatch')
        inputs[path.relative_to(REPO).as_posix()] = sha(path)
        with np.load(path,allow_pickle=False) as data:
            demos = Demonstrations({key:data[f'obs_{key}'] for key in OBS_KEYS},data['actions'],data['trajectory'])
        fitted,_ = split_by_trajectory(demos,.25,k)
        d,fallback = distances(pixels(fitted),starts)
        mask = {e['state_id']:bool(v <=4) for e,v in zip(entries,d)}
        require(fallback == 0 and sum(mask.values()) == baselines['groups'][f'k{k}']['near_starts']
                and len(fitted) == baselines['groups'][f'k{k}']['fitted_frames'],'Baseline grouping mismatch')
        rid = f'room1-SD-k{k}-step_000501760'
        item,spec = actual[rid],planned[rid]
        pin = spec['checkpoint_sha256']
        require(item['status'] == 'ok' and STARTS in spec['command'],'Wrong control evaluation')
        require(sha(safe_path(spec['checkpoint'])) == pin,'Control checkpoint pin mismatch')
        manifest = read(f'runs/train/mixed-self-distillation-pilot-SD-k{k}/manifest.json')
        clone = clone_plan['clones'][f'SD-k{k}']
        require(manifest['config']['init_from'] == clone['dir']+'/cloned.zip'
                and sha(safe_path(manifest['config']['init_from'])) == clone['sha256'],'Control starting clone mismatch')
        require(manifest['config']['ent_coef'] == .01 and manifest['accepted_steps'] == 501760
                and all(s['provenance'].get('anchor') is None and s['provenance']['attributable'] for s in manifest['sessions']),'Control training recipe mismatch')
        artifact = item['artifact']
        result = read(artifact['result_file'],artifact['result_sha256'])
        require(result['attributable'] and result['checkpoint_sha256'] == pin and result['heldout_sha256'] == held['sha256']
                and matches_text_hash(safe_path(STARTS),result['heldout_file_sha256']) and result['repeats'] == 1,'Control result provenance mismatch')
        ep_path = safe_path(artifact['episodes_file'])
        require(sha(ep_path) == artifact['episodes_sha256'],'Control episode pin mismatch')
        inputs[artifact['episodes_file']] = sha(ep_path)
        episodes = [json.loads(line) for line in ep_path.read_text().splitlines()]
        indexed = validate_episodes(episodes,entries,pin)
        full = aggregate(episodes)
        saved = published['runs'][f'SD-k{k}']['room1_curve'][-1]
        require(full['successes'] == result['successes'] == saved['successes']
                and abs(full['route_macro']-result['route_macro_success_rate']) <1e-12
                and round(full['route_macro'],4) == saved['route_macro'],'Published control outcome mismatch')
        row = {group:aggregate([indexed[sid] for sid,near in mask.items() if near == (group == 'near')]) for group in ('near','far')}
        require(sum(r['successes'] for r in row.values()) == full['successes'],'Control split success mismatch')
        losses[f'k{k}'] = {}
        for group in ('near','far'):
            baseline = baselines['clone_baselines'][f'k{k}'][group]
            require(row[group]['episodes'] == baseline['episodes'] and row[group]['represented_routes'] == 11,'Control subgroup mismatch')
            losses[f'k{k}'][group] = loss_row(baseline,row[group])
        runs[f'k{k}'] = row
    means = {}
    for group in ('near','far'):
        rows = [r[group] for r in runs.values()]
        s,n = sum(r['successes'] for r in rows),sum(r['episodes'] for r in rows)
        macro = statistics.mean(r['route_macro'] for r in rows)
        baseline = baselines['means'][group]
        means[group] = {'baseline_successes':baseline['baseline_successes_once'],'baseline_episodes':baseline['baseline_episodes_once'],
                        'baseline_success_rate':baseline['baseline_success_rate'],'baseline_mean_clone_route_macro':baseline['baseline_mean_clone_route_macro'],
                        'control_successes':s,'control_episodes':n,'control_success_rate':s/n,'control_mean_route_macro':macro,
                        'control_state_weighted_loss_percentage_points':100*(baseline['baseline_success_rate']-s/n),
                        'control_mean_route_macro_loss_percentage_points':100*(baseline['baseline_mean_clone_route_macro']-macro),
                        'a1_e0_context_two_batches':baseline['arms']}
    out = {'label':'post hoc control near/far losses; success percentage points, aggregates only; no causal claim',
           'analysis_code_sha256':text_sha256(Path(__file__)),'inputs_sha256':inputs,'groups':baselines['groups'],
           'control_finals':runs,'signed_losses':losses,'means':means,
           'record_checks':'passed: baseline inputs, shared clone/checkpoint pins, entropy .01/no anchor recipe, attributable evaluations, episode identities and full/split totals',
           'limits':['One control PPO run versus two A1/E0 PPO runs per clone; historical campaigns differ in time/concurrency.',
                     'Four shared clone baselines, stochastic observations and clustered development starts.',
                     'Thin far routes, unequal clone composition and subsequent trajectory overlap remain.',
                     'Positive loss means observed decay in success points, not normalized donor-margin units.',
                     'Neither fresh v2 set read; selection stays A1; no significance/generalization claim.']}
    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(means,indent=2))


if __name__ == '__main__':
    main()

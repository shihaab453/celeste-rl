"""Phase 1 CartPole-v1 benchmark: Our PPO implementation vs Stable-Baselines3.

Trains both implementations on CartPole-v1 across seeds 0, 1, and 2 on CPU.
Evaluates 20 fresh episodes per seed with greedy actions (pass threshold: mean return >= 475).
Verifies save and reload round-trips match evaluation returns exactly.
Saves structured benchmark results to runs/phase1/<timestamp>/results.json.

Run from repo root:
    .venv-rl/Scripts/python.exe scripts/phase1_cartpole.py
"""
from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env

from celeste_rl.ppo import ActorCritic, PPOConfig, evaluate, train_ppo


def evaluate_sb3(
    model: SB3PPO,
    env_id: str = "CartPole-v1",
    n_eval_episodes: int = 20,
    base_seed: int = 100,
) -> tuple[float, float, list[float]]:
    """Evaluate an SB3 model using greedy (deterministic) actions on fresh episodes."""
    eval_env = gym.make(env_id)
    returns: list[float] = []

    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset(seed=base_seed + ep)
        ep_return = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action.item())
            ep_return += float(reward)

        returns.append(ep_return)

    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns)), returns


def run_benchmark():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_dir = Path("runs") / "phase1" / timestamp
    models_dir = runs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    total_timesteps = 100_000
    pass_threshold = 475.0
    eval_episodes = 20

    print("=" * 80)
    print("PHASE 1: PPO CARTPOLE-V1 BENCHMARK")
    print(f"Timestamp:        {timestamp}")
    print(f"Seeds:            {seeds}")
    print(f"Timesteps:        {total_timesteps:,} steps per run")
    print(f"Pass Criterion:   Mean Return >= {pass_threshold:.1f} over {eval_episodes} evaluation episodes")
    print(f"Device:           CPU")
    print("=" * 80)

    results_data: dict[str, dict[str, dict]] = {
        "our_ppo": {},
        "sb3_ppo": {},
    }

    table_rows = []
    overall_start_time = time.perf_counter()

    # 1. Benchmark Our PPO Implementation
    print("\n--- Training Our PPO Implementation ---")
    for seed in seeds:
        print(f"\n[Our PPO] Starting Seed {seed}...")
        config = PPOConfig(
            total_timesteps=total_timesteps,
            learning_rate=1e-3,
            num_envs=4,
            num_steps=128,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=4,
            update_epochs=4,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            hidden_dim=64,
        )

        model, train_metrics = train_ppo("CartPole-v1", config=config, device="cpu")
        wall_time = train_metrics["wall_time"]

        # Evaluate model on fresh episodes
        eval_base_seed = 1000 + seed * 100
        mean_ret, std_ret, all_rets = evaluate(
            model, "CartPole-v1", n_eval_episodes=eval_episodes, seed=eval_base_seed
        )

        # Save and verify reload
        save_path = models_dir / f"our_ppo_seed_{seed}.pt"
        model.save(save_path)

        reloaded_model = ActorCritic.load(save_path, device="cpu")
        reload_mean, reload_std, reload_rets = evaluate(
            reloaded_model, "CartPole-v1", n_eval_episodes=eval_episodes, seed=eval_base_seed
        )
        reload_exact_match = (all_rets == reload_rets)

        passed = bool(mean_ret >= pass_threshold and reload_exact_match)
        status_str = "PASS" if passed else "FAIL"

        print(f"[Our PPO] Seed {seed}: Return = {mean_ret:.1f} +/- {std_ret:.1f} | Time = {wall_time:.1f}s | Reload Match = {reload_exact_match} | {status_str}")

        # Downsample learning curve to a coarse record (~10 to 15 points)
        raw_curve = train_metrics["learning_curve"]
        step_interval = max(1, len(raw_curve) // 10)
        coarse_curve = [raw_curve[i] for i in range(0, len(raw_curve), step_interval)]
        if raw_curve and coarse_curve[-1]["step"] != raw_curve[-1]["step"]:
            coarse_curve.append(raw_curve[-1])

        results_data["our_ppo"][f"seed_{seed}"] = {
            "seed": seed,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "all_returns": all_rets,
            "training_steps": total_timesteps,
            "wall_time_seconds": round(wall_time, 2),
            "reload_exact_match": reload_exact_match,
            "passed": passed,
            "model_path": str(save_path),
            "coarse_learning_curve": coarse_curve,
        }

        table_rows.append({
            "impl": "Our PPO",
            "seed": seed,
            "eval_return": f"{mean_ret:.1f} +/- {std_ret:.1f}",
            "steps": f"{total_timesteps:,}",
            "wall_time": f"{wall_time:.1f}s",
            "reload": "Exact" if reload_exact_match else "Mismatch",
            "status": status_str,
        })

    # 2. Benchmark Stable-Baselines3 PPO
    print("\n--- Training Stable-Baselines3 PPO ---")
    for seed in seeds:
        print(f"\n[SB3 PPO] Starting Seed {seed}...")
        sb3_train_env = make_vec_env("CartPole-v1", n_envs=4, seed=seed)
        sb3_model = SB3PPO(
            policy="MlpPolicy",
            env=sb3_train_env,
            learning_rate=1e-3,
            n_steps=128,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            device="cpu",
            verbose=0,
        )

        sb3_start = time.perf_counter()
        sb3_model.learn(total_timesteps=total_timesteps)
        sb3_wall_time = time.perf_counter() - sb3_start
        sb3_train_env.close()

        eval_base_seed = 1000 + seed * 100
        mean_ret, std_ret, all_rets = evaluate_sb3(
            sb3_model, "CartPole-v1", n_eval_episodes=eval_episodes, base_seed=eval_base_seed
        )

        save_path = models_dir / f"sb3_ppo_seed_{seed}.zip"
        sb3_model.save(save_path)

        reloaded_sb3 = SB3PPO.load(save_path, device="cpu")
        reload_mean, reload_std, reload_rets = evaluate_sb3(
            reloaded_sb3, "CartPole-v1", n_eval_episodes=eval_episodes, base_seed=eval_base_seed
        )
        reload_exact_match = (all_rets == reload_rets)

        passed = bool(mean_ret >= pass_threshold and reload_exact_match)
        status_str = "PASS" if passed else "FAIL"

        print(f"[SB3 PPO] Seed {seed}: Return = {mean_ret:.1f} +/- {std_ret:.1f} | Time = {sb3_wall_time:.1f}s | Reload Match = {reload_exact_match} | {status_str}")

        results_data["sb3_ppo"][f"seed_{seed}"] = {
            "seed": seed,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "all_returns": all_rets,
            "training_steps": total_timesteps,
            "wall_time_seconds": round(sb3_wall_time, 2),
            "reload_exact_match": reload_exact_match,
            "passed": passed,
            "model_path": str(save_path),
        }

        table_rows.append({
            "impl": "SB3 PPO",
            "seed": seed,
            "eval_return": f"{mean_ret:.1f} +/- {std_ret:.1f}",
            "steps": f"{total_timesteps:,}",
            "wall_time": f"{sb3_wall_time:.1f}s",
            "reload": "Exact" if reload_exact_match else "Mismatch",
            "status": status_str,
        })

    overall_elapsed = time.perf_counter() - overall_start_time

    # Assemble and write final JSON results
    benchmark_payload = {
        "metadata": {
            "timestamp": timestamp,
            "task": "Phase 1 CartPole-v1 PPO Benchmark",
            "target_environment": "CartPole-v1",
            "total_timesteps_per_run": total_timesteps,
            "pass_threshold_return": pass_threshold,
            "eval_episodes": eval_episodes,
            "total_elapsed_seconds": round(overall_elapsed, 2),
            "python_version": "3.12.13",
            "torch_version": torch.__version__,
            "gymnasium_version": gym.__version__,
            "stable_baselines3_version": sb3.__version__,
            "device": "cpu",
        },
        "results": results_data,
    }

    results_path = runs_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Implementation':<12} | {'Seed':<5} | {'Eval Return (20 eps)':<22} | {'Steps':<8} | {'Time':<8} | {'Reload':<8} | {'Status':<6}")
    print("-" * 80)
    for row in table_rows:
        print(f"{row['impl']:<12} | {row['seed']:<5} | {row['eval_return']:<22} | {row['steps']:<8} | {row['wall_time']:<8} | {row['reload']:<8} | {row['status']:<6}")
    print("=" * 80)
    print(f"Total benchmark wall time: {overall_elapsed:.1f} seconds ({overall_elapsed / 60:.2f} minutes)")
    print(f"Benchmark results saved to: {results_path}")

    all_passed = all(row["status"] == "PASS" for row in table_rows)
    print(f"All runs passed: {all_passed}")
    return all_passed


if __name__ == "__main__":
    success = run_benchmark()
    if not success:
        raise SystemExit(1)

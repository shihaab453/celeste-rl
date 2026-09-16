"""Check that the RL environment works end to end on this machine.

Run with the RL interpreter:
    .venv-rl\\Scripts\\python.exe scripts\\check_env.py

1. Prints Python, PyTorch, Gymnasium and Stable-Baselines3 versions.
2. Runs a real calculation on the GPU and compares it to the CPU result,
   because torch.cuda.is_available() alone does not prove the GPU works.
3. Trains PPO on CartPole for a short while, as a smoke test that the whole
   stack runs. This is not a learning milestone; the score is only a sanity check.
"""
import platform
import sys
import time

import gymnasium as gym
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def check_versions():
    print(f"Python:            {sys.version.split()[0]} ({platform.system()})")
    print(f"PyTorch:           {torch.__version__} (CUDA build {torch.version.cuda})")
    print(f"Gymnasium:         {gym.__version__}")
    print(f"Stable-Baselines3: {stable_baselines3.__version__}")


def check_gpu():
    if not torch.cuda.is_available():
        print("\nGPU: CUDA not available, skipping GPU check")
        return False

    device = torch.device("cuda")
    print(f"\nGPU: {torch.cuda.get_device_name(device)}")
    print(f"     compute capability {torch.cuda.get_device_capability(device)}, "
          f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    # Same random matrices on both devices, so the results should match.
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(1024, 1024, generator=generator)
    b = torch.randn(1024, 1024, generator=generator)
    cpu_result = a @ b

    gpu_result = (a.to(device) @ b.to(device)).cpu()
    max_difference = (cpu_result - gpu_result).abs().max().item()
    print(f"     max |CPU - GPU| on a 1024x1024 matrix product: {max_difference:.2e}")

    # Float32 results differ slightly between devices; large differences mean a broken setup.
    assert max_difference < 1e-2, "GPU result does not match CPU result"
    print("     GPU calculation OK")
    return True


def smoke_test_cartpole(timesteps=20_000):
    print(f"\nCartPole smoke test: PPO for {timesteps:,} steps on CPU")
    # Monitor records true episode returns and lengths for evaluate_policy.
    env = Monitor(gym.make("CartPole-v1"))

    # SB3 recommends CPU for small non-convolutional policies like this one.
    model = PPO("MlpPolicy", env, device="cpu", seed=42, verbose=0)
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    elapsed = time.perf_counter() - start

    mean_return, std_return = evaluate_policy(model, env, n_eval_episodes=10)
    env.close()
    print(f"     trained in {elapsed:.1f} s ({timesteps / elapsed:,.0f} steps/s)")
    print(f"     mean return over 10 episodes: {mean_return:.1f} +/- {std_return:.1f} (max possible 500)")

    # A random policy scores about 20. Anything well above that shows learning happened.
    assert mean_return > 50, "PPO did not learn at all; the stack may be broken"
    print("     smoke test OK")


if __name__ == "__main__":
    check_versions()
    check_gpu()
    smoke_test_cartpole()
    print("\nAll checks passed.")

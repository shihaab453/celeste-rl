# Phase 1 Walkthrough: Understanding PPO from First Principles

This document provides an accessible walkthrough of the Proximal Policy Optimization (PPO) implementation in `celeste_rl/ppo.py`.

If you have already worked through the sine regression exercise in `experiments/sine_regression.py`, you already understand the core mechanics of PyTorch: defining neural network layers (`nn.Module`), executing forward passes, calculating a loss, running backpropagation (`loss.backward()`), and stepping an optimizer (`optimizer.step()`).

Reinforcement learning (RL) builds directly on those same mechanics, but with a different problem setup.

---

## 1. From Supervised Learning to Reinforcement Learning

In `sine_regression.py`, the training loop solved a supervised regression task:
1. Input $x$ was fed to `SineRegressor`.
2. The network produced predicted values.
3. The true target $y = \sin(x)$ was known in advance.
4. The loss was calculated using mean squared error: `(prediction - target) ** 2`.
5. Autograd computed gradients that steered predictions closer to the known targets.

In reinforcement learning, there is no supervisor providing target labels for each step:
- The agent observes a state $s$ from an environment (e.g. CartPole angles and velocities).
- The agent selects an action $a$ (e.g. push left or push right).
- The environment advances, returning a new state $s'$, a scalar reward $r$, and flags indicating whether the episode terminated or was truncated.
- Nobody tells the network which action was "correct". The agent only learns whether a sequence of actions produced high or low cumulative rewards over time.

---

## 2. Core Concepts Explained

### 2.1 Policy ($\pi$)
The policy is the "Actor". It represents the agent's strategy: given an observation $s$, what action should be taken?

For discrete actions (such as CartPole's 2 actions: left or right), the policy outputs raw scores called logits for each action. These logits are passed through a softmax function to produce a probability distribution:

$$\pi(a | s) = P(\text{action } a \mid \text{state } s)$$

In `celeste_rl/ppo.py`, this is implemented inside the `ActorCritic.actor` network:
- Two linear layers with `nn.Tanh` activations (matching the architecture from `sine_regression.py`).
- Evaluated via `torch.distributions.Categorical(logits=logits)`.
- During training, actions are sampled from this distribution to explore different behaviors.
- During evaluation, the greedy action with the highest probability is selected (`logits.argmax(dim=-1)`).

### 2.2 Cumulative Return and Discounting ($\gamma$)
A trajectory is a sequence of interactions:

$$(s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)$$

The total undiscounted return from time step $t$ is the sum of rewards:

$$G_t = \sum_{k=0}^{T - t - 1} r_{t+k}$$

In practice, rewards received far in the future are discounted using a discount factor $\gamma \in (0, 1)$ (typically $\gamma = 0.99$):

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

Why discount?
1. Mathematical convergence: it bounds returns over long or infinite time horizons.
2. Preference for sooner rewards: a reward received immediately is preferable to a reward delayed into the distant future.
3. Uncertainty: future state transitions become increasingly uncertain as time progresses.

### 2.3 Value Function ($V$)
The value function is the "Critic". It predicts the expected cumulative discounted return starting from state $s$:

$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid s_t = s]$$

In `celeste_rl/ppo.py`, this is implemented inside `ActorCritic.critic`:
- A separate MLP with two hidden layers and `nn.Tanh` activations.
- The critic is essentially a standard regressor, just like `SineRegressor` in `sine_regression.py`.
- Its regression targets are the empirical returns computed by GAE, and its loss is standard mean squared error:

$$L^{VF} = \frac{1}{2} (V_\theta(s_t) - R_t)^2$$

### 2.4 Advantage ($A$)
The value function $V(s)$ tells us how good a state is on average. However, when deciding whether to encourage or discourage a specific action $a$, we need to know: was this specific action better or worse than the average action in state $s$?

This difference is the Advantage function:

$$A(s, a) = Q(s, a) - V(s)$$

Where $Q(s, a)$ is the expected return from taking action $a$ in state $s$.
- If $A(s, a) > 0$: action $a$ turned out better than average. We should increase its probability.
- If $A(s, a) < 0$: action $a$ turned out worse than average. We should decrease its probability.

### 2.5 Generalized Advantage Estimation (GAE)
How do we calculate advantage from empirical rollouts?
- A simple 1-step temporal difference (TD) error:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

  This has low variance because it only relies on one step of real reward, but it has bias because $V$ is an imperfect neural network estimate.
- A full Monte Carlo return ($G_t - V(s_t)$) has zero bias, but high variance across different episodes.

GAE blends these approaches using an exponential weighting parameter $\lambda \in [0, 1]$ (here $\lambda = 0.95$):

$$A_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

In `celeste_rl/ppo.py`, this is implemented in `compute_gae()` using a clean backwards loop:

$$A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}$$

Where $d_t$ is a boolean flag indicating whether the episode ended at step $t$.

### 2.6 Termination vs Truncation
Gymnasium distinguishes two ways an episode can end:
1. **Termination (`terminated = True`)**: The task ended naturally according to the environment rules. In CartPole-v1, the pole tilted beyond 12 degrees or the cart moved off the track. The agent failed, so there are no future rewards. The value of the next state is strictly $0.0$.
2. **Truncation (`truncated = True`)**: An artificial time limit was reached (CartPole-v1 truncates at 500 steps). The pole was still upright and the cart was still balanced! Because the task did not end, we must bootstrap future value from the final observation: $V(s_{t+1}) = V(\text{final\_obs})$.

Both flags can be true on the same step, for example when the pole falls on exactly the 500th step. The task really ended, so termination wins and the next value is $0.0$. In Celeste this matters: a death on the time-limit frame is still a death.

In both termination and truncation, $(1 - d_t) = 0$ in the recursive advantage equation: advantages from subsequent steps belonging to a new episode must never propagate backwards across the episode boundary.

### 2.7 Gymnasium 1.3.0 Vector Autoreset Behavior
When using vectorized environments in Gymnasium 1.3.0 (`SyncVectorEnv` created via `gym.make_vec`):
- By default, Gymnasium sets `autoreset_mode = AutoresetMode.NEXT_STEP`.
- In `NEXT_STEP` mode, when an episode ends at step $t$, the environment flags that sub-environment for an autoreset on the next step ($t+1$).
- On step $t+1$, the environment discards whatever action the policy selected, resets the sub-environment, and returns reward 0.0 and the reset observation.
- If a rollout collection loop naively recorded step $t+1$, it would store a transition that spans across the episode boundary (from the old terminal state to the new initial state) with an ignored action and dummy zero reward.

To prevent this, `celeste_rl/ppo.py` builds the vector environment with `vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP}`:
- The reset occurs immediately inside step $t$.
- The finished episode's true terminal state is preserved in `infos["final_obs"]` (which we use for truncation bootstrapping).
- The returned `next_obs` contains the clean starting state for the new episode.
- Step $t+1$ executes a genuine action on that initial state.
- Every single transition stored in the rollout buffer is a genuine environment step; no transitions span episode boundaries or contain autoreset artifacts.

### 2.8 Probability Ratio and Log-Probabilities
Probabilities multiply along trajectories. Because multiplying small numbers quickly causes numerical underflow, neural networks compute log-probabilities: $\log \pi(a | s)$.

To evaluate how much the policy changed between the rollout phase (under parameters $\theta_{\text{old}}$) and the current update phase (under parameters $\theta$), PPO calculates the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\right)$$

At the start of an update before any gradient steps have been taken, $\theta = \theta_{\text{old}}$, so $r_t(\theta) = 1.0$ exactly.

### 2.9 The Clipped Surrogate Objective
In standard policy gradient, if the policy updates too aggressively, the probability ratio $r_t(\theta)$ can become large, causing the policy to collapse into degenerate states from which it cannot recover.

PPO prevents this by clipping the ratio to an interval $[1 - \epsilon, 1 + \epsilon]$ (typically $\epsilon = 0.2$):

$$L^{\text{CLIP}}(\theta) = -\mathbb{E}\left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

- If advantage $\hat{A}_t > 0$: The action was good. The objective encourages increasing its probability, but clipping stops rewarding further increases once the ratio exceeds $1 + \epsilon = 1.2$.
- If advantage $\hat{A}_t < 0$: The action was bad. The objective encourages decreasing its probability, but clipping stops penalizing once the ratio drops below $1 - \epsilon = 0.8$.
- The negative sign converts this into a minimization loss suitable for standard PyTorch optimizers (`optimizer.step()`).

### 2.10 Entropy Bonus
The entropy of a distribution measures its randomness:

$$H(\pi) = -\sum_{a} \pi(a \mid s) \log \pi(a \mid s)$$

- A uniform distribution (50% left, 50% right) has maximal entropy ($\approx 0.693$ nats for 2 actions).
- A deterministic distribution (100% left, 0% right) has zero entropy.

Early in training, we want the agent to explore both actions rather than prematurely committing to one choice. Subtracting an entropy bonus from the loss:

$$\text{Loss} = L^{\text{CLIP}} + c_1 L^{\text{VF}} - c_2 H(\pi)$$

encourages the optimizer to keep the distribution broad until strong reward gradients dictate a preference.

### 2.11 Epochs and Minibatches
In classic policy gradient methods (like REINFORCE or A2C), rollout data can only be used for a single gradient update, because after one step the policy has changed and the data is no longer on-policy.

Because PPO's clipped objective actively prevents the new policy from moving too far from $\pi_{\text{old}}$, PPO can safely reuse rollout data across several epochs (e.g. 4 epochs) of mini-batches (e.g. 4 mini-batches per epoch). This improves sample efficiency by extracting more learning signal from each collected step.

---

## 3. Code Tour of `celeste_rl/ppo.py`

| Component | Function / Class | Purpose |
|---|---|---|
| Model Architecture | `ActorCritic` | Houses separate actor and critic MLPs with orthogonal weight initialization. |
| Action & Value Queries | `get_action_and_value()` | Evaluates observations, samples actions, and computes log-probabilities and entropy. |
| Evaluation Action | `get_greedy_action()` | Selects the highest probability action without exploration noise. |
| Checkpoint Persistence | `save()` and `load()` | Saves and reloads model weights along with the configuration dictionary required to reconstruct the architecture. |
| Advantage Calculation | `compute_gae()` | Backward recursion computing GAE advantages and target returns with termination and truncation handling. |
| Clipped Loss Function | `compute_policy_loss()` | Calculates probability ratios and the clipped surrogate objective. |
| Hyperparameters | `PPOConfig` | Clean dataclass holding all rollout, training, and model parameters. |
| Training Loop | `train_ppo()` | Manages vectorized environments, rollout collection, advantage normalization, mini-batch shuffling, and optimizer updates. |
| Policy Evaluation | `evaluate()` | Runs deterministic evaluation episodes and reports mean and standard deviation of returns. |

---

## 4. Benchmark Results on CartPole-v1

The implementation was validated against Stable-Baselines3 using `scripts/phase1_cartpole.py`. Both implementations were trained across seeds 0, 1, and 2 on CPU for 100,000 steps, evaluated over 20 fresh episodes with greedy actions, saved to disk, and reloaded to confirm evaluation determinism.

Pass criterion: mean evaluation return $\ge 475.0$ out of a maximum possible 500.0.

### Benchmark Results Table (Run `20260917-020124`)

| Implementation | Seed | Evaluation Return (20 eps) | Total Steps | Wall Time | Reload Match | Status |
|---|---|---|---|---|---|---|
| **Our PPO** | 0 | 500.0 +/- 0.0 | 100,000 | 17.7s | Exact | **PASS** |
| **Our PPO** | 1 | 500.0 +/- 0.0 | 100,000 | 16.8s | Exact | **PASS** |
| **Our PPO** | 2 | 500.0 +/- 0.0 | 100,000 | 16.8s | Exact | **PASS** |
| **SB3 PPO** | 0 | 500.0 +/- 0.0 | 100,000 | 15.8s | Exact | **PASS** |
| **SB3 PPO** | 1 | 500.0 +/- 0.0 | 100,000 | 15.7s | Exact | **PASS** |
| **SB3 PPO** | 2 | 500.0 +/- 0.0 | 100,000 | 15.9s | Exact | **PASS** |

### Summary Observations
1. **Convergence and Score:** Both Our PPO and Stable-Baselines3 achieved a perfect score of 500.0 across all three random seeds.
2. **Execution Speed:** Our single-file implementation trained 100,000 steps in approximately 17 seconds on CPU, directly comparable to Stable-Baselines3 (~16 seconds). Total benchmark wall time across all 6 runs was 112.2 seconds (under 2 minutes).
3. **Save and Reload Verification:** Reloading each saved model from disk reproduced the exact evaluation returns frame-for-frame across all 20 test episodes for every seed.
4. **Structured Artifacts:** Full metrics, coarse learning curves, and model checkpoints are preserved in `runs/phase1/20260917-020124/results.json`.

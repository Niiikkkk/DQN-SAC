# Soft Actor-Critic (SAC) Agent — Code Explanation

This document explains the implementation of the Soft Actor-Critic agent found at `src/agents/sac_agent.py`.

Checklist
- File overview and purpose
- Constructor (what the key arguments configure)
- Important internal components (actor, critics, target critics, optimizers)
- Key methods and their roles (get_action, critic/target_critic, update_critic, entropy, actor loss, update_alpha, target updates, update)
- Notes about temperature (alpha) auto-tuning and multi-critic backups

1) High-level overview
- The `SoftActorCritic` class implements a typical SAC agent: an actor (stochastic policy) and one or more critic networks (Q-functions). The agent supports multiple critic networks (ensemble), target critics, hard or soft target updates, optional entropy regularization, and automatic temperature (alpha) tuning.

2) Constructor and configuration
- actor: created via `make_actor(observation_shape, action_dim)`. It must return a distribution object when called with observations (e.g., a module that returns a `torch.distributions.Distribution`).
- actor_optimizer and actor_lr_scheduler: created using `make_actor_optimizer` and `make_actor_schedule`.
- critics: stored in `self.critics` (ModuleList). Each critic should return Q-values when called with (obs, action).
- target_critics: a copy of critics used to form stable targets.
- num_critic_networks: ensemble size. When >1, the implementation supports both the "mean" backup (average Qs) and the "min" backup (clipped double-Q).
- discount: gamma for TD backups.
- target_update_period / soft_target_update_rate: choose either hard target updates every N steps, or soft updates with rate tau.
- use_entropy_bonus / temperature / backup_entropy: control whether and how entropy is used in targets and losses.
- auto_tune_temperature: if True, initialize `log_alpha` (a learnable parameter), `alpha_optimizer` and `target_entropy`. `log_alpha` is initialized to log(temperature) so learning starts at the provided temperature.

3) Core components
- `actor(obs)` returns a distribution; the implementation uses reparameterized sampling (`rsample`) for actor gradient updates.
- `critic(obs, action)`: stacks outputs from all critic networks and returns a tensor of shape (num_critics, batch_size).
- `target_critic(obs, action)`: same as above but uses target networks.

4) Multi-critic backup strategy (`q_backup_strategy`)
- Accepts `next_qs` shaped (num_critics, batch_size) and returns shaped (num_critics, batch_size).
- If `target_critic_backup_type == 'mean'`, averages across critics producing a single value per sample, then expands that value back to shape (num_critics, batch_size) so each critic has the same target.
- If `target_critic_backup_type == 'min'`, takes the elementwise minimum across critics (clipped double-Q). That reduces overestimation bias.

5) Critic update (`update_critic`)
- Goal: minimize MSE between predicted Q(s,a) and TD target.
- Steps:
  - Use the actor to sample the next action distribution at `next_obs`, sample an action (reparameterized sampling not required for targets, but the code uses `.sample()` in the actor call inside `update_critic`).
  - Compute target Q-values from `self.target_critics` using `target_critic(next_obs, next_action)`, resulting in shape (num_critics, batch_size).
  - If `use_entropy_bonus` and `backup_entropy`, add temperature * entropy(next_action_distribution) to next Qs. Entropy is computed per sample and broadcasted per critic.
  - Reduce/aggregate across multiple target critics using `q_backup_strategy`.
  - Compute TD target: target = reward + discount * (1 - done) * next_qs.
  - Compute predicted Q-values for the current (obs, action) via `self.critic(obs, action)` which returns shape (num_critics, batch_size).
  - Minimize MSE loss between predicted and target values. The code calls .backward() and steps `self.critic_optimizer`.

Notes:
- Targets and predictions are shaped (num_critics, batch_size). When `q_backup_strategy` reduced the first dimension, the code expands it to keep the same shape so the same target is applied to each critic.

6) Entropy computation (`entropy`)
- Returns an estimate of entropy per batch element: -log pi(a|s) for actions sampled from the policy. The implementation uses `rsample()` and `log_prob()` so gradients are available when needed (but in target computation, entropy is used with no_grad).

7) Actor update and reparameterization (`actor_loss_reparametrize`, `update_actor`)
- `actor_loss_reparametrize`:
  - Calls `self.actor(obs)` to get a distribution.
  - Samples actions using `rsample()` so gradients flow through the sampling (reparameterization trick).
  - Computes Q-values via `self.critic(obs, action)` giving shape (num_critics, batch_size). The implementation averages across critics with `q_values.mean(dim=0)` when computing the actor objective.
  - Computes log probabilities `log_prob = action_distribution.log_prob(action)` which are used for alpha updates.
  - Actor loss (surrogate): E[ alpha * log_prob - Q(s,a) ]. The code returns this scalar loss and also returns the mean entropy for logging and `log_prob` (detached in later steps) for alpha updates.
- `update_actor`:
  - Takes the loss from `actor_loss_reparametrize`, subtracts an entropy bonus term (the code uses `loss -= entropy`), then does a gradient step on the actor optimizer.

Design notes:
- The code uses both temperature scaling inside the actor loss and an additional direct entropy bonus subtraction in `update_actor`. Depending on configuration, one might instead only include the alpha-scaled log_prob term and not subtract a separate entropy bonus. In this code both are present, which increases the effective entropy encouragement. Check your desired algorithm variant.

8) Temperature (alpha) auto-tuning (`update_alpha`, `get_temperature`)
- If `auto_tune_temperature` is True, `log_alpha` is a learnable parameter initialized to log(temperature). The optimizer `alpha_optimizer` updates `log_alpha`.
- `get_temperature()` returns `exp(log_alpha)` when tuning, otherwise returns the fixed `self.temperature` value.
- `update_alpha(log_prob)` implements dual gradient descent with loss = -alpha * (log_prob + target_entropy). Minimizing this encourages alpha to increase when policy entropy is too low (log_prob < -target_entropy) and decrease otherwise. The code averages over the batch.

9) Target networks and updates
- `update_target_critic()` performs a hard copy by calling `soft_update_target_critic(1.0)`.
- `soft_update_target_critic(tau)` mixes parameters of target and main critics: target = (1 - tau) * target + tau * main. If `target_update_period` is configured, `update()` performs a full hard update every `target_update_period` steps; otherwise performs a soft update each call using `soft_target_update_rate`.

10) Top-level `update(...)` method
- Runs `num_critic_updates` steps of critic updates (good for training stability).
- Runs a single `update_actor` step (actor update is controlled by flags relating to entropy usage; the code calls it unconditionally once implemented).
- If enabled, runs `update_alpha` using detached log probabilities from the actor update.
- Performs either hard or soft target updates depending on configuration.
- Steps learning rate schedulers for actor and critic.

11) Logging and returned values
- Many methods return dictionaries with scalar statistics (losses, mean Qs, target values, entropy, alpha) useful for logging.

12) Practical notes / things to watch
- Actor output: make sure the returned distribution object's `rsample()` and `log_prob()` shapes are consistent with action_dim (the code asserts shapes in several places).
- Entropy aggregation and sign conventions: in SAC the objective uses alpha * log_prob - Q; because log_prob is typically negative, multiplying by alpha adds a penalty proportional to negative log_prob. The implementation here also subtracts `entropy` from the actor loss—double-check if you want both terms.
- If using multiple critics, be careful about how targets are aggregated (mean vs min) as this changes bias/variance trade-offs.

References
- Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. 2018.

---
File: `src/agents/sac_agent.py` was used as the reference for this explanation.


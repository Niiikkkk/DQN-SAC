# Deep Q-Network (DQN) Agent — Code Explanation

This document explains the implementation of the DQN agent found at `src/agents/dqn_agent.py`.

Checklist
- File overview and purpose
- Constructor (parameters and their effect)
- Key methods (get_action, update_critic, update_target_critic, update)
- Double Q-learning explanation and implementation details

1) High-level overview
- The `DQNAgent` class implements a tabular/approximate Q-learning agent where the critic is a neural network mapping observations to Q-values for each discrete action. It supports an online critic and a separate target critic, learning via MSE on TD targets. The agent optionally supports Double Q-Learning, gradient clipping, and learning rate scheduling.

2) Constructor and important attributes
- `make_critic(observation_shape, num_actions)`: constructs the Q-network with output shape [batch_size, num_actions].
- `target_critic`: a second network with the same architecture used to compute stable TD targets.
- `critic_optimizer` and `lr_scheduler`: optimizer and scheduler for critic updates.
- `discount`: gamma used in TD target computation.
- `target_update_period`: number of steps between hard copies of `critic` weights to `target_critic`.
- `use_double_q`: if True, apply Double Q-Learning target selection to reduce overestimation.
- `clip_grad_norm`: maximum gradient norm for clipping (if None, no clipping).

3) Action selection (`get_action`)
- The method implements epsilon-greedy selection. Steps:
  - Forward pass: `actions = self.critic(observation)` yielding a Q-vector.
  - With probability 1 - epsilon choose the greedy action: argmax over Q-values.
  - With probability epsilon choose a random integer action in [0, num_actions).

Implementation notes:
- The code builds a batch dimension by adding a leading `None` to the observation: `observation[None]`.
- The returned value is an integer (single action index).

4) Critic update (`update_critic`)
- Purpose: minimize MSE between the Q-value for the taken action and a TD target.
- Steps in detail:
  - Compute `next_qa_values` = `self.target_critic(next_obs)` which yields Q-values for each action under next observations using the target network.
  - If `use_double_q` is True:
    - Choose the next action using the online critic (self.critic) on next_obs: `next_action = argmax(self.critic(next_obs))`.
    - Evaluate the Q-value of that action using the target critic: gather the corresponding values from `next_qa_values`.
  - Else (standard DQN): select the greedy action using `target_critic`'s Q-values: `next_action = argmax(next_qa_values)` and gather the max Q-value.
  - Compute target: target = reward + discount * (1 - done) * next_q_value
  - Compute predicted Q-values for the taken actions: `qa_values = self.critic(obs)` then `q_values = qa_values.gather(dim=1, index=action.unsqueeze(1)).squeeze(1)`.
  - Compute MSE loss: `loss = MSE(q_values, target_values)`.
  - Backpropagate, optionally clip gradients via `torch.nn.utils.clip_grad.clip_grad_norm_`, and step the optimizer.
  - Step the learning rate scheduler.

Returned stats include: `critic_loss`, mean `q_values`, mean `target_values`, and `grad_norm`.

5) Double Q-learning (why and how)
- Motivation: Standard DQN uses the same network to select and evaluate the maximizing action for the target, which can introduce overestimation bias. Double Q-Learning decouples action selection from evaluation by selecting the action with the online network and evaluating it with the target network.
- Implementation in this code: when `use_double_q` is True, `next_action` is computed as `argmax(self.critic(next_obs))` but the Q-value for that action is taken from `next_qa_values` computed by `self.target_critic`.

6) Target network updates (`update_target_critic` and `update`)
- `update_target_critic()` copies the online critic weights to the target via `load_state_dict`.
- `update(...)` performs a critic update (calls `update_critic`) and then, if `step % self.target_update_period == 0`, calls `update_target_critic()` to perform a hard update.

7) Practical implementation notes
- Shapes: The code expects `reward` shaped (batch_size,) and actions shaped (batch_size,) (integer indices). Many `.gather()` calls and `unsqueeze(1)`/`squeeze(1)` ensure correct shapes.
- Gradient clipping: `clip_grad_norm` prevents exploding gradients for stabilizing training.
- Make sure `make_critic` returns a network whose forward returns a 2D tensor [batch_size, num_actions].

8) Logging and hyperparameters
- Returned dict from `update_critic` is suitable for logging training progress.
- Choose `target_update_period`, learning rates, and whether to use Double Q carefully: Double Q reduces overestimation bias and often improves stability.

References
- Mnih et al., Human-level control through deep reinforcement learning (DQN), 2015.
- Hasselt, G. van, Double Q-learning, 2010 (and Double DQN later).

---
File: `src/agents/dqn_agent.py` was used as the reference for this explanation.


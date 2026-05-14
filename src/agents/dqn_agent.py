from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(Section 2.4): get the action from the critic using an epsilon-greedy strategy
        actions = self.critic(observation)
        action = torch.argmax(actions,dim=-1) if np.random.rand() < 1-epsilon else torch.randint(0, self.num_actions, ())
        # ENDTODO

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(Section 2.4): compute target values
            #All actions from the next state
            next_qa_values = self.target_critic(next_obs)
            if self.use_double_q:
                # TODO(Section 2.5): implement double-Q target action selection
                # In double Q we use a network to get the action and another one to compute the QA.
                
                # We pick the action with critic (which is the one we update each iteration),
                #then below we take the q based on next_qa_values, which is calculated from
                #target_criti which is the one that is updated only N steps.
                next_action = torch.argmax(self.critic(next_obs),dim=-1)
            else:
                #without double Q we use target_critic to get the QA values, and then we take the max action from those QA

                #max will return two arrays. First is the max values, second in the index of the max value
                # This is equals to do torch.argmax(next_qa_values, dim = -1)
                next_action = torch.max(next_qa_values,dim=-1)[1]

            #gather will pick the values in next_qa_values that correspond to next_action index
            # IN case where use_double_q = False this is equals to torch.max(next_qa_values,dim=-1)[0]
            # but in case of double_q the max woulnd't be good, since the network for computing the next_action is different! So use gather


            #So next_qa_values have a value for each action
            # next_q_values has only the q value for that specific action (the best)
            next_q_values = next_qa_values.gather(dim=1, index=next_action.unsqueeze(1)).squeeze(1)
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward + self.discount*(1-done.float())*next_q_values
            assert target_values.shape == (batch_size,), target_values.shape
            # ENDTODO

        # TODO(Section 2.4): train the critic with the target values
        #self.critic outputs [batch_size, actions], so those are qa values, then with gather we take the values corresponding to the actions took
        # exactly as we have done for the next_q_values
        qa_values = self.critic(obs)
        q_values = qa_values.gather(dim=1, index=action.unsqueeze(1)).squeeze(1)
        loss = self.critic_loss(q_values, target_values)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(Section 2.4): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()
        # Hint: if step % self.target_update_period == 0: ...
        # ENDTODO

        return critic_stats

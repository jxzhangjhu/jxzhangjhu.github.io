"""d10 · Debug a GRPO training loop — budget 30 min

The most reported ML-coding question at Anthropic, and it also shows up at OpenAI.
A complete GRPO script that runs end to end without raising — that is the trap. Two of
the bugs are numerical (you find them by running it), one is algorithmic (you find it by
reading the ratio formula against the objective you know).

The interviewer then asks the real question, which is not about the code:

    This loop is strictly on-policy — rollouts come from the policy being updated.
    So why is the mean importance ratio not exactly 1?

Have an answer ready before you look at `hints/`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, n_actions)
        )

    def forward(self, obs):
        return self.net(obs)                       # raw logits


def rollout(policy, obs, generator=None):
    """Sample one action per observation and return (actions, logprobs)."""
    logits = policy(obs)
    probs = F.softmax(logits, dim=-1)                     # FIX 1
    actions = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    logprobs = F.log_softmax(logits, dim=-1).gather(1, actions[:, None]).squeeze(-1)
    return actions, logprobs


def compute_advantage(rewards, group_size):
    """Group-relative advantage: standardise the reward within each prompt's group."""
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-5)  # FIX 2
    return adv.reshape(-1)


def grpo_loss(logprobs, old_logprobs, advantages, clip_eps=0.2):
    """Clipped surrogate objective."""
    ratio = (logprobs - old_logprobs).exp()               # FIX 3
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def train(policy, opt, obs, reward_fn, group_size=4, steps=5, generator=None):
    """One rollout batch per step, one optimiser step per batch (strictly on-policy)."""
    history = []
    for _ in range(steps):
        actions, old_logprobs = rollout(policy, obs, generator)
        rewards = reward_fn(obs, actions)
        advantages = compute_advantage(rewards, group_size)

        logits = policy(obs)
        logprobs = F.log_softmax(logits, dim=-1).gather(1, actions[:, None]).squeeze(-1)

        loss = grpo_loss(logprobs, old_logprobs.detach(), advantages.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history

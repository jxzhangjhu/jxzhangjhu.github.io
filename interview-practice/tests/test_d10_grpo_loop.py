"""Tests for d10 · Debug a GRPO training loop. Run: python -m pytest tests/test_d10_grpo_loop.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d10_grpo_loop as g  # noqa: E402


def test_sampling_follows_the_policy_distribution():
    torch.manual_seed(0)
    policy = g.Policy(obs_dim=2, n_actions=3)
    with torch.no_grad():                      # a policy with a known, skewed answer
        policy.net[2].weight.zero_()
        policy.net[2].bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
    obs = torch.zeros(4000, 2)
    actions, _ = g.rollout(policy, obs)
    emp = torch.bincount(actions, minlength=3).float() / actions.numel()
    want = F.softmax(torch.tensor([2.0, 0.0, -2.0]), dim=-1)
    assert torch.allclose(emp, want, atol=0.02), (
        f"sampled {emp.tolist()} but the policy distribution is {want.tolist()}")


def test_advantage_is_finite_for_tied_and_singleton_groups():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    adv = g.compute_advantage(rewards, group_size=4)
    assert torch.isfinite(adv).all(), f"advantage contains {adv.tolist()} for a tied group"
    singleton = g.compute_advantage(torch.tensor([1.0, 0.0]), group_size=1)
    assert torch.isfinite(singleton).all() and torch.equal(singleton, torch.zeros(2))


def test_on_policy_surrogate_equals_negative_mean_advantage():
    lp = torch.tensor([-1.2, -0.7, -2.0, -0.3])
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss = g.grpo_loss(lp, lp.clone(), adv)      # new == old: ratio is exactly 1
    assert torch.allclose(loss, -adv.mean(), atol=1e-6), (
        f"on-policy loss should be {-adv.mean():.4f} but is {loss:.4f}")


def test_end_to_end_training_is_finite_and_learns():
    """All three fixed: the loop runs on a tied-reward batch and moves toward the reward."""
    torch.manual_seed(0)
    policy = g.Policy(obs_dim=4, n_actions=3)
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)
    # Four sampled completions per prompt, laid out contiguously as the loop expects.
    obs = torch.randn(8, 4).repeat_interleave(4, dim=0)

    def reward_fn(o, a):                          # action 1 is always correct
        return (a == 1).float()

    history = g.train(policy, opt, obs, reward_fn, group_size=4, steps=40)
    assert all(torch.isfinite(torch.tensor(h)) for h in history), \
        f"loss went non-finite: {history}"
    with torch.no_grad():
        p = F.softmax(policy(obs), dim=-1).mean(0)
    assert p[1] > 0.5, (
        f"after 40 steps the policy puts only {p[1]:.2f} on the rewarded action; "
        "it is not learning")

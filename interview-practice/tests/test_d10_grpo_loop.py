"""Tests for d10 · Debug a GRPO training loop. Run: python -m pytest tests/test_d10_grpo_loop.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d10_grpo_loop as g  # noqa: E402


def test_bug1_sampling_follows_the_policy_distribution():
    """Actions must be sampled from softmax(logits), not from the raw logits.

    torch.multinomial treats its input as unnormalised *weights*, so feeding logits
    silently samples from the wrong distribution — and any negative logit raises or
    distorts. The script still runs, which is why this survives review.
    """
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
        f"sampled {emp.tolist()} but the policy says {want.tolist()}: "
        "is multinomial being fed logits instead of probabilities?")


def test_bug2_advantage_survives_a_tied_group():
    """When every completion in a group earns the same reward the std is 0.

    This is not a corner case — it is the common case early and late in training, when
    the policy fails or solves every sample of a prompt. Dividing by a bare std gives
    NaN, which then silently poisons every parameter on the next step.
    """
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    adv = g.compute_advantage(rewards, group_size=4)
    assert torch.isfinite(adv).all(), (
        f"advantage contains {adv.tolist()}: a tied group divided by std = 0")


def test_bug3_ratio_is_an_exponential_not_a_difference():
    """The importance ratio is exp(new - old). A raw log difference is not a ratio.

    The tell: with new == old the ratio must be exactly 1, so an unclipped surrogate
    equals the advantage. A log difference gives 0 there, and the whole objective
    collapses to zero gradient at the on-policy point.
    """
    lp = torch.tensor([-1.2, -0.7, -2.0, -0.3])
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss = g.grpo_loss(lp, lp.clone(), adv)      # new == old: ratio is exactly 1
    assert torch.allclose(loss, -adv.mean(), atol=1e-6), (
        f"at ratio 1 the loss should be {-adv.mean():.4f} but is {loss:.4f}: "
        "the ratio is probably a log difference rather than exp(log difference)")


def test_end_to_end_training_is_finite_and_learns():
    """All three fixed: the loop runs on a tied-reward batch and moves toward the reward."""
    torch.manual_seed(0)
    policy = g.Policy(obs_dim=4, n_actions=3)
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)
    obs = torch.randn(32, 4)

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

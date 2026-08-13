"""Prove every test is satisfiable, by running it against the reference solution.

A test that no correct implementation can pass is worse than no test: you would burn
your practice time debugging the harness. This swaps a reference-backed adapter in for
each stub, runs the suite, then puts the stubs back.

    python _validate.py            # manifest + stub tests against the reference
    python _validate.py --drills   # manifest + drill tests against the fixed drill

Drill tests are excluded from the default run: they are supposed to fail while the
planted bugs are still there.
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUBS = HERE / "stubs"
sys.path.insert(0, str(HERE))
from problems import PROBLEMS, DRILLS  # noqa: E402

# Reference code sometimes exposes a different (older) API than the stub asks for, so a
# few problems need a thin adapter rather than a straight re-export.
ADAPTERS = {
"p01_mha": "from reference import CausalSelfAttention  # noqa: F401\n",

"p02_kv_cache": '''
import torch  # noqa: F401
from reference import GroupedQueryAttention


class CachedAttention(GroupedQueryAttention):
    """Reference GQA mutates a dict in place; the stub API passes the cache explicitly."""

    def __init__(self, d_model, n_heads):
        super().__init__(d_model, n_heads, n_heads)

    def forward(self, x, cache=None):
        d = {} if cache is None else {"k": cache[0], "v": cache[1]}
        y = super().forward(x, d)
        return y, (d["k"], d["v"])
''',

"p04_rope": "from reference import rope_cache, apply_rope  # noqa: F401\n",
"p05_rmsnorm": "from reference import RMSNorm  # noqa: F401\n",
"p08_cross_entropy": "from reference import cross_entropy  # noqa: F401\n",
"p10_training_loop": "from reference import overfit_tiny  # noqa: F401\n",
"p14_sampling": "from reference import sample_next  # noqa: F401\n",

"p18_lora": "from reference import LoRALinear  # noqa: F401\n",

"p22_bpe": "from reference import bpe_train, bpe_encode  # noqa: F401\n",

"p19_grpo_loss": '''
from reference import grpo_loss as _ref


def grpo_loss(logp, logp_old, logp_ref, rewards, mask, group_size,
              clip_eps=0.2, beta=0.04):
    """The stub takes group_size positionally; the reference takes it last as a kwarg."""
    return _ref(logp, logp_old, logp_ref, rewards, mask,
                clip_eps=clip_eps, beta=beta, group_size=group_size)
''',
}


def validate_manifest():
    """Every metadata row must have a runnable stub, reset copy, test, and three hints."""
    errors = []
    for p in PROBLEMS:
        stem = f"{p.id}_{p.name}"
        expected = [
            STUBS / f"{stem}.py",
            STUBS / ".pristine" / f"{stem}.py",
            HERE / "tests" / f"test_{stem}.py",
            HERE / "hints" / f"{stem}.md",
        ]
        errors.extend(str(path.relative_to(HERE)) for path in expected if not path.exists())
        hint = expected[-1]
        if hint.exists():
            text = hint.read_text(encoding="utf-8")
            levels = [text.count(f"## Level {i}") for i in range(1, 4)]
            if levels != [1, 1, 1]:
                errors.append(f"{hint.relative_to(HERE)}: expected exactly Levels 1, 2, 3")

    for d in DRILLS:
        stem = f"{d.id}_{d.name}"
        expected = [
            HERE / "drills" / f"{stem}.py",
            HERE / "drills" / ".pristine" / f"{stem}.py",
            HERE / "drills" / ".solutions" / f"{stem}.py",
            HERE / "tests" / f"test_{stem}.py",
            HERE / "hints" / f"{stem}.md",
        ]
        errors.extend(str(path.relative_to(HERE)) for path in expected if not path.exists())
        hint = expected[-1]
        if hint.exists():
            text = hint.read_text(encoding="utf-8")
            levels = [text.count(f"## Level {i}") for i in range(1, 4)]
            if levels != [1, 1, 1]:
                errors.append(f"{hint.relative_to(HERE)}: expected exactly Levels 1, 2, 3")

    if errors:
        print("manifest errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"manifest: {len(PROBLEMS)} problems and {len(DRILLS)} drills are complete")
    return 0


def validate_drills():
    """Each drill must be solvable: swap in .solutions/ and the whole file must pass."""
    rc = 0
    for sol in sorted((HERE / "drills" / ".solutions").glob("d*.py")):
        buggy = HERE / "drills" / sol.name
        bak = buggy.with_suffix(".bak")
        shutil.copy(buggy, bak)
        try:
            shutil.copy(sol, buggy)
            name = sol.stem
            t = HERE / "tests" / f"test_{name}.py"
            print(f"\n--- {name} (fixed) ---")
            rc |= subprocess.run(
                [sys.executable, "-m", "pytest", str(t), "-q", "--no-header"], cwd=HERE
            ).returncode
        finally:
            shutil.copy(bak, buggy)
            bak.unlink()
    print("\nbuggy drills restored")
    print("every drill is solvable" if rc == 0 else "!! a drill cannot be solved")
    return rc

# The generated problems re-export the reference symbol directly; the hand-written ones
# above need an adapter because their stub API differs from the older reference API.
for _pid, _sym in {
    "p03_gqa": "GroupedQueryAttention", "p06_swiglu": "SwiGLU",
    "p28_mla": "MultiHeadLatentAttention",
    "p07_transformer_block": "Block",
    "p09_loss_masking": "build_sft_labels, build_packed_sft_labels, build_packed_attention",
    "p11_autograd": "Value", "p12_attention_backward": "attention_backward",
    "p13_mlp_backward": "mlp_backward", "p15_speculative": "speculative_accept",
    "p16_online_softmax": "online_softmax_weighted_sum",
    "p17_flash_attention": "flash_attention_forward", "p20_dpo_loss": "dpo_loss",
    "p21_gae": "compute_gae",
    "p24_nn_vectorized": "nearest_neighbour", "p25_batchnorm": "BatchNorm1dScratch",
    "p26_data_filtering": "filter_annotations",
    "p27_cauchy_simulation": "light_source_samples, cauchy_pdf",
}.items():
    ADAPTERS.setdefault(_pid, f"from reference import {_sym}  # noqa: F401\n")
ADAPTERS["p23_moe_routing"] = (
    "from reference import top1_route, load_balancing_loss  # noqa: F401\n"
)


def main():
    manifest_rc = validate_manifest()
    if manifest_rc:
        return manifest_rc
    if "--drills" in sys.argv:
        return validate_drills()
    backup = HERE / ".stubs_backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(STUBS, backup)
    try:
        for name, src in ADAPTERS.items():
            (STUBS / f"{name}.py").write_text(src.lstrip(), encoding="utf-8")
        # only the stub tests: drill tests are meant to fail until you fix the drill
        stub_tests = sorted(str(t) for t in (HERE / "tests").glob("test_p*.py"))
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *stub_tests, "-q", "--no-header"], cwd=HERE)
        rc = proc.returncode
    finally:
        shutil.rmtree(STUBS)
        shutil.move(str(backup), str(STUBS))
    print("\nstubs restored")
    if rc == 0:
        print("every test is satisfiable by the reference solution")
    else:
        print("!! some tests cannot be passed even by the reference — fix the tests")
    return rc


if __name__ == "__main__":
    sys.exit(main())

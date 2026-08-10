"""Timed practice runner.

    python run.py                  # list the problem set and your status
    python run.py p01              # start the clock, run that problem's tests
    python run.py mha              # same, by name
    python run.py --cold           # the from-an-empty-file set, in order
    python run.py --drill d03      # a debug drill
    python run.py --reset p01      # restore the stub so you can redo it

The clock is the point. Reading an implementation and being able to produce one
under a 20-minute budget are different skills, and only the second is tested.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from problems import PROBLEMS, BY_ID, BY_NAME, COLD, DRILLS, DRILLS_BY_ID  # noqa: E402

LOG = HERE / "attempts.local.json"      # gitignored: your own record


def load_log():
    return json.loads(LOG.read_text()) if LOG.exists() else {}


def record(pid, seconds, passed):
    log = load_log()
    log.setdefault(pid, []).append(
        {"date": date.today().isoformat(), "seconds": round(seconds), "passed": passed})
    LOG.write_text(json.dumps(log, indent=2))


def status_line(p, log):
    tries = log.get(p.id, [])
    if not tries:
        return "—"
    last = tries[-1]
    mark = "pass" if last["passed"] else "FAIL"
    best = min((t["seconds"] for t in tries if t["passed"]), default=None)
    best_s = f", best {best // 60}:{best % 60:02d}" if best else ""
    return f"{mark} on {last['date']}{best_s} ({len(tries)} tries)"


def listing():
    log = load_log()
    print(f"\n{'id':<5}{'problem':<22}{'budget':>7}  {'cold':<5} {'reported':<26} status")
    print("-" * 100)
    section = None
    for p in PROBLEMS:
        if p.section != section:
            section = p.section
            print(f"  [{section}]")
        print(f"{p.id:<5}{p.name:<22}{p.minutes:>5}m  {'*' if p.cold else ' ':<5} "
              f"{p.seen:<26} {status_line(p, log)}")
    print(f"\n  cold-start set ({len(COLD)}): " + " ".join(p.name for p in COLD))
    print(f"\n{'id':<5}{'drill':<22}{'budget':>7}  reported")
    print("-" * 100)
    for d in DRILLS:
        print(f"{d.id:<5}{d.name:<22}{d.minutes:>5}m  {d.seen}")
    print()


def run_one(p, timed=True):
    test = HERE / "tests" / f"test_{p.id}_{p.name}.py"
    if not test.exists():
        print(f"no test yet for {p.id} ({test.name})")
        return False
    print(f"\n=== {p.id} · {p.title} — budget {p.minutes} min ===")
    print(f"    edit  stubs/{p.id}_{p.name}.py")
    print(f"    hints hints/{p.id}_{p.name}.md   (three levels, read only when stuck)")
    if timed:
        input("\n    press Enter to start the clock, Ctrl-C to abort... ")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "pytest", str(test), "-q"], cwd=HERE)
    dt = time.time() - t0
    ok = proc.returncode == 0
    if timed:
        over = "" if dt <= p.minutes * 60 else f"  ({dt / 60 - p.minutes:.1f} min over)"
        print(f"\n    {'PASS' if ok else 'FAIL'} in {int(dt) // 60}:{int(dt) % 60:02d}{over}")
        record(p.id, dt, ok)
    return ok


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("target", nargs="?")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--drill")
    ap.add_argument("--reset")
    ap.add_argument("--no-timer", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    a = ap.parse_args()

    if a.help:
        print(__doc__)
        return 0
    if a.reset:
        p = BY_ID.get(a.reset) or BY_NAME.get(a.reset)
        src = HERE / "stubs" / ".pristine" / f"{p.id}_{p.name}.py"
        shutil.copy(src, HERE / "stubs" / f"{p.id}_{p.name}.py")
        print(f"restored stubs/{p.id}_{p.name}.py")
        return 0
    if a.drill:
        d = DRILLS_BY_ID[a.drill]
        print(f"\n=== {d.id} · {d.name} — budget {d.minutes} min ===")
        print(f"    the bug is somewhere in drills/{d.id}_{d.name}.py")
        print(f"    run:  python -m pytest tests/test_{d.id}_{d.name}.py -q")
        return 0
    if a.cold:
        failed = [p.name for p in COLD if not run_one(p, timed=not a.no_timer)]
        print(f"\ncold set: {len(COLD) - len(failed)}/{len(COLD)} passed"
              + (f", still failing: {' '.join(failed)}" if failed else ""))
        return 0
    if not a.target:
        listing()
        return 0

    p = BY_ID.get(a.target) or BY_NAME.get(a.target)
    if not p:
        print(f"unknown problem: {a.target}")
        return 1
    return 0 if run_one(p, timed=not a.no_timer) else 1


if __name__ == "__main__":
    sys.exit(main())

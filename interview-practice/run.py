"""Timed practice runner.

    python run.py                  # list the problem set and your status
    python run.py list             # explicit form of the same command
    python run.py p01              # start the clock, run that problem's tests
    python run.py mha              # same, by name
    python run.py --cold           # the from-an-empty-file set, in order
    python run.py --drill d03      # a debug drill
    python run.py --reset p01      # restore the stub so you can redo it
    python run.py --reset d03      # restore a drill's planted bug

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
DRILLS_BY_NAME = {d.name: d for d in DRILLS}
GITHUB = "https://github.com/jxzhangjhu/jxzhangjhu.github.io/blob/master/interview-practice"


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
    best_s = f", best {best // 60}:{best % 60:02d}" if best is not None else ""
    return f"{mark} on {last['date']}{best_s} ({len(tries)} tries)"


def listing():
    log = load_log()
    print(f"\n{'id':<5}{'problem':<22}{'budget':>7}  {'cold':<5} {'reported':<46} status")
    print("-" * 120)
    section = None
    for p in PROBLEMS:
        if p.section != section:
            section = p.section
            print(f"  [{section}]")
        print(f"{p.id:<5}{p.name:<22}{p.minutes:>5}m  {'*' if p.cold else ' ':<5} "
              f"{p.seen:<46} {status_line(p, log)}")
    print(f"\n  cold-start set ({len(COLD)}): " + " ".join(p.name for p in COLD))
    print(f"\n{'id':<5}{'drill':<22}{'budget':>7}  reported")
    print("-" * 120)
    for d in DRILLS:
        print(f"{d.id:<5}{d.name:<22}{d.minutes:>5}m  {d.seen:<46} {status_line(d, log)}")
    print()


def run_one(p, timed=True):
    test = HERE / "tests" / f"test_{p.id}_{p.name}.py"
    if not test.exists():
        print(f"no test yet for {p.id} ({test.name})")
        return False
    print(f"\n=== {p.id} · {p.title} — budget {p.minutes} min ===")
    print(f"    edit  stubs/{p.id}_{p.name}.py")
    print(f"    hints hints/{p.id}_{p.name}.md   (three levels, read only when stuck)")
    print(f"    test  tests/test_{p.id}_{p.name}.py")
    print(f"    web   {GITHUB}/stubs/{p.id}_{p.name}.py")
    print(f"          {GITHUB}/hints/{p.id}_{p.name}.md")
    print(f"          {GITHUB}/tests/test_{p.id}_{p.name}.py")
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


def run_drill(d, timed=True):
    test = HERE / "tests" / f"test_{d.id}_{d.name}.py"
    if not test.exists():
        print(f"no test yet for {d.id} ({test.name})")
        return False
    print(f"\n=== {d.id} · {d.title} — budget {d.minutes} min ===")
    print(f"    debug drills/{d.id}_{d.name}.py")
    print(f"    hints hints/{d.id}_{d.name}.md   (three levels, read only when stuck)")
    print(f"    test  tests/test_{d.id}_{d.name}.py")
    print(f"    web   {GITHUB}/drills/{d.id}_{d.name}.py")
    print(f"          {GITHUB}/hints/{d.id}_{d.name}.md")
    print(f"          {GITHUB}/tests/test_{d.id}_{d.name}.py")
    if timed:
        input("\n    press Enter to start the clock, Ctrl-C to abort... ")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "pytest", str(test), "-q"], cwd=HERE)
    dt = time.time() - t0
    ok = proc.returncode == 0
    if timed:
        over = "" if dt <= d.minutes * 60 else f"  ({dt / 60 - d.minutes:.1f} min over)"
        print(f"\n    {'PASS' if ok else 'FAIL'} in {int(dt) // 60}:{int(dt) % 60:02d}{over}")
        record(d.id, dt, ok)
    return ok


def reset(target):
    p = BY_ID.get(target) or BY_NAME.get(target)
    if p:
        stem = f"{p.id}_{p.name}.py"
        src = HERE / "stubs" / ".pristine" / stem
        dst = HERE / "stubs" / stem
    else:
        d = DRILLS_BY_ID.get(target) or DRILLS_BY_NAME.get(target)
        if not d:
            print(f"unknown problem or drill: {target}")
            return 1
        stem = f"{d.id}_{d.name}.py"
        src = HERE / "drills" / ".pristine" / stem
        dst = HERE / "drills" / stem
    if not src.exists():
        print(f"reset source is missing: {src.relative_to(HERE)}")
        return 1
    shutil.copy(src, dst)
    print(f"restored {dst.relative_to(HERE)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Timed ML-coding practice runner")
    ap.add_argument("target", nargs="?")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--drill")
    ap.add_argument("--reset")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-timer", action="store_true")
    a = ap.parse_args()

    if a.reset:
        return reset(a.reset)
    if a.drill:
        d = DRILLS_BY_ID.get(a.drill) or DRILLS_BY_NAME.get(a.drill)
        if not d:
            print(f"unknown drill: {a.drill}")
            return 1
        return 0 if run_drill(d, timed=not a.no_timer) else 1
    if a.cold:
        failed = [p.name for p in COLD if not run_one(p, timed=not a.no_timer)]
        print(f"\ncold set: {len(COLD) - len(failed)}/{len(COLD)} passed"
              + (f", still failing: {' '.join(failed)}" if failed else ""))
        return 1 if failed else 0
    if a.list or not a.target or a.target == "list":
        listing()
        return 0

    p = BY_ID.get(a.target) or BY_NAME.get(a.target)
    if not p:
        print(f"unknown problem: {a.target}")
        return 1
    return 0 if run_one(p, timed=not a.no_timer) else 1


if __name__ == "__main__":
    sys.exit(main())

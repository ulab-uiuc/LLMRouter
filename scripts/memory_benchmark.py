#!/usr/bin/env python3
"""Memory benchmark for LLMRouter.

Measures peak resident memory (RSS) for a set of import scenarios, each run in a
fresh subprocess so footprints don't accumulate. The headline metric is the peak
RSS of importing the full router suite (`import llmrouter.models`) — i.e. what it
costs to load the library before doing any routing.

Usage:
    python scripts/memory_benchmark.py            # human-readable table
    python scripts/memory_benchmark.py --json     # machine-readable (for the loop)
    python scripts/memory_benchmark.py --target 450   # exit 0 if headline <= 450 MB
"""
import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys

# scenario name -> import statement(s) to execute before measuring peak RSS
SCENARIOS = {
    "interpreter": "pass",
    "import llmrouter": "import llmrouter",
    "import llmrouter.models": "import llmrouter.models",
    "import cli.router_inference": "from llmrouter.cli import router_inference",
    "import openclaw_router.server": "import openclaw_router.server",
}
HEADLINE = "import llmrouter.models"
REPEATS = 3  # median of N subprocess runs to damp noise


def _peak_rss_mb(code: str) -> float:
    snippet = (
        "import resource, platform\n"
        f"{code}\n"
        "rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        # ru_maxrss is bytes on macOS, kilobytes on Linux
        "rss = rss/1048576 if platform.system() == 'Darwin' else rss/1024\n"
        "print(rss)\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", snippet], text=True, stderr=subprocess.DEVNULL
    )
    return float(out.strip())


def run() -> dict:
    results = {}
    for name, code in SCENARIOS.items():
        samples = [_peak_rss_mb(code) for _ in range(REPEATS)]
        results[name] = round(statistics.median(samples), 1)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--target", type=float, default=None,
                    help="headline RSS (MB) target; exit 0 if met")
    args = ap.parse_args()

    results = run()
    headline = results[HEADLINE]
    interp = results["interpreter"]

    if args.json:
        print(json.dumps({
            "scenarios_mb": results,
            "headline_metric": HEADLINE,
            "headline_mb": headline,
            "library_overhead_mb": round(headline - interp, 1),
            "python": platform.python_version(),
        }, indent=2))
    else:
        print(f"\n=== LLMRouter memory benchmark (peak RSS, median of {REPEATS}) ===")
        print(f"Python {platform.python_version()} on {platform.system()}\n")
        for name, mb in results.items():
            bar = "#" * int(mb / 15)
            print(f"  {name:<32} {mb:>7.1f} MB  {bar}")
        print(f"\n  HEADLINE ({HEADLINE}): {headline:.1f} MB")
        print(f"  library overhead over bare interpreter: {headline - interp:.1f} MB")

    if args.target is not None:
        ok = headline <= args.target
        print(f"\n  target {args.target:.1f} MB -> {'MET' if ok else 'NOT met'} "
              f"(headline {headline:.1f} MB)")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

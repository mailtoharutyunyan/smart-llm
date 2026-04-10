"""
Automated Evolution Orchestrator — ACS V3.1

Runs the full evolution loop as a single command:
  1. Filter raw traces
  2. Build dataset (with validation)
  3. Pre-eval baseline
  4. Train LoRA adapter
  5. Post-eval regression check
  6. Promote or reject (with human confirmation)

Usage:
  python evolve.py                    # Full interactive loop
  python evolve.py --dry-run          # Show plan without executing
  python evolve.py --auto-reject      # Skip human confirmation on reject
  python evolve.py --skip-filter      # Skip filtering (use existing accepted traces)
"""
import json
import os
import sys
import logging
import datetime

from world_model import WorldModel
from policy import PolicyLayer
from executive import ExecutiveController
from core_acs import CoreACS
from trace_collector import TraceCollector
from quality_filter import QualityFilter
from dataset_builder import DatasetBuilder
from trainer import Trainer
from model_registry import ModelRegistry
from evaluation_suite import EvaluationSuite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("acs.evolve")


def build_stack():
    """Initialize the full component stack."""
    wm = WorldModel(db_path="./acs/graph/world.db")
    policy = PolicyLayer()
    exe = ExecutiveController(policy)
    core = CoreACS(wm)
    collector = TraceCollector()
    registry = ModelRegistry()
    qf = QualityFilter(core)
    dataset_builder = DatasetBuilder()
    trainer = Trainer(registry)
    evaluator = EvaluationSuite(core)

    return {
        "wm": wm,
        "policy": policy,
        "executive": exe,
        "core": core,
        "collector": collector,
        "registry": registry,
        "quality_filter": qf,
        "dataset_builder": dataset_builder,
        "trainer": trainer,
        "evaluator": evaluator,
    }


def hr(char="═", width=60):
    """Print a horizontal rule."""
    print(f"\n{char * width}")


def confirm(prompt: str) -> bool:
    """Ask human for yes/no confirmation."""
    while True:
        try:
            answer = input(f"\n  ⚡ {prompt} [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted by user.")
            sys.exit(1)
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer y or n.")


def step_header(step_num: int, title: str):
    """Print a formatted step header."""
    hr()
    print(f"  STEP {step_num}: {title}")
    hr("─")


def run_evolution(
    dry_run: bool = False,
    auto_reject: bool = False,
    skip_filter: bool = False,
):
    """Execute the full evolution loop."""
    start_time = datetime.datetime.now()
    
    hr()
    print("  ACS V3.1 — AUTOMATED EVOLUTION ORCHESTRATOR")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if dry_run:
        print("  MODE: DRY RUN (no training will execute)")
    hr()

    stack = build_stack()

    # ── STEP 1: Filter raw traces ───────────────────────────
    step_header(1, "QUALITY FILTER")

    if skip_filter:
        print("  [SKIPPED] Using existing accepted traces.")
    else:
        raw_traces = stack["collector"].get_all_traces()
        filtered_file = "./acs/training/filtered_traces/accepted.jsonl"

        already_filtered = set()
        if os.path.exists(filtered_file):
            with open(filtered_file, "r") as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        already_filtered.add(t.get("trace_id"))
                    except json.JSONDecodeError:
                        continue

        new_traces = [
            t for t in raw_traces
            if t.get("trace_id") not in already_filtered
        ]

        if not new_traces:
            print("  No new traces to filter.")
        else:
            print(f"  Filtering {len(new_traces)} new traces...")
            accepted, rejected = 0, 0
            for trace in new_traces:
                if stack["quality_filter"].filter_trace(trace):
                    accepted += 1
                else:
                    rejected += 1

            rate = stack["quality_filter"].acceptance_rate()
            print(f"  Results: {accepted} accepted, {rejected} rejected")
            print(f"  Acceptance rate: {rate['rate']*100:.1f}% ({rate['status']})")

            if rate["status"] == "ALERT_TOO_LENIENT":
                print("\n  ⚠️  WARNING: Acceptance rate > 40%. Evaluator may be too lenient.")
                if not confirm("Continue anyway?"):
                    print("  Aborting evolution loop.")
                    return
            elif rate["status"] == "ALERT_TOO_STRICT":
                print("\n  ⚠️  WARNING: Acceptance rate < 10%. Check model quality or filter config.")

    # Count total accepted traces
    filtered_file = "./acs/training/filtered_traces/accepted.jsonl"
    total_accepted = 0
    if os.path.exists(filtered_file):
        with open(filtered_file, "r") as f:
            total_accepted = sum(1 for _ in f)

    print(f"\n  Total accepted traces available: {total_accepted}")

    if total_accepted < 100:
        print(f"  ⚠️  Only {total_accepted} traces. Minimum 500 recommended for training.")
        if not confirm("Continue with fewer traces?"):
            print("  Aborting. Collect more traces first.")
            return

    # ── STEP 2: Build dataset ───────────────────────────────
    step_header(2, "DATASET BUILD + VALIDATION")

    traces = []
    if os.path.exists(filtered_file):
        with open(filtered_file, "r") as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    train_ds, eval_ds = stack["dataset_builder"].build_dataset(traces)
    print(f"  Training set: {len(train_ds)} traces")
    print(f"  Evaluation set: {len(eval_ds)} traces")

    report = stack["dataset_builder"].validate(train_ds)
    print(f"\n  Validation: {'✅ PASSED' if report['valid'] else '❌ FAILED'}")
    print(f"  Domain balance OK: {report.get('domain_ok')}")
    print(f"  Error correction: {report.get('error_correction_pct', 0)*100:.1f}%")
    print(f"  Shuffled traces: {report.get('shuffled_pct', 0)*100:.1f}%")
    print(f"  Style collapse OK: {report.get('style_ok')}")

    if not report["valid"]:
        print("\n  ❌ Dataset validation failed. Cannot proceed to training.")
        print(f"  Details: {json.dumps(report, indent=4)}")
        return

    # ── STEP 3: Pre-evaluation baseline ─────────────────────
    step_header(3, "PRE-TRAINING EVALUATION (Baseline)")

    current = stack["registry"].get_current()
    print(f"  Current model: {current['version']}")

    if dry_run:
        print("  [DRY RUN] Skipping evaluation.")
        pre_tier1, pre_tier2 = current.get("tier1", 0.0), current.get("tier2", 0.0)
    else:
        print("  Running Tier 1 + Tier 2...")
        pre_result = stack["evaluator"].run_full(current["version"])
        pre_tier1 = pre_result["tier1"]
        pre_tier2 = pre_result["tier2"]

    print(f"  Tier 1 (deterministic): {pre_tier1:.4f}")
    print(f"  Tier 2 (reasoning):     {pre_tier2:.4f}")

    # ── STEP 4: Training ────────────────────────────────────
    step_header(4, "LORA TRAINING")

    if dry_run:
        dry_info = stack["trainer"].dry_run()
        print(f"  [DRY RUN] Would train version: {dry_info.get('next_model_version')}")
        print(f"  Traces: {dry_info.get('trace_count')}")
        print(f"  Estimated time: {dry_info.get('estimated_time')}")
        print(f"  Config: rank={dry_info['config']['rank']}, epochs={dry_info['config']['epochs']}")
        print("\n  Dry run complete. No changes made.")
        return

    if not confirm("Proceed with LoRA training? (This may take hours)"):
        print("  Training cancelled by operator.")
        return

    # Find latest dataset version
    datasets_dir = "./acs/training/datasets/"
    versions = sorted(os.listdir(datasets_dir))
    if not versions:
        print("  ❌ No datasets found.")
        return
    latest_ds = versions[-1]

    print(f"  Training on dataset: {latest_ds}")
    print(f"  Start time: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("  ⏳ Training in progress...\n")

    train_result = stack["trainer"].train(latest_ds)

    if train_result["status"] != "completed":
        print(f"\n  ❌ Training failed: {train_result['status']}")
        if train_result.get("error"):
            print(f"  Error: {train_result['error']}")
        stack["registry"].reject(
            train_result["version"],
            f"training_{train_result['status']}"
        )
        return

    new_version = train_result["version"]
    print(f"\n  ✅ Training completed: {new_version}")

    # ── STEP 5: Post-evaluation ─────────────────────────────
    step_header(5, "POST-TRAINING EVALUATION + REGRESSION CHECK")

    print(f"  Evaluating new model: {new_version}")
    post_result = stack["evaluator"].run_full(new_version)
    post_tier1 = post_result["tier1"]
    post_tier2 = post_result["tier2"]

    print(f"  Tier 1 (deterministic): {post_tier1:.4f}")
    print(f"  Tier 2 (reasoning):     {post_tier2:.4f}")

    # Regression check
    regression = stack["evaluator"].run_tier3_regression(
        post_tier1, post_tier2, pre_tier1, pre_tier2
    )

    print(f"\n  Tier 1 delta: {regression['tier1_delta']:+.4f} ({'✅' if regression['tier1_ok'] else '❌'})")
    print(f"  Tier 2 delta: {regression['tier2_delta']:+.4f} ({'✅' if regression['tier2_ok'] else '❌'})")

    # ── STEP 6: Promote or Reject ───────────────────────────
    step_header(6, "DECISION")

    if regression["passed"]:
        print(f"  ✅ All regression guards passed.")
        print(f"  Tier 2 improvement: {regression['tier2_delta']*100:.2f}%")

        if confirm(f"Promote {new_version} to production?"):
            stack["registry"].promote(new_version, post_tier1, post_tier2)
            print(f"\n  🚀 {new_version} is now the active model!")
        else:
            stack["registry"].reject(new_version, "human_declined_promotion")
            print(f"\n  {new_version} rejected by operator.")
    else:
        print(f"  ❌ REGRESSION DETECTED")
        if not regression["tier1_ok"]:
            print(f"     Tier 1 dropped by {abs(regression['tier1_delta'])*100:.2f}% (max allowed: 2%)")
        if not regression["tier2_ok"]:
            print(f"     Tier 2 did not improve by >= 2%")

        stack["registry"].reject(
            new_version,
            f"regression: T1={regression['tier1_delta']:+.4f}, T2={regression['tier2_delta']:+.4f}"
        )
        print(f"\n  {new_version} automatically rejected.")

    # ── Plateau check ───────────────────────────────────────
    plateau = stack["evaluator"].check_plateau()
    if plateau.get("plateau"):
        hr()
        print("  ⚠️  PLATEAU DETECTED")
        print("  Two consecutive runs with < 2% Tier 2 improvement.")
        print("  Recommendation: Transition to Phase B (knowledge injection)")
        print("  Run: python acs.py phase --transition")
        hr()

    # ── Summary ─────────────────────────────────────────────
    elapsed = datetime.datetime.now() - start_time
    hr()
    print(f"  Evolution loop completed in {elapsed}")
    print(f"  Model registry: python acs.py model --list")
    hr()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ACS Automated Evolution Orchestrator")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Show plan without executing training")
    parser.add_argument("--auto-reject", action="store_true", dest="auto_reject",
                        help="Skip human confirmation on rejection")
    parser.add_argument("--skip-filter", action="store_true", dest="skip_filter",
                        help="Skip filtering step, use existing accepted traces")
    args = parser.parse_args()

    run_evolution(
        dry_run=args.dry_run,
        auto_reject=args.auto_reject,
        skip_filter=args.skip_filter,
    )

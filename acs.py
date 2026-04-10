#!/usr/bin/env python3
"""
ACS V3.1 — Autonomous Cognitive System CLI
Full integration of all 23 components.
"""
import argparse
import json
import sys
import os
import logging

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
from phase_b_builder import PhaseBBuilder

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("acs")


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
    phase_b = PhaseBBuilder(wm, core)

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
        "phase_b": phase_b,
    }


# ── Command handlers ────────────────────────────────────────

def cmd_think(args, stack):
    """Run structured reasoning on a query."""
    query = args.query
    result = stack["executive"].execute(query, stack["core"])

    # Auto-collect trace if DEEP or STANDARD
    mode = result.get("cognitive_mode", "")
    if mode in ("DEEP", "STANDARD"):
        stack["collector"].collect(result)

    if args.dev:
        print(json.dumps(result, indent=2, default=str))
    else:
        reasoning = result.get("reasoning_trace", {})
        if reasoning:
            print(f"\n{'═' * 60}")
            print(f"  Mode: {mode} | Domain: {result.get('domain', '?')}")
            print(f"  Confidence: {reasoning.get('confidence', '?')}")
            print(f"{'═' * 60}")
            print(f"\n{reasoning.get('answer', result.get('result', ''))}\n")
            if reasoning.get("gaps"):
                print(f"  Gaps: {', '.join(reasoning['gaps'])}")
            print(f"\n{'─' * 60}")
            print(
                f"  Trace ID: {result.get('trace_id', '?')} | "
                f"Latency: {result.get('latency_ms', '?')}ms"
            )
        else:
            print(result.get("result", "[no result]"))


def cmd_chat(args, stack):
    """Interactive chat loop."""
    print("ACS V3.1 Interactive Chat")
    print("Type 'quit' to exit. Traces auto-collected.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = stack["executive"].execute(query, stack["core"], interactive=True)
        mode = result.get("cognitive_mode", "")

        if mode in ("DEEP", "STANDARD"):
            stack["collector"].collect(result)

        reasoning = result.get("reasoning_trace", {})
        if reasoning:
            print(f"\n[{mode}] {reasoning.get('answer', '')}")
            print(
                f"  (confidence: {reasoning.get('confidence', '?')})\n"
            )
        else:
            print(f"\n[{mode}] {result.get('result', '')}\n")


def cmd_traces(args, stack):
    """Trace management commands."""
    if args.stats:
        stats = stack["collector"].stats()
        print(json.dumps(stats, indent=2))

    elif args.quality_report:
        report = stack["quality_filter"].acceptance_rate()
        print(json.dumps(report, indent=2))

    elif args.evaluator_drift:
        mode = "monthly" if getattr(args, 'monthly', False) else "weekly"
        result = stack["quality_filter"].check_evaluator_drift(mode=mode)
        print(json.dumps(result, indent=2))
        if result.get("drifted"):
            print("\n  ⚠️  EVALUATOR DRIFT DETECTED. See recommendation above.")
    else:
        stats = stack["collector"].stats()
        print(json.dumps(stats, indent=2))


def cmd_dataset(args, stack):
    """Dataset management commands."""
    if args.build:
        print("Loading filtered traces...")
        filtered_file = "./acs/training/filtered_traces/accepted.jsonl"
        traces = []
        if os.path.exists(filtered_file):
            with open(filtered_file, "r") as f:
                for line in f:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if len(traces) < 500:
            print(
                f"WARNING: Only {len(traces)} filtered traces "
                f"(need 500 minimum). Building anyway for testing."
            )

        dataset = stack["dataset_builder"].build_dataset(traces)
        print(f"Dataset built: {len(dataset)} traces")

    elif args.validate:
        # Find latest dataset
        datasets_dir = "./acs/training/datasets/"
        if not os.path.exists(datasets_dir):
            print("No datasets found.")
            return
        versions = sorted(os.listdir(datasets_dir))
        if not versions:
            print("No datasets found.")
            return
        latest = versions[-1]
        dataset_file = os.path.join(
            datasets_dir, latest, "dataset.jsonl"
        )
        if not os.path.exists(dataset_file):
            print(f"No dataset file in {latest}")
            return

        traces = []
        with open(dataset_file, "r") as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        report = stack["dataset_builder"].validate(traces)
        print(json.dumps(report, indent=2))

    elif args.status:
        datasets_dir = "./acs/training/datasets/"
        if not os.path.exists(datasets_dir):
            print("No datasets found.")
            return
        for ver in sorted(os.listdir(datasets_dir)):
            report_file = os.path.join(
                datasets_dir, ver, "diversity_report.json"
            )
            if os.path.exists(report_file):
                with open(report_file, "r") as f:
                    report = json.load(f)
                print(
                    f"  {ver}: {report.get('total_traces', '?')} traces, "
                    f"valid={report.get('valid', '?')}"
                )
            else:
                print(f"  {ver}: (no report)")
    else:
        print("Use --status, --validate, or --build")


def cmd_eval(args, stack):
    """Evaluation commands."""
    if args.run:
        current = stack["registry"].get_current()
        print(f"Running full evaluation for {current['version']}...")
        result = stack["evaluator"].run_full(current["version"])
        print(json.dumps(
            {k: v for k, v in result.items()
             if k not in ("tier1_detail", "tier2_detail")},
            indent=2,
        ))

    elif args.plateau_check:
        result = stack["evaluator"].check_plateau()
        print(json.dumps(result, indent=2))

    elif args.compare:
        v1, v2 = args.compare
        versions = {
            v["version"]: v
            for v in stack["registry"].list_versions()
        }
        if v1 in versions and v2 in versions:
            print(f"\n  {v1}: Tier1={versions[v1]['tier1']:.4f}, "
                  f"Tier2={versions[v1]['tier2']:.4f}")
            print(f"  {v2}: Tier1={versions[v2]['tier1']:.4f}, "
                  f"Tier2={versions[v2]['tier2']:.4f}")
            t2_delta = versions[v2]["tier2"] - versions[v1]["tier2"]
            print(f"\n  Tier2 delta: {t2_delta:+.4f}")
        else:
            print(f"Version not found. Available: "
                  f"{list(versions.keys())}")
    else:
        print("Use --run, --plateau-check, or --compare V1 V2")


def cmd_train(args, stack):
    """Training commands."""
    if args.trigger:
        datasets_dir = "./acs/training/datasets/"
        if not os.path.exists(datasets_dir):
            print("No datasets available. Run 'dataset --build' first.")
            return
        versions = sorted(os.listdir(datasets_dir))
        if not versions:
            print("No datasets available.")
            return
        latest = versions[-1]
        print(f"Training with dataset {latest}...")
        result = stack["trainer"].train(latest)
        print(json.dumps(result, indent=2, default=str))

    elif args.dry_run:
        result = stack["trainer"].dry_run()
        print(json.dumps(result, indent=2))

    elif args.status:
        print("Training status: idle (no active training)")
    else:
        print("Use --trigger, --dry-run, or --status")


def cmd_model(args, stack):
    """Model management commands."""
    if args.list:
        print("\nModel Registry:")
        print(f"{'─' * 60}")
        for v in stack["registry"].list_versions():
            status = v.get("status", "?")
            marker = " ← ACTIVE" if status == "active" else ""
            print(
                f"  {v['version']:10s} | "
                f"T1={v.get('tier1', 0):.4f} | "
                f"T2={v.get('tier2', 0):.4f} | "
                f"{status}{marker}"
            )
        print(f"{'─' * 60}")

    elif args.current:
        current = stack["registry"].get_current()
        print(json.dumps(current, indent=2))

    elif args.rollback:
        success = stack["registry"].rollback(args.rollback)
        if success:
            print(f"Rolled back to {args.rollback}")
        else:
            print(f"Failed to rollback to {args.rollback}")
    else:
        print("Use --list, --current, or --rollback VERSION")


def cmd_phase(args, stack):
    """Phase management commands."""
    if args.check:
        plateau = stack["evaluator"].check_plateau()
        if plateau.get("plateau"):
            print("Current phase: A (PLATEAU DETECTED)")
            print("Recommendation: Transition to Phase B")
        else:
            print("Current phase: A (active)")
            print("No plateau detected yet.")
        print(json.dumps(plateau, indent=2))

    elif args.transition:
        limit = getattr(args, 'limit', 50)
        stack["phase_b"].build_knowledge_dataset(limit=limit)
    else:
        print("Use --check or --transition")

def cmd_ingest(args, stack):
    """Knowledge Ingestion component."""
    topic = getattr(args, 'topic', None)
    limit = getattr(args, 'limit', 1)
    stack["phase_b"].ingest_wikipedia(args.source, topic=topic, limit=limit)


def cmd_executive(args, stack):
    """Executive controller stats."""
    if args.stats:
        print(json.dumps(
            stack["executive"].get_stats(), indent=2
        ))
    elif args.log:
        print("Executive log: see ./acs/logs/executive/")
    else:
        print(json.dumps(
            stack["executive"].get_stats(), indent=2
        ))


def cmd_policy(args, stack):
    """Policy layer management."""
    if args.state:
        print(json.dumps(stack["policy"].get_state(), indent=2))
    elif args.reset:
        stack["policy"].reset()
        print("Policy reset to defaults.")
    elif args.history:
        path = stack["policy"].outcomes_path
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
            for line in lines[-20:]:
                print(line.strip())
        else:
            print("No outcome history yet.")
    else:
        print(json.dumps(stack["policy"].get_state(), indent=2))


def cmd_filter_pipeline(args, stack):
    """Run the quality filter on all raw unfiltered traces."""
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
        print("No new traces to filter.")
        return

    print(f"Filtering {len(new_traces)} new traces...")
    accepted, rejected = 0, 0
    for trace in new_traces:
        if stack["quality_filter"].filter_trace(trace):
            accepted += 1
        else:
            rejected += 1

    rate = stack["quality_filter"].acceptance_rate()
    print(f"\nResults: {accepted} accepted, {rejected} rejected")
    print(f"Acceptance rate: {rate['rate']*100:.1f}% ({rate['status']})")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ACS V3.1 — Autonomous Cognitive System"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # Core
    chat_p = subparsers.add_parser("chat", help="Interactive chat")
    think_p = subparsers.add_parser("think", help="Structured reasoning")
    think_p.add_argument("query", type=str)
    think_p.add_argument("--dev", action="store_true",
                         help="Show full trace JSON")

    # Inspection
    for cmd in ["dreams", "stats", "self", "inspect",
                "surprised", "gaps"]:
        subparsers.add_parser(cmd, help=f"Inspection: {cmd}")

    # Executive / Policy
    exec_p = subparsers.add_parser("executive")
    exec_p.add_argument("--stats", action="store_true")
    exec_p.add_argument("--log", action="store_true")

    policy_p = subparsers.add_parser("policy")
    policy_p.add_argument("--state", action="store_true")
    policy_p.add_argument("--history", action="store_true")
    policy_p.add_argument("--why", action="store_true")
    policy_p.add_argument("--reset", action="store_true")

    # Evolution Loop
    traces_p = subparsers.add_parser("traces")
    traces_p.add_argument("--stats", action="store_true")
    traces_p.add_argument("--quality-report", action="store_true",
                          dest="quality_report")
    traces_p.add_argument("--evaluator-drift", action="store_true",
                          dest="evaluator_drift")
    traces_p.add_argument("--monthly", action="store_true",
                          help="Use monthly full-audit mode for drift detection (20 traces)")
    traces_p.add_argument("--filter", action="store_true",
                          help="Run quality filter on raw traces")

    # Evolution Orchestrator
    evolve_p = subparsers.add_parser("evolve", help="Run full automated evolution loop")
    evolve_p.add_argument("--dry-run", action="store_true", dest="dry_run",
                          help="Show plan without executing")
    evolve_p.add_argument("--skip-filter", action="store_true", dest="skip_filter",
                          help="Skip filtering, use existing accepted traces")

    dataset_p = subparsers.add_parser("dataset")
    dataset_p.add_argument("--status", action="store_true")
    dataset_p.add_argument("--validate", action="store_true")
    dataset_p.add_argument("--build", action="store_true")

    eval_p = subparsers.add_parser("eval")
    eval_p.add_argument("--run", action="store_true")
    eval_p.add_argument("--compare", nargs=2, metavar=("V1", "V2"))
    eval_p.add_argument("--plateau-check", action="store_true",
                        dest="plateau_check")

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--status", action="store_true")
    train_p.add_argument("--trigger", action="store_true")
    train_p.add_argument("--dry-run", action="store_true",
                         dest="dry_run")

    model_p = subparsers.add_parser("model")
    model_p.add_argument("--list", action="store_true")
    model_p.add_argument("--current", action="store_true")
    model_p.add_argument("--rollback", type=str, metavar="VERSION")

    phase_p = subparsers.add_parser("phase")
    phase_p.add_argument("--check", action="store_true")
    phase_p.add_argument("--transition", action="store_true")
    phase_p.add_argument("--limit", type=int, default=50, help="Number of traces to generate for Phase B")

    # Data
    ingest_p = subparsers.add_parser("ingest")
    ingest_p.add_argument("source", choices=["wiki-simple", "wiki"])
    ingest_p.add_argument("--topic", type=str, default=None, help="Specific topic to ingest. Omit for random.")
    ingest_p.add_argument("--limit", type=int, default=1, help="Number of abstracts to process.")

    for op in ["contradict", "analogy", "dream", "background"]:
        subparsers.add_parser(op, help=f"Data mode: {op}")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Build component stack
    stack = build_stack()

    # Route to handler
    handlers = {
        "think": cmd_think,
        "chat": cmd_chat,
        "traces": lambda a, s: (
            cmd_filter_pipeline(a, s)
            if getattr(a, "filter", False)
            else cmd_traces(a, s)
        ),
        "dataset": cmd_dataset,
        "eval": cmd_eval,
        "train": cmd_train,
        "model": cmd_model,
        "phase": cmd_phase,
        "ingest": cmd_ingest,
        "executive": cmd_executive,
        "policy": cmd_policy,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args, stack)
    elif args.command == "evolve":
        # Import and run the evolution orchestrator directly
        from evolve import run_evolution
        run_evolution(
            dry_run=getattr(args, 'dry_run', False),
            skip_filter=getattr(args, 'skip_filter', False),
        )
    else:
        print(f"\n[STUB] Command '{args.command}' is mapped but the handler is scheduled for development in Phase B.")
        print("Currently, Phase A exclusively targets pure autonomous reasoning evolution and training integration.")


if __name__ == "__main__":
    main()

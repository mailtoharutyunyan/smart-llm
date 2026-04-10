#!/usr/bin/env python3
"""
ACS V3.1 MCP Server — Exposes the Autonomous Cognitive System
as tools for OpenCode, Claude Desktop, or any MCP client.
"""

import json
import logging
import os
import sys

# Force all logging to stderr so it doesn't corrupt MCP stdio
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("acs.mcp")

# Ensure the ACS project directory is on the path
ACS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ACS_DIR)
os.chdir(ACS_DIR)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core_acs import CoreACS  # noqa: E402
from dataset_builder import DatasetBuilder  # noqa: E402
from evaluation_suite import EvaluationSuite  # noqa: E402
from executive import ExecutiveController  # noqa: E402
from model_registry import ModelRegistry  # noqa: E402
from policy import PolicyLayer  # noqa: E402
from quality_filter import QualityFilter  # noqa: E402
from trace_collector import TraceCollector  # noqa: E402
from world_model import WorldModel  # noqa: E402

# ── Initialize ACS stack ────────────────────────────────────
logger.info("Initializing ACS component stack...")

wm = WorldModel(db_path=os.path.join(ACS_DIR, "acs/graph/world.db"))
policy = PolicyLayer(
    config_path=os.path.join(ACS_DIR, "acs/self_model/policy.json"),
    outcomes_path=os.path.join(ACS_DIR, "acs/logs/policy/outcomes.jsonl"),
)
executive = ExecutiveController(policy)
core = CoreACS(wm)
collector = TraceCollector(raw_dir=os.path.join(ACS_DIR, "acs/training/raw_traces/"))
registry = ModelRegistry(
    registry_file=os.path.join(ACS_DIR, "acs/models/registry.json"),
    adapters_dir=os.path.join(ACS_DIR, "acs/models/adapters/"),
)
quality_filter = QualityFilter(
    core,
    log_dir=os.path.join(ACS_DIR, "acs/logs/quality_filter/"),
    filtered_dir=os.path.join(ACS_DIR, "acs/training/filtered_traces/"),
)
dataset_builder = DatasetBuilder(output_dir=os.path.join(ACS_DIR, "acs/training/datasets/"))
evaluator = EvaluationSuite(
    core,
    history_dir=os.path.join(ACS_DIR, "acs/logs/training/"),
)

logger.info("ACS stack initialized successfully.")

# ── Create MCP Server ───────────────────────────────────────
mcp = FastMCP("acs")


@mcp.tool()
def acs_think(query: str, deep: bool = False) -> str:
    """Think about a question using structured reasoning.
    Returns a full reasoning trace with decomposition, steps,
    self-critique, confidence, and identified knowledge gaps.
    Set deep=True for thorough multi-step analysis.
    """
    # Rotate evaluator to prevent style lock-in
    evaluator_models = ["lmstudio/qwen3-7b", "lmstudio/deepseek-coder-v2", "lmstudio/mistral-7b", "lmstudio/phi-4"]
    iteration = len(registry.list_versions())
    core.evaluator_model = evaluator_models[iteration % len(evaluator_models)]

    result = executive.execute(query, core)

    # Override mode if deep requested
    if deep and result.get("cognitive_mode") != "DEEP":
        result = core.think(query, "DEEP")
        result["cognitive_mode"] = "DEEP"

    # Auto-collect trace
    if result.get("cognitive_mode") in ("DEEP", "STANDARD"):
        collector.collect(result)

        # Policy Feedback Loop: Learn from outcome
        reasoning = result.get("reasoning_trace", {})
        confidence = reasoning.get("confidence", 0.0)
        try:
            conf_val = float(confidence)
        except ValueError:
            conf_val = 0.0

        success = conf_val >= 0.6
        policy.record_outcome(
            query=query, mode=result.get("cognitive_mode", "STANDARD"), success=success, confidence=conf_val
        )

        # Periodically adapt thresholds based on recent outcomes
        import random

        if random.random() < 0.2:  # 20% chance
            policy.adapt_thresholds()

    reasoning = result.get("reasoning_trace", {})
    if reasoning:
        output = {
            "answer": reasoning.get("answer", ""),
            "confidence": reasoning.get("confidence"),
            "decomposition": reasoning.get("decomposition"),
            "self_critique": reasoning.get("self_critique"),
            "gaps": reasoning.get("gaps"),
            "mode": result.get("cognitive_mode"),
            "domain": result.get("domain"),
            "trace_id": result.get("trace_id"),
            "latency_ms": result.get("latency_ms"),
        }
    else:
        output = {"answer": result.get("result", ""), "mode": result.get("cognitive_mode")}

    return json.dumps(output, indent=2)


@mcp.tool()
def acs_quick(query: str) -> str:
    """Quick answer without structured reasoning trace.
    Faster but no decomposition or self-critique.
    """
    return core.quick_generate(query)


@mcp.tool()
def acs_world_model_search(concept: str) -> str:
    """Search the World Model knowledge graph for a concept.
    Returns the node data and its connections.
    """
    node = wm.get_node(concept)
    if not node:
        return json.dumps({"found": False, "concept": concept})

    # Get neighbors
    neighbors = []
    for _, target, data in wm.graph.edges(concept, data=True):
        neighbors.append(
            {
                "target": target,
                "relation": data.get("relation", ""),
            }
        )

    return json.dumps(
        {
            "found": True,
            "concept": concept,
            "domain": node.get("domain", ""),
            "properties": node.get("properties", {}),
            "connections": neighbors,
        },
        indent=2,
    )


@mcp.tool()
def acs_world_model_add(concept: str, domain: str, properties: str = "{}") -> str:
    """Add a concept to the World Model knowledge graph.
    Properties should be a JSON string.
    """
    try:
        props = json.loads(properties)
    except json.JSONDecodeError:
        props = {"raw": properties}

    wm.add_node(concept, domain=domain, properties=props)
    wm.save_to_disk()
    return json.dumps(
        {
            "added": True,
            "concept": concept,
            "domain": domain,
            "total_nodes": wm.graph.number_of_nodes(),
        }
    )


@mcp.tool()
def acs_world_model_connect(source: str, target: str, relation: str) -> str:
    """Create a relationship between two concepts in the World Model."""
    edge_id = wm.add_edge(source, target, relation)
    wm.save_to_disk()
    return json.dumps(
        {
            "connected": True,
            "source": source,
            "target": target,
            "relation": relation,
            "edge_id": edge_id,
        }
    )


@mcp.tool()
def acs_find_path(source: str, target: str) -> str:
    """Find the shortest path between two concepts in the World Model."""
    path = wm.find_path(source, target)
    return json.dumps(
        {
            "source": source,
            "target": target,
            "path": path,
            "found": path is not None,
        }
    )


@mcp.tool()
def acs_find_contradictions() -> str:
    """Scan the World Model for logical contradictions."""
    contradictions = wm.find_contradictions()
    return json.dumps(
        {
            "contradictions_found": len(contradictions),
            "details": contradictions,
        },
        indent=2,
    )


@mcp.tool()
def acs_traces_stats() -> str:
    """Get statistics on collected reasoning traces."""
    stats = collector.stats()
    acceptance = quality_filter.acceptance_rate()
    return json.dumps(
        {
            "collection": stats,
            "quality_filter": acceptance,
        },
        indent=2,
    )


@mcp.tool()
def acs_model_status() -> str:
    """Get current model version and registry status."""
    current = registry.get_current()
    versions = registry.list_versions()
    return json.dumps(
        {
            "current": current,
            "all_versions": versions,
        },
        indent=2,
    )


@mcp.tool()
def acs_eval_plateau() -> str:
    """Check if the evolution loop has reached a plateau."""
    result = evaluator.check_plateau()
    return json.dumps(result, indent=2)


@mcp.tool()
def acs_policy_state() -> str:
    """Get current policy thresholds and adaptation state."""
    return json.dumps(policy.get_state(), indent=2)


# ── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting ACS MCP server on stdio...")
    mcp.run(transport="stdio")

"""
Component 0A: Executive Controller
Routes queries to cognitive modes based on complexity analysis.
5 modes: INSTANT, STANDARD, DEEP, CLARIFY, DEFER.
"""
import logging

logger = logging.getLogger("acs.executive")


class ExecutiveController:
    """Manages cognitive modes and query routing."""

    MODES = ["INSTANT", "STANDARD", "DEEP", "CLARIFY", "DEFER"]

    COMPLEXITY_WORDS = [
        "why", "how", "compare", "evaluate", "design",
        "explain", "analyze", "what if", "trade-off",
        "implications", "relationship", "differences",
    ]
    DEEP_WORDS = [
        "deeply", "thoroughly", "comprehensive", "detail",
        "philosophical", "fundamental", "underlying",
    ]

    def __init__(self, policy_layer):
        self.policy = policy_layer
        self.stats = {
            "INSTANT": 0, "STANDARD": 0, "DEEP": 0,
            "CLARIFY": 0, "DEFER": 0,
        }

    def route_query(self, query: str) -> str:
        """Determine cognitive mode from query complexity."""
        words = len(query.split())
        q_lower = query.lower()

        has_complexity = any(
            w in q_lower for w in self.COMPLEXITY_WORDS
        )
        wants_depth = any(
            w in q_lower for w in self.DEEP_WORDS
        )

        if self.policy.should_defer(query):
            return "DEFER"

        if words < 5 and not has_complexity:
            return "INSTANT"

        if wants_depth or words > 40:
            return "DEEP"

        if has_complexity or words > 15:
            if self.policy.should_deep_think():
                return "DEEP"
            return "STANDARD"

        if words <= 15 and not has_complexity:
            return "CLARIFY"

        return "STANDARD"

    def execute(self, query: str, acs_core, interactive: bool = False) -> dict:
        """Execute a query through the appropriate cognitive mode."""
        mode = self.route_query(query)
        self.stats[mode] += 1
        logger.info("Routing query to %s mode", mode)

        trace = {
            "query": query,
            "cognitive_mode": mode,
            "result": None,
        }

        if mode == "DEFER":
            trace["result"] = (
                "Query deferred — out of domain or "
                "confidence too low."
            )
        elif mode == "CLARIFY":
            trace["result"] = (
                "Could you elaborate? I want to give "
                "you a thorough answer."
            )
        elif mode == "INSTANT":
            trace["result"] = acs_core.quick_generate(query)
        else:
            # STANDARD or DEEP — full trace
            trace_data = acs_core.think(query, mode, enable_best_of_3=(not interactive))
            trace.update(trace_data)

        return trace

    def get_stats(self) -> dict:
        """Return mode usage statistics."""
        total = sum(self.stats.values())
        return {
            "total_queries": total,
            "by_mode": dict(self.stats),
            "distribution": {
                k: round(v / total, 3) if total > 0 else 0
                for k, v in self.stats.items()
            },
        }

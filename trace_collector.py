"""
Component 14: Trace Collector
Captures DEEP and STANDARD mode traces to JSONL (append-only).
INSTANT and CLARIFY are too shallow — not captured.
"""
import json
import os
import logging
import datetime
from collections import Counter

logger = logging.getLogger("acs.trace_collector")


class TraceCollector:
    """Captures and stores reasoning traces for the evolution loop."""

    CAPTURABLE_MODES = {"DEEP", "STANDARD"}

    def __init__(
        self,
        raw_dir: str = "./acs/training/raw_traces/",
    ):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        self.log_file = os.path.join(self.raw_dir, "traces.jsonl")

    def collect(self, trace_data: dict) -> bool:
        """Store a trace if it's from a capturable mode."""
        mode = trace_data.get("cognitive_mode", "")
        if mode not in self.CAPTURABLE_MODES:
            logger.debug(
                "Skipping trace in mode %s (not capturable)", mode
            )
            return False

        with open(self.log_file, "a") as f:
            f.write(json.dumps(trace_data) + "\n")
        logger.info(
            "Collected trace %s (mode=%s)",
            trace_data.get("trace_id", "?"),
            mode,
        )
        return True

    def get_all_traces(self) -> list[dict]:
        """Load all raw traces from disk."""
        traces = []
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return traces

    def stats(self) -> dict:
        """Return collection statistics."""
        traces = self.get_all_traces()
        if not traces:
            return {
                "raw_traces_collected": 0,
                "by_mode": {},
                "by_domain": {},
            }

        mode_counts = Counter(t.get("cognitive_mode") for t in traces)
        domain_counts = Counter(t.get("domain") for t in traces)

        return {
            "raw_traces_collected": len(traces),
            "by_mode": dict(mode_counts),
            "by_domain": dict(domain_counts),
        }

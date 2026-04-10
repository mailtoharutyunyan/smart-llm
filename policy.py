"""
Component 0B: Policy Adaptation Layer
Daily threshold updates. Learns when to reason deeply.
Persists state to JSON.
"""

import datetime
import json
import logging
import os

logger = logging.getLogger("acs.policy")


class PolicyLayer:
    """Adaptive thresholds for cognitive mode selection."""

    def __init__(
        self,
        config_path: str = "./acs/self_model/policy.json",
        outcomes_path: str = "./acs/logs/policy/outcomes.jsonl",
    ):
        self.config_path = config_path
        self.outcomes_path = outcomes_path
        self.deep_think_threshold = 0.5
        self.confidence_floor = 0.4
        self.defer_domains: list[str] = [
            "medical diagnosis",
            "legal binding advice",
            "financial investment",
            "self harm",
            "illegal",
            "extremism",
        ]
        self._load_policy()

    def _load_policy(self):
        """Load persisted policy state."""
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                data = json.load(f)
                self.deep_think_threshold = data.get("deep_think_threshold", 0.5)
                self.confidence_floor = data.get("confidence_floor", 0.4)
                self.defer_domains = data.get("defer_domains", [])

    def save_policy(self):
        """Persist policy state to disk."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(
                {
                    "deep_think_threshold": self.deep_think_threshold,
                    "confidence_floor": self.confidence_floor,
                    "defer_domains": self.defer_domains,
                    "last_updated": datetime.datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def should_defer(self, query: str) -> bool:
        """Check if query should be deferred."""
        q_lower = query.lower()
        return any(domain in q_lower for domain in self.defer_domains)

    def should_deep_think(self) -> bool:
        """Policy-driven decision on deep vs standard thinking."""
        import random

        return random.random() > self.deep_think_threshold

    def record_outcome(
        self,
        query: str,
        mode: str,
        success: bool,
        confidence: float,
    ):
        """Record a reasoning outcome for threshold adaptation."""
        try:
            confidence = float(confidence) if isinstance(confidence, (int, float, str)) else 0.0
        except ValueError:
            confidence = 0.0

        os.makedirs(os.path.dirname(self.outcomes_path), exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query[:100],
            "mode": mode,
            "success": success,
            "confidence": confidence,
        }
        with open(self.outcomes_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def adapt_thresholds(self):
        """Daily adaptation based on recent outcomes."""
        if not os.path.exists(self.outcomes_path):
            logger.info("No outcomes to adapt from.")
            return

        recent = []
        with open(self.outcomes_path) as f:
            for line in f:
                try:
                    recent.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Use last 50 outcomes
        recent = recent[-50:]
        if not recent:
            return

        success_rate = sum(1 for o in recent if o.get("success")) / len(recent)

        # Adapt: if doing well, allow more STANDARD over DEEP
        if success_rate > 0.8:
            self.deep_think_threshold = min(0.9, self.deep_think_threshold + 0.05)
        elif success_rate < 0.5:
            self.deep_think_threshold = max(0.2, self.deep_think_threshold - 0.05)

        self.save_policy()
        logger.info(
            "Adapted thresholds: deep_think=%.2f (success=%.2f)",
            self.deep_think_threshold,
            success_rate,
        )

    def get_state(self) -> dict:
        """Return current policy state."""
        return {
            "deep_think_threshold": self.deep_think_threshold,
            "confidence_floor": self.confidence_floor,
            "defer_domains": self.defer_domains,
        }

    def reset(self):
        """Reset to defaults."""
        self.deep_think_threshold = 0.5
        self.confidence_floor = 0.4
        self.defer_domains = []
        self.save_policy()
        logger.info("Policy reset to defaults.")

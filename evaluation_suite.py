"""
Component 17: Evaluation Suite
Tier 1 (deterministic), Tier 2 (reasoning quality),
Tier 3 (regression guard), Plateau detection.
"""

import datetime
import json
import logging
import os

logger = logging.getLogger("acs.evaluation_suite")

# ── Tier 1 benchmark questions ──────────────────────────────
TIER1_MATH = [
    {"q": "What is 17 * 23?", "a": "391"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "Solve for x: 2x + 5 = 15", "a": "5"},
    {"q": "What is 15% of 200?", "a": "30"},
    {"q": "What is 2^10?", "a": "1024"},
    {"q": "What is the sum of the first 10 natural numbers?", "a": "55"},
    {"q": "What is 999 + 1?", "a": "1000"},
    {"q": "What is 7! (7 factorial)?", "a": "5040"},
    {"q": "If a triangle has sides 3, 4, 5, what is its area?", "a": "6"},
    {"q": "What is log base 2 of 256?", "a": "8"},
]

TIER1_LOGIC = [
    {
        "q": "All cats are animals. Some animals are pets. Can we conclude all cats are pets?",
        "a": "no",
    },
    {
        "q": "If it rains, the ground is wet. The ground is wet. Did it rain?",
        "a": "not necessarily",
    },
    {
        "q": "A is taller than B. B is taller than C. Is A taller than C?",
        "a": "yes",
    },
]

# ── Tier 2 complex reasoning questions ──────────────────────
TIER2_QUESTIONS = [
    "Why do economic recessions tend to be cyclical?",
    "Compare the philosophical implications of determinism vs free will.",
    "How does natural selection lead to speciation?",
    "What would happen if the speed of light were halved?",
    "Why do democracies sometimes elect authoritarian leaders?",
    "Explain the relationship between entropy and information theory.",
    "How does confirmation bias affect scientific research?",
    "What are the trade-offs between privacy and security in AI systems?",
    "Why is the measurement problem in quantum mechanics so significant?",
    "Compare the strengths and weaknesses of capitalism vs socialism.",
]


class EvaluationSuite:
    """Three-tier evaluation with plateau detection."""

    def __init__(
        self,
        core_acs,
        history_dir: str = "./acs/logs/training/",
    ):
        self.core = core_acs
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)
        self.history_file = os.path.join(self.history_dir, "eval_history.jsonl")

    def _normalize_answer(self, text: str) -> str:
        """Extract core answer for comparison."""
        text = text.lower().strip()
        # Strip common wrappers
        for prefix in ["the answer is ", "answer: ", "result: "]:
            if text.startswith(prefix):
                text = text[len(prefix) :]
        return text.strip().rstrip(".")

    # ── TIER 1: Deterministic ───────────────────────────────

    def run_tier1(self) -> dict:
        """Run deterministic math + logic benchmarks."""
        logger.info("Running Tier 1 evaluation...")
        results = {"math": [], "logic": []}

        for item in TIER1_MATH:
            try:
                raw_out = self.core.quick_generate(f"Answer concisely with just the number: {item['q']}")
                answer = self._normalize_answer(raw_out)
                correct = item["a"].lower() in answer
                results["math"].append(
                    {
                        "question": item["q"],
                        "expected": item["a"],
                        "got": answer,
                        "correct": correct,
                    }
                )
            except Exception as e:
                results["math"].append(
                    {
                        "question": item["q"],
                        "error": str(e),
                        "correct": False,
                    }
                )

        for item in TIER1_LOGIC:
            try:
                raw_out = self.core.quick_generate(f"Answer yes, no, or not necessarily: {item['q']}")
                answer = self._normalize_answer(raw_out)
                correct = item["a"].lower() in answer
                results["logic"].append(
                    {
                        "question": item["q"],
                        "expected": item["a"],
                        "got": answer,
                        "correct": correct,
                    }
                )
            except Exception as e:
                results["logic"].append(
                    {
                        "question": item["q"],
                        "error": str(e),
                        "correct": False,
                    }
                )

        math_correct = sum(1 for r in results["math"] if r.get("correct"))
        logic_correct = sum(1 for r in results["logic"] if r.get("correct"))
        total = len(results["math"]) + len(results["logic"])
        correct = math_correct + logic_correct
        score = correct / total if total > 0 else 0.0

        results["score"] = round(score, 4)
        results["correct"] = correct
        results["total"] = total
        logger.info("Tier 1: %d/%d correct (%.2f%%)", correct, total, score * 100)
        return results

    # ── TIER 2: Reasoning Quality ───────────────────────────

    def run_tier2(self) -> dict:
        """Score reasoning quality via independent evaluator."""
        logger.info("Running Tier 2 evaluation...")
        scores_list = []

        for question in TIER2_QUESTIONS:
            try:
                # Generate a trace
                trace = self.core.think(question, "DEEP")
                # Evaluate it independently
                eval_scores = self.core.evaluate_trace(trace)
                avg = (
                    sum(
                        eval_scores.get(k, 0)
                        for k in [
                            "decomposition_quality",
                            "plan_coherence",
                            "step_validity",
                            "self_awareness",
                            "conclusion_support",
                            "confidence_calibration",
                        ]
                    )
                    / 6
                )
                scores_list.append(
                    {
                        "question": question,
                        "scores": eval_scores,
                        "average": round(avg, 4),
                    }
                )
            except Exception as e:
                scores_list.append(
                    {
                        "question": question,
                        "error": str(e),
                        "average": 0.0,
                    }
                )

        overall = sum(s["average"] for s in scores_list) / len(scores_list) if scores_list else 0.0

        result = {
            "score": round(overall, 4),
            "per_question": scores_list,
        }
        logger.info("Tier 2: overall score = %.4f", overall)
        return result

    # ── TIER 3: Regression Guard ────────────────────────────

    def run_tier3_regression(
        self,
        new_tier1: float,
        new_tier2: float,
        last_tier1: float,
        last_tier2: float,
    ) -> dict:
        """Check for regression: Tier1 no drop >2%, Tier2 must improve."""
        tier1_delta = new_tier1 - last_tier1
        tier2_delta = new_tier2 - last_tier2

        tier1_ok = tier1_delta >= -0.02
        tier2_ok = tier2_delta >= 0.02

        return {
            "tier1_delta": round(tier1_delta, 4),
            "tier2_delta": round(tier2_delta, 4),
            "tier1_ok": tier1_ok,
            "tier2_ok": tier2_ok,
            "passed": tier1_ok and tier2_ok,
        }

    # ── FULL EVALUATION RUN ─────────────────────────────────

    def run_full(self, model_version: str = "unknown") -> dict:
        """Run Tier 1 + Tier 2 and persist results."""
        tier1 = self.run_tier1()
        tier2 = self.run_tier2()

        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_version": model_version,
            "tier1": tier1["score"],
            "tier2": tier2["score"],
            "tier1_detail": tier1,
            "tier2_detail": tier2,
        }

        # Persist to history
        with open(self.history_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        return result

    # ── PLATEAU DETECTION ───────────────────────────────────

    def check_plateau(self) -> dict:
        """Check if last two runs show < 2% Tier 2 gain."""
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file) as f:
                for line in f:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if len(history) < 3:
            return {
                "plateau": False,
                "reason": f"Not enough history ({len(history)} runs, need 3)",
                "history": [{"version": h.get("model_version"), "tier2": h.get("tier2")} for h in history],
            }

        recent = history[-3:]
        deltas = [recent[i + 1]["tier2"] - recent[i]["tier2"] for i in range(len(recent) - 1)]

        is_plateau = all(d < 0.02 for d in deltas)

        return {
            "plateau": is_plateau,
            "recent_deltas": [round(d, 4) for d in deltas],
            "recommendation": (
                "PLATEAU DETECTED — Consider transition to Phase B (knowledge injection)"
                if is_plateau
                else "Continue Phase A"
            ),
            "history": [{"version": h.get("model_version"), "tier2": h.get("tier2")} for h in history],
        }

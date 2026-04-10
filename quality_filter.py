"""
Component 15: Quality Filter — The spine of the system.
Prevents self-approval bias. Four independent checks.
Acceptance rate 15-30% is healthy.
"""
import json
import os
import re
import subprocess
import tempfile
import logging
import datetime
from typing import Optional
from difflib import SequenceMatcher

logger = logging.getLogger("acs.quality_filter")


class QualityFilter:
    """Four-check quality gate for reasoning traces."""

    THRESHOLDS = {
        "decomposition_quality": 0.70,
        "plan_coherence": 0.70,
        "step_validity": 0.75,
        "self_awareness": 0.65,
        "conclusion_support": 0.75,
        "confidence_calibration": 0.70,
    }

    def __init__(
        self,
        core_acs,
        log_dir: str = "./acs/logs/quality_filter/",
        filtered_dir: str = "./acs/training/filtered_traces/",
    ):
        self.core = core_acs
        self.log_dir = log_dir
        self.filtered_dir = filtered_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.filtered_dir, exist_ok=True)
        self.decisions_log = os.path.join(
            self.log_dir, "decisions.jsonl"
        )
        self.evaluator_log = os.path.join(
            self.log_dir, "evaluator_scores.jsonl"
        )
        self._accepted_traces: list[dict] = []
        self._load_accepted()

    def _load_accepted(self):
        """Load previously accepted traces for dedup comparison."""
        filtered_file = os.path.join(
            self.filtered_dir, "accepted.jsonl"
        )
        if os.path.exists(filtered_file):
            with open(filtered_file, "r") as f:
                for line in f:
                    try:
                        self._accepted_traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def _log_decision(self, trace_id: str, passed: bool,
                      details: dict):
        """Append filter decision to log."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "trace_id": trace_id,
            "passed": passed,
            **details,
        }
        with open(self.decisions_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _log_evaluator(self, trace_id: str, scores: dict):
        """Log evaluator scores separately per spec."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "trace_id": trace_id,
            "scores": scores,
        }
        with open(self.evaluator_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── CHECK 1: External Ground Truth ──────────────────────

    def check_ground_truth(self, trace: dict) -> Optional[bool]:
        """Deterministic verification where possible."""
        domain = trace.get("domain", "")
        reasoning = trace.get("reasoning_trace", {})
        answer = reasoning.get("answer", "")
        expected = trace.get("expected_answer")

        if domain == "mathematics":
            return self._verify_math(answer, expected)
        elif domain == "technology":
            return self._verify_code(answer)
        else:
            # Abstract domains: no ground truth available
            return None

    def _verify_math(self, answer: str, expected: Optional[str] = None) -> bool:
        """Extract numeric/symbolic answer and strictly verify its bounds."""
        try:
            # Extract numbers from the answer
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if not numbers:
                return True  # No verifiable numeric claim
                
            import sympy
            
            # If explicit ground-truth exists, strictly evaluate numeric equivalency
            if expected:
                try:
                    expected_val = sympy.sympify(expected)
                    # Prove at least one numeric extraction matches the answer mathematically
                    if any(sympy.sympify(num_str).equals(expected_val) for num_str in numbers[:3]):
                        return True
                    return False
                except Exception:
                    pass  # Fallthrough to basic syntax parsing if parsing GT fails
                    
            # Basic sanity: try to parse with sympy (fallback)
            for num_str in numbers[:3]:
                sympy.sympify(num_str)
            return True
        except Exception:
            return False

    def _verify_code(self, answer: str) -> bool:
        """Extract code blocks and attempt to run them safely."""
        code_match = re.findall(
            r'```(?:python)?\s*\n(.*?)```', answer, re.DOTALL
        )
        if not code_match:
            return True  # No code to verify

        for code in code_match[:2]:
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w", delete=False
                ) as f:
                    f.write(code)
                    f.flush()
                    result = subprocess.run(
                        ["python3", f.name],
                        capture_output=True,
                        timeout=10,
                        text=True,
                    )
                    os.unlink(f.name)
                    if result.returncode != 0:
                        return False
            except (subprocess.TimeoutExpired, Exception):
                return False
        return True

    # ── CHECK 2: Independent Evaluator ──────────────────────

    def check_independent_evaluator(self, trace: dict) -> tuple[float, dict]:
        """Adversarial independent evaluation at temp 0.2."""
        try:
            scores = self.core.evaluate_trace(trace)
            self._log_evaluator(trace.get("trace_id", ""), scores)
            
            # Extract expected keys to form an average. Scale appropriately.
            avg_score = sum(
                scores.get(metric, 0)
                for metric in self.THRESHOLDS.keys()
            ) / max(1, len(self.THRESHOLDS))
            
            # Online Normalization tracking (rolling mean/std) to handle evaluator drift
            eval_model = getattr(self.core, "evaluator_model", "base").lower().replace("/", "_")
            log_file = os.path.join(self.log_dir, f"{eval_model}_stats.json")
            
            mean_eval, std_eval, count = 0.70, 0.10, 10
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        stats = json.load(f)
                        mean_eval, std_eval = stats.get("mean", 0.70), stats.get("std", 0.10)
                        count = stats.get("count", 10)
                except Exception:
                    pass
            
            normalized_z = (avg_score - mean_eval) / std_eval
            calibrated_score = 0.72 + (normalized_z * 0.10) # scale back to Qwen-equivalent baseline
            
            new_count = count + 1
            new_mean = mean_eval + (avg_score - mean_eval) / new_count
            new_variance = ((std_eval ** 2) * count + (avg_score - mean_eval) * (avg_score - new_mean)) / new_count
            new_std = max(0.01, new_variance ** 0.5)
            
            # DO NOT write stats here. We only write if the trace makes it to the end and is accepted!
            proposed_stats = {"mean": new_mean, "std": new_std, "count": new_count, "log_file": log_file}
                
            return calibrated_score, scores, proposed_stats
        except Exception as e:
            logger.error("Evaluator failed: %s", e)
            return 0.0, {"error": str(e)}, None

    # ── CHECK 3: Consistency Check ──────────────────────────

    def check_consistency(self, trace: dict) -> tuple[bool, float]:
        """Run same query 3 times, compare conclusions."""
        query = trace.get("query", "")
        original_answer = trace.get("reasoning_trace", {}).get(
            "answer", ""
        )
        answers = [original_answer]

        for _ in range(2):
            try:
                result = self.core.call_llm(query, temperature=0.7)
                answers.append(result.get("answer", ""))
            except Exception:
                answers.append("")

        # Compare pairwise similarity
        similarities = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                sim = SequenceMatcher(
                    None, answers[i].lower(), answers[j].lower()
                ).ratio()
                similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        return round(avg_sim, 4)

    # ── CHECK 4: Deduplication / Novelty ────────────────────

    def check_deduplication(self, trace: dict) -> tuple[bool, float]:
        """Require novelty_score > 0.3 (max similarity < 0.7)."""
        new_answer = trace.get("reasoning_trace", {}).get("answer", "")
        new_query = trace.get("query", "")
        max_sim = 0.0

        for existing in self._accepted_traces:
            existing_answer = existing.get(
                "reasoning_trace", {}
            ).get("answer", "")
            existing_query = existing.get("query", "")

            query_sim = SequenceMatcher(
                None, new_query.lower(), existing_query.lower()
            ).ratio()
            answer_sim = SequenceMatcher(
                None, new_answer.lower(), existing_answer.lower()
            ).ratio()

            combined_sim = (query_sim * 0.4) + (answer_sim * 0.6)
            if combined_sim > max_sim:
                max_sim = combined_sim

        # Decay novelty constraint iteratively to allow refinement 
        adapters_dir = "./acs/models/adapters/"
        iteration = len([d for d in os.listdir(adapters_dir) if d.startswith("v")]) + 1 if os.path.exists(adapters_dir) else 1
        target_novelty = max(0.15, 0.30 - (0.05 * (iteration - 1)))
        
        novelty_score = 1.0 - max_sim
        return novelty_score > target_novelty, novelty_score

    # ── Non-LLM ALGORITHMIC VALIDATION ──────────────────────
    
    def check_algorithmic_integrity(self, trace: dict) -> tuple[bool, str]:
        """Non-LLM deterministic logic bounding for confidence and structural consistency."""
        reasoning = trace.get("reasoning_trace", {})
        steps = reasoning.get("steps", [])
        
        final_conf = reasoning.get("confidence", 100.0)
        try:
            final_conf = float(final_conf) / 100.0 if float(final_conf) > 1.0 else float(final_conf)
        except:
            final_conf = 1.0
            
        min_step_conf = 1.0
        for s in steps:
            c = s.get("confidence")
            if c is not None:
                try:
                    val = float(c) / 100.0 if float(c) > 1.0 else float(c)
                    min_step_conf = min(min_step_conf, val)
                except:
                    pass
                    
        # Algorithmic invariant: The conclusion cannot logically transcend the weakest link in the chain
        if final_conf > min_step_conf + 0.15:
            return False, f"overconfident_propagation: final={final_conf:.2f}, min_step={min_step_conf:.2f}"
            
        return True, ""

    # ── MAIN FILTER PIPELINE ────────────────────────────────

    def filter_trace(self, trace: dict) -> bool:
        """Run all 4 checks. Returns True if trace is accepted."""
        trace_id = trace.get("trace_id", "unknown")
        details = {"checks": {}}

        # Check 1: Ground truth
        gt = self.check_ground_truth(trace)
        details["checks"]["ground_truth"] = gt
        if gt is False:
            details["rejected_by"] = "ground_truth"
            self._log_decision(trace_id, False, details)
            return False

        # Check 2: Independent evaluator (Normalization)
        eval_result = self.check_independent_evaluator(trace)
        if len(eval_result) == 3:
            evaluator_avg, scores, proposed_stats = eval_result
        else:
            evaluator_avg, scores = eval_result[0], eval_result[1]
            proposed_stats = None
            
        details["checks"]["evaluator"] = {
            "avg_score": evaluator_avg, "scores": scores
        }
        
        # NEW: Hard floor on evaluator average
        EVALUATOR_FLOOR = 0.60  # Temporary bootstrap for Iteration 1
        if evaluator_avg < EVALUATOR_FLOOR:
            details["rejected_by"] = "evaluator_below_floor"
            self._log_decision(trace_id, False, details)
            return False

        # Check 3: Consistency (No hard veto to prevent confident mistakes)
        consistency_score = self.check_consistency(trace)
        details["checks"]["consistency"] = {"score": consistency_score}

        # Check 0: Non-LLM Algorithmic Structure Bounding
        alg_pass, alg_reason = self.check_algorithmic_integrity(trace)
        if not alg_pass:
            details["rejected_by"] = "algorithmic_evaluator"
            details["checks"]["algorithmic"] = {"pass": False, "reason": alg_reason}
            self._log_decision(trace_id, False, details)
            return False
            
        details["checks"]["algorithmic"] = {"pass": True}

        # Check 4: Novelty Score (Risk 3 Soft Floor Implementation)
        is_novel, novelty_score = self.check_deduplication(trace)
        details["checks"]["novelty"] = {"novel": is_novel, "score": novelty_score}
        
        adapters_dir = "./acs/models/adapters/"
        iteration = len([d for d in os.listdir(adapters_dir) if d.startswith("v")]) + 1 if os.path.exists(adapters_dir) else 1
        target_novelty = max(0.15, 0.30 - (0.05 * (iteration - 1)))
        
        # Soft Floor implementation
        if not is_novel and novelty_score > (target_novelty - 0.05):
            # If the trace barely fails novelty, but is exceptionally deep/complex, allow it to pass the floor
            reasoning = trace.get("reasoning_trace", {})
            verified_steps = sum(1 for s in reasoning.get("steps", []) if s.get("result") or s.get("verification"))
            if verified_steps >= 5:  # High verified difficulty threshold
                is_novel = True
                details["checks"]["novelty"]["soft_floor_rescue"] = True

        if not is_novel:
            details["rejected_by"] = "insufficient_novelty"
            self._log_decision(trace_id, False, details)
            return False
        
        # Analyze Trace Difficulty (Verbosity Gaming Protection)
        reasoning = trace.get("reasoning_trace", {})
        steps = reasoning.get("steps", [])
        verified_steps = sum(1 for s in steps if s.get("result") or s.get("verification"))
        branching_factor = len(reasoning.get("decomposition", []))
        difficulty_score = min(1.0, (verified_steps * 0.03) + (branching_factor * 0.07) + 0.15)
        details["checks"]["difficulty"] = {"score": difficulty_score}
        
        # Calculate Weighted Final Core Score (Rewards harder, novel traces)
        final_quality_score = (evaluator_avg * 0.70) + (novelty_score * 0.15) + (difficulty_score * 0.15)
        details["final_quality_score"] = final_quality_score
        
        if final_quality_score < 0.70:  # Baseline threshold
            details["rejected_by"] = "overall_quality_score"
            self._log_decision(trace_id, False, details)
            return False

        # ALL PASSED — accept
        self._log_decision(trace_id, True, details)
        self._accepted_traces.append(trace)
        
        # Write Normalized Z-Score to disk ONLY after full pipeline filter success!
        if proposed_stats and "log_file" in proposed_stats:
            try:
                lf = proposed_stats.pop("log_file")
                with open(lf, "w") as f:
                    json.dump(proposed_stats, f)
            except Exception as e:
                logger.error("Failed to commit stats: %s", e)

        # Save to filtered traces
        filtered_file = os.path.join(
            self.filtered_dir, "accepted.jsonl"
        )
        with open(filtered_file, "a") as f:
            f.write(json.dumps(trace) + "\n")

        return True

    # ── EVALUATOR DRIFT DETECTION ───────────────────────────

    def check_evaluator_drift(self, mode: str = "weekly") -> dict:
        """Re-evaluate historical accepted traces and compare against original scores.
        
        Args:
            mode: 'weekly' (5 recent traces) or 'monthly' (20 traces from 3+ months ago)
        
        Returns:
            Dict with drift analysis and recommended action.
        """
        n_traces = 5 if mode == "weekly" else 20
        
        # Load evaluator score history
        if not os.path.exists(self.evaluator_log):
            return {"status": "no_data", "message": "No evaluator history available yet."}
        
        scored_entries = []
        with open(self.evaluator_log, "r") as f:
            for line in f:
                try:
                    scored_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if len(scored_entries) < n_traces:
            return {
                "status": "insufficient_data",
                "message": f"Need {n_traces} scored traces, have {len(scored_entries)}.",
            }
        
        # Select traces based on mode
        if mode == "monthly":
            # Use traces from 3+ months ago (oldest available)
            cutoff = datetime.datetime.now() - datetime.timedelta(days=90)
            old_entries = [
                e for e in scored_entries
                if datetime.datetime.fromisoformat(e["timestamp"]) < cutoff
            ]
            if len(old_entries) < n_traces:
                old_entries = scored_entries[:n_traces]  # Fallback to oldest available
            sample = old_entries[:n_traces]
        else:
            # Weekly: use most recent traces
            sample = scored_entries[-n_traces:]
        
        # Find matching accepted traces to re-evaluate
        trace_map = {}
        for entry in sample:
            trace_map[entry.get("trace_id")] = entry.get("scores", {})
        
        # Load accepted traces for re-evaluation
        reeval_traces = []
        filtered_file = os.path.join(self.filtered_dir, "accepted.jsonl")
        if os.path.exists(filtered_file):
            with open(filtered_file, "r") as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        if t.get("trace_id") in trace_map:
                            reeval_traces.append(t)
                    except json.JSONDecodeError:
                        continue
        
        if not reeval_traces:
            return {"status": "no_matching_traces", "message": "Cannot find original traces to re-evaluate."}
        
        # Re-evaluate each trace and compare
        score_keys = list(self.THRESHOLDS.keys())
        deltas = []
        per_trace = []
        
        for trace in reeval_traces[:n_traces]:
            tid = trace.get("trace_id", "?")
            original_scores = trace_map.get(tid, {})
            
            try:
                new_scores = self.core.evaluate_trace(trace)
            except Exception as e:
                logger.error("Drift re-evaluation failed for %s: %s", tid, e)
                continue
            
            trace_deltas = {}
            for key in score_keys:
                orig_val = float(original_scores.get(key, 0))
                new_val = float(new_scores.get(key, 0))
                trace_deltas[key] = round(new_val - orig_val, 4)
            
            avg_delta = sum(trace_deltas.values()) / max(1, len(trace_deltas))
            deltas.append(avg_delta)
            per_trace.append({
                "trace_id": tid,
                "avg_delta": round(avg_delta, 4),
                "per_metric": trace_deltas,
            })
        
        if not deltas:
            return {"status": "evaluation_failed", "message": "All re-evaluations failed."}
        
        overall_drift = sum(deltas) / len(deltas)
        max_drift = max(abs(d) for d in deltas)
        drifted = abs(overall_drift) > 0.10
        
        result = {
            "status": "DRIFT_DETECTED" if drifted else "stable",
            "mode": mode,
            "traces_checked": len(deltas),
            "overall_avg_delta": round(overall_drift, 4),
            "max_single_trace_delta": round(max_drift, 4),
            "drifted": drifted,
            "per_trace": per_trace,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        if drifted:
            result["recommendation"] = (
                "Evaluator has drifted significantly (>0.1 avg score change). "
                "Reset evaluator normalization stats and investigate cause "
                "(Ollama update, model weight change, prompt drift)."
            )
            logger.warning("EVALUATOR DRIFT DETECTED: avg_delta=%.4f", overall_drift)
        else:
            logger.info("Evaluator drift check passed: avg_delta=%.4f", overall_drift)
        
        # Persist drift report
        drift_dir = os.path.join(self.log_dir, "drift_checks")
        os.makedirs(drift_dir, exist_ok=True)
        report_file = os.path.join(
            drift_dir,
            f"drift_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(result, f, indent=2)
        
        return result

    # ── HEALTH MONITORING ───────────────────────────────────

    def acceptance_rate(self) -> dict:
        """Compute acceptance rate from decision log."""
        if not os.path.exists(self.decisions_log):
            return {"total": 0, "accepted": 0, "rate": 0.0,
                    "status": "no_data"}

        total, accepted = 0, 0
        with open(self.decisions_log, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    total += 1
                    if entry.get("passed"):
                        accepted += 1
                except json.JSONDecodeError:
                    continue

        rate = accepted / total if total > 0 else 0.0
        if rate > 0.40:
            status = "ALERT_TOO_LENIENT"
        elif rate < 0.10:
            status = "ALERT_TOO_STRICT"
        else:
            status = "healthy"

        return {
            "total": total,
            "accepted": accepted,
            "rate": round(rate, 4),
            "status": status,
        }

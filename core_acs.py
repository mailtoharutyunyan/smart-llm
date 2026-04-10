"""
Component 1-13: Core ACS Integration Layer
Wraps LM Studio (OpenAI-compatible) for structured reasoning.
"""

import datetime
import json
import logging
import time
import uuid

from openai import OpenAI

logger = logging.getLogger("acs.core")

REASONING_SYSTEM_PROMPT = """You are a structured reasoning engine. When given a question, you MUST respond with valid JSON only. No markdown, no preamble, no explanation outside the JSON.

Your response format:
{
  "decomposition": ["sub-problem 1", "sub-problem 2", ...],
  "plan": "Step-by-step plan as a string",
  "steps": [
    {"step": 1, "thought": "...", "result": "..."},
    {"step": 2, "thought": "...", "result": "..."}
  ],
  "self_critique": "Honest critique of your own reasoning",
  "verification": "How you verified your answer",
  "answer": "Your final answer",
  "confidence": 0.0 to 1.0,
  "gaps": ["things you're uncertain about"]
}

Rules:
- Be thorough in decomposition. Break the problem into real sub-problems.
- Each step must show actual reasoning, not filler.
- Self-critique must be genuinely critical. Find real weaknesses.
- Confidence must honestly reflect your certainty. Don't default to 0.85.
- Gaps must list what you genuinely don't know.
- Return ONLY the JSON object. Nothing else."""

EVALUATOR_SYSTEM_PROMPT = """You are a strict critic evaluating reasoning quality.
Your job is to find problems, not to approve.
Be skeptical. Look for:
- Unjustified assumptions
- Missing steps in reasoning chain
- Overconfident conclusions
- Circular reasoning
- Ignored counterarguments
- Confidence not matching evidence quality

Score STRICTLY. Use this rubric:
- 0.9-1.0: Flawless reasoning with no assumptions or missing steps. Almost impossible to achieve.
- 0.7-0.8: Good reasoning with minor gaps. This is genuinely excellent.
- 0.5-0.6: Acceptable but has noticeable flaws.
- 0.3-0.4: Major logical gaps or overconfidence.
- 0.0-0.2: Nonsense or circular reasoning.

Most traces should score between 0.4 and 0.65. Be harsh.
A score above 0.7 should be rare and reserved for exceptional reasoning.

You MUST respond with valid JSON only:
{
  "decomposition_quality": 0.0 to 1.0,
  "plan_coherence": 0.0 to 1.0,
  "step_validity": 0.0 to 1.0,
  "self_awareness": 0.0 to 1.0,
  "conclusion_support": 0.0 to 1.0,
  "confidence_calibration": 0.0 to 1.0,
  "flaws_found": ["flaw 1", "flaw 2", ...],
  "overall_assessment": "brief summary"
}

Return ONLY the JSON object. Nothing else."""


class CoreACS:
    """Components 1-13: LLM inference via LM Studio + World Model."""

    def __init__(
        self,
        world_model,
        base_url: str = "http://localhost:1234/v1",
        model_name: str = "openai/gpt-oss-20b",
        evaluator_model: str | None = None,
    ):
        self.world_model = world_model
        self.base_url = base_url
        self.model_name = model_name
        self.evaluator_model = evaluator_model or model_name
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")

    def _call_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Raw chat completion call to LM Studio."""
        use_model = model or self.model_name
        try:
            response = self.client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    def _parse_json_response(self, raw: str) -> dict:
        """Extract JSON from LLM response reliably, bypassing strict markdown boundaries."""
        import re

        text = raw.strip()
        try:
            # Fallback 1: Direct loads
            return json.loads(text)
        except Exception:
            pass

        try:
            # Fallback 2: Regex extraction
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        # Hard Fallback
        logger.error("JSON parsing critically failed for LLM response.")
        return {
            "decomposition": ["[JSON parsing failed]"],
            "plan": "Error mapping LLM output.",
            "steps": [{"step": 1, "thought": "Extracting valid structure failed.", "result": "Failed."}],
            "self_critique": "Failed to output JSON.",
            "verification": "None.",
            "answer": "[Parse Error]",
            "confidence": 0.0,
            "gaps": ["Formatting"],
        }

    def quick_generate(self, prompt: str) -> str:
        """Fast generation for INSTANT mode — no trace structure."""
        try:
            return self._call_chat(
                "You are a helpful assistant. Be concise.",
                prompt,
                temperature=0.3,
                max_tokens=512,
            )
        except Exception:
            return "[LLM unavailable — INSTANT mode failed]"

    def call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        is_evaluator: bool = False,
    ) -> dict:
        """Structured LLM call returning parsed JSON dict."""
        if is_evaluator:
            system = EVALUATOR_SYSTEM_PROMPT
            model = self.evaluator_model
        else:
            system = REASONING_SYSTEM_PROMPT
            model = self.model_name

        raw = self._call_chat(system, prompt, temperature=temperature, model=model)
        return self._parse_json_response(raw)

    def evaluate_trace(self, trace: dict) -> dict:
        """Call independent evaluator to score a reasoning trace."""
        trace_text = json.dumps(trace.get("reasoning_trace", {}), indent=2)
        prompt = f"Evaluate this reasoning trace:\n\nQuery: {trace.get('query', '')}\n\nTrace:\n{trace_text}"
        return self.call_llm(prompt, temperature=0.1, is_evaluator=True)

    def _detect_domain(self, query: str) -> str:
        """Simple keyword-based domain detection."""
        q = query.lower()
        domain_keywords = {
            "physics": [
                "physics",
                "force",
                "energy",
                "quantum",
                "gravity",
                "particle",
                "wave",
                "relativity",
                "thermodynamic",
            ],
            "biology": ["biology", "cell", "dna", "evolution", "protein", "organism", "gene", "species", "ecosystem"],
            "history": ["history", "war", "empire", "century", "revolution", "civilization", "ancient", "medieval"],
            "technology": [
                "technology",
                "computer",
                "software",
                "algorithm",
                "ai",
                "machine learning",
                "programming",
                "code",
            ],
            "philosophy": ["philosophy", "ethics", "moral", "consciousness", "existence", "epistemology", "ontology"],
            "mathematics": ["math", "equation", "proof", "theorem", "calculus", "algebra", "geometry", "statistics"],
            "economics": ["economics", "market", "gdp", "inflation", "trade", "fiscal", "monetary", "supply", "demand"],
            "psychology": ["psychology", "behavior", "cognitive", "emotion", "mental", "perception", "motivation"],
            "linguistics": ["language", "grammar", "syntax", "semantics", "phonology", "morphology", "linguistic"],
            "ethics": ["ethical", "moral dilemma", "rights", "justice", "fairness", "responsibility"],
        }
        for domain, keywords in domain_keywords.items():
            if any(kw in q for kw in keywords):
                return domain
        return "general"

    def _detect_question_type(self, query: str) -> str:
        """Classify the question type for dataset diversity."""
        q = query.lower()
        if any(w in q for w in ["why", "cause", "reason"]):
            return "causal"
        if any(w in q for w in ["what if", "predict", "would happen", "will happen"]):
            return "predictive"
        if any(w in q for w in ["compare", "vs", "difference", "versus"]):
            return "comparative"
        if any(w in q for w in ["how does", "how do", "explain", "describe"]):
            return "explanatory"
        if any(w in q for w in ["what can't", "don't know"]):
            return "meta-cognitive"
        return "explanatory"

    def _get_context_from_world_model(self, query: str) -> str:
        """Retrieve relevant entities from World Model using FTS5-indexed search."""
        found_context = []
        try:
            # Extract key terms from query for targeted search
            q_words = [w for w in query.split() if len(w) > 3]
            seen_concepts = set()

            for term in q_words[:5]:  # Limit to 5 search terms
                matches = self.world_model.search_nodes(term, limit=3)
                for match in matches:
                    concept = match["concept"]
                    if concept in seen_concepts:
                        continue
                    seen_concepts.add(concept)

                    found_context.append(
                        f"Concept '{concept}' (Domain: {match.get('domain', '')}): "
                        f"{json.dumps(match.get('properties', {}))}"
                    )
                    # Include outgoing edges for context
                    for _, target, edata in self.world_model.graph.edges(concept, data=True):
                        found_context.append(f"  - {edata.get('relation', 'related to')} -> {target}")

                    if len(seen_concepts) >= 5:  # Hard cap on context nodes
                        break
                if len(seen_concepts) >= 5:
                    break

        except Exception as e:
            logger.error("World model context retrieval failed: %s", e)

        if found_context:
            return "\n\nBackground Knowledge from World Model:\n" + "\n".join(found_context)
        return ""

    def think(self, query: str, mode: str, enable_best_of_3: bool = True) -> dict:
        """Execute structured reasoning, returning a full trace."""
        domain = self._detect_domain(query)
        question_type = self._detect_question_type(query)
        start_time = time.time()

        depth_instruction = ""
        if mode == "DEEP":
            depth_instruction = (
                " Think very deeply. Use at least 5 decomposition steps. "
                "Consider counterarguments. Be extra critical in self-critique."
            )

        prompt = f"{query}{depth_instruction}"

        # Add RAG context
        wm_context = self._get_context_from_world_model(query)
        if wm_context:
            prompt += wm_context

        llm_calls = 0
        try:
            if mode == "DEEP" and enable_best_of_3:
                logger.info("DEEP mode activated: Executing Inference-Time Scaling (Best-of-3)")
                best_score = -1.0
                best_reasoning = None

                for i in range(3):
                    try:
                        candidate = self.call_llm(prompt, temperature=0.8)
                        llm_calls += 1

                        eval_result = self.evaluate_trace({"query": query, "reasoning_trace": candidate})
                        llm_calls += 1

                        v_validity = float(eval_result.get("step_validity", 0.0))
                        v_support = float(eval_result.get("conclusion_support", 0.0))
                        v_decomp = float(eval_result.get("decomposition_quality", 0.0))
                        score = (v_validity + v_support + v_decomp) / 3.0

                        if score > best_score:
                            best_score = score
                            best_reasoning = candidate
                    except Exception as e:
                        logger.error("DEEP path %d failed: %s", i, e)

                if best_reasoning is None:
                    raise Exception("All DEEP reasoning paths failed.")
                reasoning = best_reasoning
                logger.info("Selected best trace with score: %.2f", best_score)

            else:
                reasoning = self.call_llm(prompt, temperature=0.7)
                llm_calls += 1

                # Reflection Loop
                confidence = reasoning.get("confidence", 0.0)
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.0

                if confidence < 0.6:
                    logger.info("Confidence %.2f < 0.6. Triggering reflection pass.", confidence)
                    critique = reasoning.get("self_critique", "No critique provided")
                    reflection_prompt = (
                        f"Original Query: {query}\n"
                        f"Your previous attempt had low confidence ({confidence}) and this critique: '{critique}'.\n"
                        f"Please try again, correcting your mistakes and addressing the flaws in your reasoning. "
                        f"Provide a new, fully updated return in valid JSON format."
                    )
                    if wm_context:
                        reflection_prompt += wm_context

                    reasoning = self.call_llm(reflection_prompt, temperature=0.7)
                    llm_calls += 1

        except Exception as e:
            logger.error("think() failed: %s", e)
            reasoning = {
                "decomposition": ["[LLM call failed]"],
                "plan": "[error]",
                "steps": [{"step": 1, "thought": str(e), "result": "failed"}],
                "self_critique": "LLM was unavailable.",
                "verification": "None",
                "answer": "[error — LLM unavailable]",
                "confidence": 0.0,
                "gaps": ["LLM connection failed"],
            }

        latency_ms = (time.time() - start_time) * 1000

        trace_data = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "cognitive_mode": mode,
            "domain": domain,
            "question_type": question_type,
            "reasoning_trace": reasoning,
            "llm_calls_used": llm_calls,
            "latency_ms": round(latency_ms, 2),
            "model_version": "v1_base",
        }
        return trace_data

"""
Component 19: Phase B Builder
Handles pure Knowledge Injection from Wikipedia into the World Model,
and generates Phase B datasets (World Model Facts mapped into Reasoning Traces).
"""

import datetime
import json
import logging
import os
import random
import urllib.parse
import urllib.request

logger = logging.getLogger("acs.phase_b")


class PhaseBBuilder:
    """Orchestrates Phase B domain knowledge ingestion and dataset generation."""

    def __init__(self, world_model, core_acs, output_dir="./acs/training/datasets/phase_b/"):
        self.wm = world_model
        self.core = core_acs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _fetch_wikipedia_extract(self, source: str, topic: str = None) -> tuple[str, str]:
        """Fetch plain-text extracted intro from standard or simple Wikipedia."""
        lang = "simple" if source == "wiki-simple" else "en"
        base_url = f"https://{lang}.wikipedia.org/w/api.php"

        # If no topic, request a random article title first
        if not topic:
            rand_url = f"{base_url}?action=query&list=random&rnnamespace=0&rnlimit=1&format=json"
            try:
                req = urllib.request.Request(rand_url, headers={"User-Agent": "ACS_Bot/3.1"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    topic = data["query"]["random"][0]["title"]
            except Exception as e:
                logger.error("Failed to fetch random topic: %s", e)
                return "Unknown", ""

        # Fetch actual extract
        quoted_topic = urllib.parse.quote(topic)
        extract_url = f"{base_url}?action=query&prop=extracts&exintro=1&explaintext=1&titles={quoted_topic}&format=json"

        try:
            req = urllib.request.Request(extract_url, headers={"User-Agent": "ACS_Bot/3.1"})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if page_id == "-1":
                        return topic, ""
                    return page_data.get("title", topic), page_data.get("extract", "")
        except Exception as e:
            logger.error("Failed to fetch extract for %s: %s", topic, e)
            return topic, ""

        return topic, ""

    def ingest_wikipedia(self, source: str, topic: str = None, limit: int = 1):
        """Extract ontological triads from Wikipedia and inject into the World Model."""
        logger.info(f"Initiating Knowledge Ingestion (source={source}, limit={limit})")

        for i in range(limit):
            title, text = self._fetch_wikipedia_extract(source, topic)
            if not text or len(text) < 50:
                logger.warning(f"Skipping {title} - insufficient text.")
                continue

            logger.info(f"[{i + 1}/{limit}] Processed Abstract: {title}")

            prompt = f"""
            Extract core factual relationships from this text.
            Format as JSON with two arrays: 'entities' and 'relations'.

            Entity schema: {{"concept": "name", "domain": "topic", "properties": {{"type": "category"}}}}
            Relation schema: {{"source": "concept1", "target": "concept2", "relation": "is/has/causes/requires", "confidence": 0.9}}

            Text:
            {text[:1500]}
            """

            result = self.core.call_llm(prompt, temperature=0.2)

            entities = result.get("entities", [])
            relations = result.get("relations", [])

            # Map into World Model
            for e in entities:
                self.wm.add_node(e.get("concept", ""), e.get("domain", "general"), properties=e.get("properties", {}))

            for r in relations:
                # Ensure nodes exist
                source_node = r.get("source")
                target_node = r.get("target")
                if not self.wm.get_node(source_node):
                    self.wm.add_node(source_node, "general")
                if not self.wm.get_node(target_node):
                    self.wm.add_node(target_node, "general")

                self.wm.add_edge(
                    source_node,
                    target_node,
                    r.get("relation", "related_to"),
                    properties={"confidence": r.get("confidence", 0.9)},
                )

            self.wm.save_to_disk()
            logger.info(f"Successfully injected {len(entities)} nodes and {len(relations)} edges into World Model.")

    def build_knowledge_dataset(self, limit: int = 50):
        """Transition Phase: Exports World Model subgraphs as Q&A Reasoning Traces."""
        logger.info("Executing Phase B Transition: Converting World Model to Reasoning Traces...")

        edges = list(self.wm.graph.edges(keys=True, data=True))
        if not edges:
            logger.error("World Model is completely empty. Cannot transition to Phase B.")
            return

        # Sample random edges
        target_edges = random.sample(edges, min(limit, len(edges)))
        dataset = []

        for u, v, _k, data in target_edges:
            relation = data.get("relation", "related to")
            query = f"Explain comprehensively: why does {u} {relation} {v}?"
            logger.info(f"Phase B Generation: '{query}'")

            # Use Core ACS to think deeply utilizing the injected facts implicitly (or via RAG)
            trace = self.core.think(query, mode="DEEP", enable_best_of_3=False)
            # Explicitly mark as Phase B Knowledge Trace
            trace["phase"] = "B"
            dataset.append(trace)

        # Save output
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ds_dir = os.path.join(self.output_dir, f"v_{stamp}")
        os.makedirs(ds_dir, exist_ok=True)

        out_file = os.path.join(ds_dir, "phase_b_knowledge.jsonl")
        with open(out_file, "w") as f:
            for t in dataset:
                f.write(json.dumps(t) + "\n")

        logger.info(f"Phase B dataset generation complete: {len(dataset)} traces saved to {out_file}")

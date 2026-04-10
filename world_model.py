import sqlite3
import networkx as nx
import json
import uuid
import time

class WorldModel:
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.graph = nx.MultiDiGraph()
        if db_path != ":memory:":
            import os
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                concept TEXT PRIMARY KEY,
                domain TEXT,
                properties TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source TEXT,
                target TEXT,
                relation TEXT,
                timestamp REAL,
                properties TEXT,
                FOREIGN KEY(source) REFERENCES nodes(concept),
                FOREIGN KEY(target) REFERENCES nodes(concept)
            )
        ''')
        # FTS5 virtual table for scalable full-text search on concepts
        c.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                concept, domain, content='nodes', content_rowid='rowid'
            )
        ''')
        self.conn.commit()

    def add_node(self, concept: str, domain: str, properties: dict = None):
        props = properties or {}
        self.graph.add_node(concept, domain=domain, properties=props)

    def get_node(self, concept: str):
        if concept in self.graph:
            return self.graph.nodes[concept]
        return None

    def update_node(self, concept: str, domain: str = None, properties: dict = None):
        if concept in self.graph:
            if domain is not None:
                self.graph.nodes[concept]['domain'] = domain
            if properties is not None:
                self.graph.nodes[concept]['properties'] = properties

    def delete_node(self, concept: str):
        if concept in self.graph:
            self.graph.remove_node(concept)

    def add_edge(self, source: str, target: str, relation: str, timestamp: float = None, properties: dict = None):
        props = properties or {}
        if timestamp is None:
            timestamp = time.time()
        edge_id = str(uuid.uuid4())
        self.graph.add_edge(source, target, key=edge_id, relation=relation, timestamp=timestamp, properties=props)
        return edge_id

    def get_edge(self, source: str, target: str, edge_id: str):
        if self.graph.has_edge(source, target, key=edge_id):
            return self.graph[source][target][edge_id]
        return None

    def update_edge(self, source: str, target: str, edge_id: str, relation: str = None, timestamp: float = None, properties: dict = None):
        if self.graph.has_edge(source, target, key=edge_id):
            edge = self.graph[source][target][edge_id]
            if relation is not None:
                edge['relation'] = relation
            if timestamp is not None:
                edge['timestamp'] = timestamp
            if properties is not None:
                edge['properties'] = properties

    def delete_edge(self, source: str, target: str, edge_id: str):
        if self.graph.has_edge(source, target, key=edge_id):
            self.graph.remove_edge(source, target, key=edge_id)

    def save_to_disk(self):
        c = self.conn.cursor()
        c.execute('BEGIN TRANSACTION')
        c.execute('DELETE FROM edges')
        c.execute('DELETE FROM nodes')
        # Rebuild FTS index
        c.execute("DELETE FROM nodes_fts")
        
        for node, data in self.graph.nodes(data=True):
            c.execute('INSERT INTO nodes (concept, domain, properties) VALUES (?, ?, ?)',
                      (node, data.get('domain', ''), json.dumps(data.get('properties', {}))))
            c.execute('INSERT INTO nodes_fts (concept, domain) VALUES (?, ?)',
                      (node, data.get('domain', '')))
            
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            c.execute('INSERT INTO edges (id, source, target, relation, timestamp, properties) VALUES (?, ?, ?, ?, ?, ?)',
                      (k, u, v, data.get('relation', ''), data.get('timestamp', 0.0), json.dumps(data.get('properties', {}))))
            
        self.conn.commit()

    def load_from_disk(self):
        self.graph.clear()
        c = self.conn.cursor()
        for row in c.execute('SELECT concept, domain, properties FROM nodes'):
            self.graph.add_node(row[0], domain=row[1], properties=json.loads(row[2]))
            
        # Temporarily suppress foreign key requirements during bulk loading if needed,
        # but here we load nodes then edges so it's fine.
        for row in c.execute('SELECT id, source, target, relation, timestamp, properties FROM edges'):
            self.graph.add_edge(row[1], row[2], key=row[0], relation=row[3], timestamp=row[4], properties=json.loads(row[5]))

    def find_path(self, source: str, target: str, cutoff: int = 10):
        try:
            # Protect against Path Explosion via strict depth cutoffs
            paths = nx.single_source_shortest_path(self.graph, source, cutoff=cutoff)
            path = paths.get(target, None)
            
            if path:
                # Mathematical Confidence Propagation (A * B * C)
                confidence = 1.0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_data = self.graph.get_edge_data(u, v)
                    # Extract confidence from nested properties JSON
                    max_conf = max((float(d.get('properties', {}).get('confidence', 1.0)) for d in edge_data.values()), default=1.0)
                    confidence *= max_conf
                
                return {"path": path, "propagated_confidence": round(confidence, 4)}
            return None
        except Exception:
            return None

    def query_truth_at_time(self, source: str, target: str, time_query: float):
        """Resolve canonical edge truth through time, filtering out overridden historical facts."""
        if not self.graph.has_edge(source, target):
            return []
            
        edge_data = self.graph.get_edge_data(source, target)
        valid_edges = [data for data in edge_data.values() if data.get('timestamp') is not None and data['timestamp'] <= time_query]
        
        conflicting_pairs = [("is", "is_not"), ("has", "has_not"), ("can", "cannot"), ("causes", "prevents"), ("requires", "excludes")]
        conflict_map = {}
        for pos, neg in conflicting_pairs:
            conflict_map[pos] = (pos, neg)
            conflict_map[neg] = (pos, neg)
            
        latest_state = {}
        for data in sorted(valid_edges, key=lambda x: x['timestamp']):
            rel = data['relation']
            if rel in conflict_map:
                pair = conflict_map[rel]
                latest_state[pair] = rel  # Overwrites older truth in the same conflict family
            else:
                latest_state[rel] = rel
                
        return list(set(latest_state.values()))

    def temporal_query(self, start_time: float, end_time: float):
        result = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            ts = data.get('timestamp')
            if ts is not None and start_time <= ts <= end_time:
                # include full structure for context
                result.append({"source": u, "target": v, "edge_id": k, "data": data})
        return result

    def find_contradictions(self):
        contradictions = []
        visited_pairs = set()
        
        for u, v in self.graph.edges():
            pair = (u, v)
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)
            
            edge_data = self.graph.get_edge_data(u, v)
            if edge_data:
                relations = {}
                for k, data in edge_data.items():
                    rel = data.get('relation', '')
                    if rel not in relations:
                        relations[rel] = []
                    relations[rel].append({"edge_id": k, "data": data})
                
                # Check for semantic antonym couples denoting contradiction
                conflicting_pairs = [
                    ("is", "is_not"), 
                    ("has", "has_not"), 
                    ("can", "cannot"), 
                    ("causes", "prevents"), 
                    ("requires", "excludes")
                ]
                
                for pos, neg in conflicting_pairs:
                    if pos in relations and neg in relations:
                        contradictions.append({
                            "source": u,
                            "target": v,
                            "conflict_type": f"{pos} vs {neg}",
                            "positive_edges": relations[pos],
                            "negative_edges": relations[neg]
                        })
                        
        # Detect Transitive "Semantic" Contradictions (e.g. A is B, B is C, A is_not C)
        for a in self.graph.nodes:
            for b in self.graph.successors(a):
                if a == b: continue
                # Does A is B exist?
                if not any(d['relation'] == 'is' for d in self.graph.get_edge_data(a, b).values()):
                    continue
                
                for c in self.graph.successors(b):
                    if b == c or a == c: continue
                    # Does B is C exist?
                    if not any(d['relation'] == 'is' for d in self.graph.get_edge_data(b, c).values()):
                        continue
                    
                    # Both A is B, and B is C exist. Check if A is_not C exists!
                    if self.graph.has_edge(a, c):
                        if any(d['relation'] == 'is_not' for d in self.graph.get_edge_data(a, c).values()):
                            contradictions.append({
                                "source": a,
                                "target": c,
                                "conflict_type": "transitive_semantic_mismatch",
                                "transitive_chain": f"{a} is {b}, {b} is {c}",
                                "direct_conflict": f"{a} is_not {c}"
                            })
                            
        return contradictions

    def search_nodes(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5-powered full-text search on node concepts.
        
        Falls back to in-memory substring scan if FTS table is empty
        (e.g. before first save_to_disk call).
        """
        results = []
        
        # Try FTS5 first (indexed, O(log N))
        try:
            c = self.conn.cursor()
            # Escape FTS special characters and use prefix matching
            safe_query = query.replace('"', '""')
            rows = c.execute(
                'SELECT concept, domain FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT ?',
                (f'"{safe_query}"*', limit)
            ).fetchall()
            
            if rows:
                for concept, domain in rows:
                    node_data = self.graph.nodes.get(concept, {})
                    results.append({
                        "concept": concept,
                        "domain": domain,
                        "properties": node_data.get("properties", {}),
                    })
                return results
        except Exception:
            pass
        
        # Fallback: in-memory substring scan (for pre-save or FTS failures)
        q_lower = query.lower()
        count = 0
        for concept, data in self.graph.nodes(data=True):
            if len(concept) > 2 and q_lower in concept.lower():
                results.append({
                    "concept": concept,
                    "domain": data.get("domain", ""),
                    "properties": data.get("properties", {}),
                })
                count += 1
                if count >= limit:
                    break
        
        return results

    def close(self):
        if self.conn:
            self.conn.close()

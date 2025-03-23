import json
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from app.models.document import GraphEdge, GraphNode
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRAG:
    """
    Graph-based RAG for relationship-aware retrieval.
    """

    def __init__(self, graph_path: str = "graph_rag.pkl"):
        """
        Initialize the GraphRAG.

        Args:
            graph_path: Path to save/load the graph
        """
        self.graph_path = graph_path
        # Use a MultiDiGraph instead of DiGraph to allow multiple edges between nodes
        # This is useful for representing different types of relationships
        self.graph = nx.MultiDiGraph()

    def add_node(
        self,
        node_id: str,
        content: str,
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a node to the graph.

        Args:
            node_id: Node ID
            content: Node content
            node_type: Node type (entity, concept, etc.)
            metadata: Optional node metadata

        Returns:
            Node ID
        """
        self.graph.add_node(
            node_id, content=content, type=node_type, metadata=metadata or {}
        )
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Add an edge to the graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Relation type
            weight: Edge weight
            metadata: Optional edge metadata

        Returns:
            Tuple of source and target node IDs
        """
        # Check if nodes exist
        if source_id not in self.graph.nodes:
            logger.warning(f"Source node {source_id} does not exist")
            return None

        if target_id not in self.graph.nodes:
            logger.warning(f"Target node {target_id} does not exist")
            return None

        # Add edge
        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation_type,
            weight=weight,
            metadata=metadata or {},
        )

        return (source_id, target_id)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node data or None if not found
        """
        if node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            return {
                "id": node_id,
                "content": node_data.get("content"),
                "type": node_data.get("type"),
                "metadata": node_data.get("metadata", {}),
            }
        return None

    def get_neighbors(
        self, node_id: str, relation_type: Optional[str] = None, max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get neighbors of a node.

        Args:
            node_id: Node ID
            relation_type: Optional relation type filter
            max_depth: Maximum depth to traverse

        Returns:
            List of neighbor nodes
        """
        if node_id not in self.graph.nodes:
            return []

        # Use BFS to get neighbors up to max_depth
        visited = set([node_id])
        queue = [(node_id, 0)]  # (node_id, depth)
        neighbors = []

        while queue:
            current_id, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            for neighbor_id in self.graph.neighbors(current_id):
                edge_data = self.graph.get_edge_data(current_id, neighbor_id)

                # Filter by relation type if specified
                if relation_type and edge_data.get("relation") != relation_type:
                    continue

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

                    # Add to results
                    node_data = self.graph.nodes[neighbor_id]
                    neighbors.append(
                        {
                            "id": neighbor_id,
                            "content": node_data.get("content"),
                            "type": node_data.get("type"),
                            "relation": edge_data.get("relation"),
                            "weight": edge_data.get("weight", 1.0),
                            "depth": depth + 1,
                            "metadata": node_data.get("metadata", {}),
                        }
                    )

        return neighbors

    def search(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        max_results: int = 5,
        fast_mode: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search the graph for nodes matching the query.

        Args:
            query: Search query
            node_types: Optional list of node types to filter by
            relation_types: Optional list of relation types to filter by
            max_results: Maximum number of results to return
            fast_mode: Whether to use fast mode (limited semantic search)

        Returns:
            List of matching nodes
        """
        import gc
        import re
        import time
        from collections import Counter

        # Force garbage collection before starting search to free memory
        gc.collect()

        start_time = time.time()
        logger.info(f"Searching graph for: {query}")
        query_lower = query.lower()
        results = []

        # Step 1: Initial keyword matching (optimized)
        direct_matches = []

        # Use query terms for more efficient matching
        query_terms = query_lower.split()

        # Always use sampling for initial search to reduce memory usage
        logger.info(
            f"Graph has {len(self.graph.nodes)} nodes. Using sampling for initial search."
        )
        import random

        sample_size = min(500, len(self.graph.nodes))  # Reduced sample size
        sampled_nodes = random.sample(list(self.graph.nodes), sample_size)
        nodes_to_search = [
            (node_id, self.graph.nodes[node_id]) for node_id in sampled_nodes
        ]

        # Perform keyword matching
        for node_id, node_data in nodes_to_search:
            # Skip if we already have enough direct matches
            if len(direct_matches) >= max_results * 2:
                break

            content = node_data.get("content", "").lower()
            node_type = node_data.get("type")

            # Filter by node type if specified
            if node_types and node_type not in node_types:
                continue

            # Quick check if any query term is in content before doing more expensive operations
            if any(term in content for term in query_terms):
                # Check if full query is in content
                if query_lower in content:
                    # Calculate a simple relevance score based on position and frequency
                    position = content.find(query_lower)
                    frequency = content.count(query_lower)
                    # Better scoring formula that considers both position and frequency
                    score = (frequency * 0.5) + (1.0 / (position + 1) * 0.5)

                    direct_matches.append(
                        {
                            "id": node_id,
                            "content": node_data.get("content"),
                            "type": node_type,
                            "score": score,
                            "metadata": node_data.get("metadata", {}),
                            "match_type": "direct",
                        }
                    )

        keyword_time = time.time() - start_time
        logger.info(
            f"Keyword matching completed in {keyword_time:.2f}s, found {len(direct_matches)} matches"
        )

        # If we have enough direct matches, skip semantic search to save memory
        if len(direct_matches) >= max_results:
            logger.info(
                f"Found {len(direct_matches)} direct matches, skipping semantic search"
            )
            direct_matches.sort(key=lambda x: x["score"], reverse=True)
            return direct_matches[:max_results]

        # Step 2: Add semantic matches using TF-IDF if we have few direct matches
        semantic_matches = []
        if len(direct_matches) < max_results:
            semantic_start = time.time()
            try:
                # Extract query terms (excluding common words)
                stop_words = {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "with",
                    "by",
                }
                filtered_query_terms = [
                    term.lower()
                    for term in query.split()
                    if term.lower() not in stop_words
                ]

                # If we have meaningful query terms
                if filtered_query_terms:
                    # Use a very limited sample for semantic search to reduce memory usage
                    max_nodes_for_tfidf = 200  # Reduced from 500

                    # Get node contents for TF-IDF
                    node_contents = []
                    node_ids = []

                    # First, try to find nodes that contain at least one query term
                    count = 0
                    for node_id, node_data in self.graph.nodes(data=True):
                        # Skip nodes already in direct matches
                        if any(match["id"] == node_id for match in direct_matches):
                            continue

                        # Filter by node type if specified
                        node_type = node_data.get("type")
                        if node_types and node_type not in node_types:
                            continue

                        content = node_data.get("content", "").lower()

                        # Check if any query term is in the content
                        if any(term in content for term in filtered_query_terms):
                            node_contents.append(content)
                            node_ids.append(node_id)
                            count += 1

                            if count >= max_nodes_for_tfidf:
                                break

                    # If we don't have enough nodes, add some random ones
                    if count < min(50, max_nodes_for_tfidf):  # Reduced from 100
                        import random

                        remaining_nodes = list(
                            set(self.graph.nodes)
                            - set(node_ids)
                            - set(match["id"] for match in direct_matches)
                        )
                        if remaining_nodes:
                            sample_size = min(
                                max_nodes_for_tfidf - count, len(remaining_nodes)
                            )
                            sampled_nodes = random.sample(remaining_nodes, sample_size)

                            for node_id in sampled_nodes:
                                node_data = self.graph.nodes[node_id]
                                node_type = node_data.get("type")

                                if node_types and node_type not in node_types:
                                    continue

                                node_contents.append(node_data.get("content", ""))
                                node_ids.append(node_id)
                                count += 1

                    logger.info(
                        f"Selected {len(node_contents)} nodes for semantic search"
                    )

                    # If we have nodes to compare against
                    if node_contents:
                        # Import TF-IDF here to reduce memory usage when not needed
                        from sklearn.feature_extraction.text import \
                            TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity

                        # Add query to the corpus
                        all_texts = node_contents + [query]

                        # Create TF-IDF vectorizer with reduced features
                        vectorizer = TfidfVectorizer(
                            min_df=1,
                            stop_words="english",
                            lowercase=True,
                            max_features=500,  # Reduced from 1000
                        )

                        # Compute TF-IDF vectors
                        tfidf_matrix = vectorizer.fit_transform(all_texts)

                        # Get query vector (last one)
                        query_vector = tfidf_matrix[-1]

                        # Compute similarity between query and each node
                        similarities = cosine_similarity(
                            query_vector, tfidf_matrix[:-1]
                        )[0]

                        # Add semantic matches
                        similarity_threshold = 0.3  # Increased from 0.2
                        for i, similarity in enumerate(similarities):
                            if similarity > similarity_threshold:
                                node_id = node_ids[i]
                                node_data = self.graph.nodes[node_id]

                                semantic_matches.append(
                                    {
                                        "id": node_id,
                                        "content": node_data.get("content"),
                                        "type": node_type,
                                        "score": float(similarity)
                                        * 0.8,  # Scale semantic matches slightly lower
                                        "metadata": node_data.get("metadata", {}),
                                        "match_type": "semantic",
                                    }
                                )

                        # Clean up to free memory
                        del tfidf_matrix
                        del vectorizer
                        del similarities
                        gc.collect()
            except Exception as e:
                logger.error(f"Error in semantic search: {str(e)}")

            semantic_time = time.time() - semantic_start
            logger.info(
                f"Semantic matching completed in {semantic_time:.2f}s, found {len(semantic_matches)} matches"
            )

        # Combine direct and semantic matches
        all_matches = direct_matches + semantic_matches

        # Skip connected nodes search if we have enough matches
        if len(all_matches) >= max_results:
            all_matches.sort(key=lambda x: x["score"], reverse=True)
            return all_matches[:max_results]

        # Step 3: Add connected nodes to high-scoring matches (only if we don't have enough results)
        connected_matches = []
        if all_matches:
            connected_start = time.time()

            # Get top matches to expand
            top_matches = sorted(all_matches, key=lambda x: x["score"], reverse=True)[
                : min(2, len(all_matches))
            ]  # Reduced from 3

            # Limit the number of neighbors to check per match
            max_neighbors_to_check = 10  # Reduced from 20

            for match in top_matches:
                # Get neighbors with high edge weights
                neighbors_checked = 0

                for neighbor_id in self.graph.neighbors(match["id"]):
                    # Limit the number of neighbors we check
                    neighbors_checked += 1
                    if neighbors_checked > max_neighbors_to_check:
                        break

                    # Skip if already in results
                    if any(r["id"] == neighbor_id for r in all_matches) or any(
                        r["id"] == neighbor_id for r in connected_matches
                    ):
                        continue

                    # For MultiDiGraph, get_edge_data returns a dict of edge keys to edge attributes
                    edge_data_dict = self.graph.get_edge_data(match["id"], neighbor_id)

                    # Process each edge between these nodes
                    for edge_key, edge_data in edge_data_dict.items():
                        relation = edge_data.get("relation")
                        weight = edge_data.get("weight", 0.5)

                        # Filter by relation type if specified
                        if relation_types and relation not in relation_types:
                            continue

                        # Only include high-weight connections
                        if weight > 0.5:  # Increased from 0.4
                            node_data = self.graph.nodes[neighbor_id]
                            node_type = node_data.get("type")

                            # Filter by node type if specified
                            if node_types and node_type not in node_types:
                                continue

                            # Add connected node with score based on original match and edge weight
                            connected_matches.append(
                                {
                                    "id": neighbor_id,
                                    "content": node_data.get("content"),
                                    "type": node_type,
                                    "score": match["score"]
                                    * weight
                                    * 0.7,  # Scale connected matches lower
                                    "metadata": node_data.get("metadata", {}),
                                    "match_type": "connected",
                                    "connected_to": match["id"],
                                    "relation": relation,
                                }
                            )

                            # Break after finding one good edge for this neighbor
                            break

            connected_time = time.time() - connected_start
            logger.info(
                f"Connected node matching completed in {connected_time:.2f}s, found {len(connected_matches)} matches"
            )

        # Combine all results, sort by score, and limit
        results = all_matches + connected_matches
        results.sort(key=lambda x: x["score"], reverse=True)

        # Force garbage collection again
        gc.collect()

        total_time = time.time() - start_time
        logger.info(
            f"Total search completed in {total_time:.2f}s. Found {len(results)} matches ({len(direct_matches)} direct, "
            f"{len(semantic_matches)} semantic, {len(connected_matches)} connected)"
        )

        return results[:max_results]

    def get_subgraph(
        self,
        node_ids: List[str],
        include_neighbors: bool = False,
        max_neighbors: int = 3,
    ) -> nx.MultiDiGraph:
        """
        Get a subgraph containing the specified nodes.

        Args:
            node_ids: List of node IDs
            include_neighbors: Whether to include neighbors
            max_neighbors: Maximum number of neighbors to include per node

        Returns:
            NetworkX MultiDiGraph
        """
        # Start with the specified nodes
        nodes = set(node_id for node_id in node_ids if node_id in self.graph.nodes)

        # Add neighbors if requested
        if include_neighbors:
            for node_id in list(nodes):
                # Get top neighbors by weight
                neighbors = []
                for neighbor_id in self.graph.neighbors(node_id):
                    # For MultiDiGraph, get_edge_data returns a dict of edge keys to edge attributes
                    edge_data_dict = self.graph.get_edge_data(node_id, neighbor_id)
                    # Get the highest weight among all edges between these nodes
                    max_weight = max(
                        [data.get("weight", 0.0) for data in edge_data_dict.values()]
                    )
                    neighbors.append((neighbor_id, max_weight))

                # Sort by weight and add top neighbors
                neighbors.sort(key=lambda x: x[1], reverse=True)
                for neighbor_id, _ in neighbors[:max_neighbors]:
                    nodes.add(neighbor_id)

        # Create subgraph
        return self.graph.subgraph(nodes)

    def get_important_nodes(
        self, top_n: int = 10, method: str = "pagerank"
    ) -> List[Dict[str, Any]]:
        """
        Get the most important nodes in the graph using various centrality measures.

        Args:
            top_n: Number of top nodes to return
            method: Centrality method to use ('pagerank', 'betweenness', 'degree', 'eigenvector')

        Returns:
            List of important nodes with their scores
        """
        if len(self.graph.nodes) == 0:
            return []

        # Create a simplified graph for centrality calculations
        # (convert MultiDiGraph to DiGraph with weights)
        G = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            weight = data.get("weight", 1.0)
            # If edge already exists, use the higher weight
            if G.has_edge(u, v):
                G[u][v]["weight"] = max(G[u][v]["weight"], weight)
            else:
                G.add_edge(u, v, weight=weight)

        # Calculate centrality based on the specified method
        if method == "pagerank":
            # PageRank - identifies nodes that are linked to by many other important nodes
            centrality = nx.pagerank(G, weight="weight")
        elif method == "betweenness":
            # Betweenness - identifies nodes that act as bridges between different parts of the graph
            centrality = nx.betweenness_centrality(G, weight="weight")
        elif method == "degree":
            # Degree - simply counts the number of connections
            centrality = nx.degree_centrality(G)
        elif method == "eigenvector":
            # Eigenvector - similar to PageRank, identifies nodes connected to important nodes
            centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
        else:
            # Default to PageRank
            centrality = nx.pagerank(G, weight="weight")

        # Sort nodes by centrality score
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        # Get top N nodes
        results = []
        for node_id, score in sorted_nodes[:top_n]:
            node_data = self.graph.nodes[node_id]
            results.append(
                {
                    "id": node_id,
                    "content": node_data.get("content"),
                    "type": node_data.get("type"),
                    "score": score,
                    "metadata": node_data.get("metadata", {}),
                    "centrality_method": method,
                }
            )

        return results

    def save(self) -> bool:
        """
        Save the graph to disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(
                os.path.dirname(os.path.abspath(self.graph_path)), exist_ok=True
            )

            # Save graph
            with open(self.graph_path, "wb") as f:
                pickle.dump(self.graph, f)

            logger.info(f"Graph saved to {self.graph_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
            return False

    def load(self) -> bool:
        """
        Load the graph from disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.graph_path):
                logger.warning(f"Graph file {self.graph_path} does not exist")
                return False

            # Load graph
            with open(self.graph_path, "rb") as f:
                self.graph = pickle.load(f)

            logger.info(
                f"Graph loaded from {self.graph_path} with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            return False

    def clear(self) -> None:
        """
        Clear the graph.
        """
        self.graph = nx.MultiDiGraph()

        # Remove graph file if it exists
        if os.path.exists(self.graph_path):
            os.remove(self.graph_path)
            logger.info(f"Graph file {self.graph_path} removed")

    def analyze_graph(self) -> Dict[str, Any]:
        """
        Analyze the graph structure and return statistics.

        Returns:
            Dictionary with graph statistics
        """
        if len(self.graph.nodes) == 0:
            return {"nodes": 0, "edges": 0, "empty": True}

        # Basic statistics
        num_nodes = len(self.graph.nodes)
        num_edges = len(self.graph.edges)

        # Edge type distribution
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            relation = data.get("relation", "unknown")
            edge_types[relation] = edge_types.get(relation, 0) + 1

        # Connected components
        # Convert to undirected for connected components analysis
        undirected = self.graph.to_undirected()
        connected_components = list(nx.connected_components(undirected))

        # Strongly connected components in directed graph
        strongly_connected = list(nx.strongly_connected_components(self.graph))

        # Degree statistics
        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]

        avg_in_degree = sum(in_degrees) / num_nodes if num_nodes > 0 else 0
        avg_out_degree = sum(out_degrees) / num_nodes if num_nodes > 0 else 0
        max_in_degree = max(in_degrees) if in_degrees else 0
        max_out_degree = max(out_degrees) if out_degrees else 0

        # Density (ratio of actual edges to possible edges)
        density = nx.density(self.graph)

        # Try to compute diameter (may be slow for large graphs)
        diameter = None
        if len(connected_components) > 0 and len(connected_components[0]) > 1:
            try:
                # Use the largest connected component
                largest_cc = max(connected_components, key=len)
                subgraph = undirected.subgraph(largest_cc)
                diameter = nx.diameter(subgraph)
            except (nx.NetworkXError, Exception) as e:
                logger.warning(f"Could not compute diameter: {str(e)}")

        return {
            "nodes": num_nodes,
            "edges": num_edges,
            "density": density,
            "edge_types": edge_types,
            "connected_components": len(connected_components),
            "largest_component_size": (
                len(max(connected_components, key=len)) if connected_components else 0
            ),
            "strongly_connected_components": len(strongly_connected),
            "largest_strongly_connected_size": (
                len(max(strongly_connected, key=len)) if strongly_connected else 0
            ),
            "avg_in_degree": avg_in_degree,
            "avg_out_degree": avg_out_degree,
            "max_in_degree": max_in_degree,
            "max_out_degree": max_out_degree,
            "diameter": diameter,
        }

    def build_from_database(self, db: Session) -> Tuple[int, int]:
        """
        Build the graph from database chunks. Creates nodes from chunks
        and edges based on semantic similarity.

        Args:
            db: Database session

        Returns:
            Tuple of (number of nodes, number of edges)
        """
        import re
        from collections import Counter

        import numpy as np
        from app.models.document import DocumentChunk
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Clear existing graph
        self.clear()

        # Get all chunks
        chunks = db.query(DocumentChunk).all()
        logger.info(f"Building graph from {len(chunks)} document chunks")

        # Create nodes from chunks
        for chunk in chunks:
            node_id = chunk.id
            self.add_node(
                node_id,
                chunk.content,
                "chunk",  # All nodes are of type 'chunk' for now
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.meta_data,
                },
            )

        # Create edges between sequential chunks from same document
        sequence_edges = 0
        for chunk in chunks:
            next_chunk = (
                db.query(DocumentChunk)
                .filter(
                    DocumentChunk.document_id == chunk.document_id,
                    DocumentChunk.chunk_index == chunk.chunk_index + 1,
                )
                .first()
            )

            if next_chunk:
                self.add_edge(
                    chunk.id,
                    next_chunk.id,
                    "sequence",
                    weight=1.0,
                    metadata={"type": "sequential"},
                )
                sequence_edges += 1

        logger.info(f"Created {sequence_edges} sequential edges")

        # Create semantic edges based on content similarity
        try:
            # Extract content and IDs
            chunk_contents = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.id for chunk in chunks]

            # Skip semantic edges if there are too few chunks
            if len(chunks) < 2:
                logger.warning("Not enough chunks to create semantic edges")
                return len(self.graph.nodes), len(self.graph.edges)

            # Create TF-IDF vectorizer
            logger.info("Computing TF-IDF vectors for semantic similarity")
            vectorizer = TfidfVectorizer(
                min_df=1, stop_words="english", lowercase=True, max_features=5000
            )

            # Compute TF-IDF vectors
            try:
                tfidf_matrix = vectorizer.fit_transform(chunk_contents)
                logger.info(f"Created TF-IDF matrix with shape {tfidf_matrix.shape}")

                # Compute pairwise cosine similarity
                logger.info("Computing pairwise cosine similarity")
                cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

                # Create semantic edges for similar chunks
                semantic_edges = 0
                similarity_threshold = 0.3  # Minimum similarity to create an edge
                max_edges_per_node = 5  # Maximum number of semantic edges per node

                for i in range(len(chunks)):
                    # Get top similar chunks (excluding self)
                    similarities = list(enumerate(cosine_similarities[i]))
                    # Filter out self and sort by similarity (descending)
                    similar_chunks = [
                        (j, sim)
                        for j, sim in similarities
                        if i != j and sim >= similarity_threshold
                    ]
                    similar_chunks.sort(key=lambda x: x[1], reverse=True)

                    # Add edges to top similar chunks
                    for j, similarity in similar_chunks[:max_edges_per_node]:
                        # Skip if chunks are from the same document and adjacent (already have sequence edge)
                        if (
                            chunks[i].document_id == chunks[j].document_id
                            and abs(chunks[i].chunk_index - chunks[j].chunk_index) == 1
                        ):
                            continue

                        # Add semantic edge
                        self.add_edge(
                            chunk_ids[i],
                            chunk_ids[j],
                            "semantic",
                            weight=float(similarity),
                            metadata={
                                "type": "semantic",
                                "similarity": float(similarity),
                            },
                        )
                        semantic_edges += 1

                logger.info(f"Created {semantic_edges} semantic edges")

                # Create reference edges for chunks that reference the same entities
                reference_edges = 0

                # Extract named entities (simple approach using capitalized words)
                def extract_entities(text):
                    # Simple entity extraction - words starting with capital letters
                    # In a production system, use a proper NER model
                    words = re.findall(r"\b[A-Z][a-zA-Z]*\b", text)
                    return [w for w in words if len(w) > 1]  # Filter out single letters

                # Extract entities from each chunk
                chunk_entities = []
                for chunk in chunks:
                    entities = extract_entities(chunk.content)
                    chunk_entities.append(set(entities))

                # Create edges between chunks that share entities
                entity_threshold = 2  # Minimum number of shared entities

                for i in range(len(chunks)):
                    for j in range(i + 1, len(chunks)):
                        # Skip if already connected by sequence or semantic edge
                        if self.graph.has_edge(
                            chunk_ids[i], chunk_ids[j]
                        ) or self.graph.has_edge(chunk_ids[j], chunk_ids[i]):
                            continue

                        # Find shared entities
                        shared_entities = chunk_entities[i].intersection(
                            chunk_entities[j]
                        )

                        if len(shared_entities) >= entity_threshold:
                            # Calculate weight based on number of shared entities
                            weight = min(1.0, len(shared_entities) / 10)

                            # Add bidirectional reference edges
                            self.add_edge(
                                chunk_ids[i],
                                chunk_ids[j],
                                "reference",
                                weight=weight,
                                metadata={
                                    "type": "reference",
                                    "shared_entities": list(shared_entities),
                                },
                            )
                            reference_edges += 1

                logger.info(f"Created {reference_edges} reference edges")

            except Exception as e:
                logger.error(f"Error computing semantic similarities: {str(e)}")
                logger.exception("Detailed error:")

        except Exception as e:
            logger.error(f"Error creating semantic edges: {str(e)}")
            logger.exception("Detailed error:")

        total_nodes = len(self.graph.nodes)
        total_edges = len(self.graph.edges)
        logger.info(f"Graph built with {total_nodes} nodes and {total_edges} edges")
        return total_nodes, total_edges

    def save_to_database(self, db: Session) -> Tuple[int, int]:
        """
        Save the graph to the database.

        Args:
            db: Database session

        Returns:
            Tuple of (number of nodes saved, number of edges saved)
        """
        # Clear existing nodes and edges
        db.query(GraphEdge).delete()
        db.query(GraphNode).delete()

        # Save nodes
        node_count = 0
        for node_id, node_data in self.graph.nodes(data=True):
            node = GraphNode(
                id=node_id,
                chunk_id=node_data.get("metadata", {}).get("chunk_id"),
                node_type=node_data.get("type"),
                content=node_data.get("content"),
                meta_data=node_data.get("metadata", {}),
            )
            db.add(node)
            node_count += 1

        # Save edges
        edge_count = 0
        for source_id, target_id, edge_data in self.graph.edges(data=True):
            edge = GraphEdge(
                id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                relation_type=edge_data.get("relation"),
                weight=edge_data.get("weight", 1.0),
                meta_data=edge_data.get("metadata", {}),
            )
            db.add(edge)
            edge_count += 1

        db.commit()
        return node_count, edge_count

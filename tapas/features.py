"""Feature engineering and network analysis utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import networkx as nx
import pandas as pd
from community import community_louvain
from loguru import logger
import typer

from tapas.config import PROCESSED_DATA_DIR

app = typer.Typer()


class NetworkAnalyzer:
    """Class for analyzing network properties from graphs.
    
    This class implements the paper's methodology:
    - Calculations are performed on the FULL graph first
    - Filtering (removing isolated nodes) happens AFTER calculations
    - This ensures consistency with the paper's reported values
    """

    def __init__(self, graph: Optional[nx.Graph] = None):
        """
        Initialize NetworkAnalyzer with an optional graph.

        Args:
            graph: Optional NetworkX graph to analyze
        """
        self.graph = graph
        # Cache for computed properties to avoid recalculation
        self._betweenness_cache: Optional[Dict[int, float]] = None
        self._edge_betweenness_cache: Optional[Dict[Tuple[int, int], float]] = None
        self._filtered_graph_cache: Optional[nx.Graph] = None

    @staticmethod
    def load_adjacency_matrix(file_path: Path) -> nx.Graph:
        """
        Load a graph from an adjacency matrix CSV file.

        Args:
            file_path: Path to the adjacency matrix CSV file

        Returns:
            NetworkX graph object
        """
        try:
            # Read adjacency matrix (space-separated, no headers)
            adj_matrix = np.loadtxt(file_path, delimiter=" ")
            # Convert to networkx graph
            G = nx.from_numpy_array(adj_matrix)
            # Ensure undirected
            if G.is_directed():
                G = G.to_undirected()
            return G
        except Exception as e:
            logger.error(f"Error loading graph from {file_path}: {e}")
            raise

    @staticmethod
    def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
        """
        Get the largest connected component of a graph.

        Args:
            G: NetworkX graph

        Returns:
            Subgraph representing the largest connected component
        """
        if G.number_of_nodes() == 0:
            return nx.Graph()
        if nx.is_connected(G):
            return G
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()

    @staticmethod
    def get_connected_components(G: nx.Graph) -> List[nx.Graph]:
        """
        Get all connected components of a graph.

        Args:
            G: NetworkX graph

        Returns:
            List of subgraphs, each representing a connected component
        """
        return [G.subgraph(c).copy() for c in nx.connected_components(G)]

    @staticmethod
    def filter_connected_nodes(G: nx.Graph) -> nx.Graph:
        """
        Filter out isolated nodes (nodes with degree 0) from the graph.

        Args:
            G: NetworkX graph

        Returns:
            A new graph containing only nodes with at least one edge.
        """
        nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]
        return G.subgraph(nodes_with_edges).copy()

    def get_filtered_graph(self, G: Optional[nx.Graph] = None) -> nx.Graph:
        """
        Get filtered graph (connected nodes only), with caching.

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.

        Returns:
            Filtered graph containing only nodes with at least one edge
        """
        if self._filtered_graph_cache is not None:
            return self._filtered_graph_cache
        
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        
        self._filtered_graph_cache = self.filter_connected_nodes(graph)
        return self._filtered_graph_cache

    def get_connected_nodes_count(self, G: Optional[nx.Graph] = None) -> int:
        """
        Count the number of connected nodes (nodes with at least one neighbor).

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.

        Returns:
            Number of connected nodes
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        return sum(1 for node in graph.nodes() if graph.degree(node) > 0)

    def get_average_degree(self, G: Optional[nx.Graph] = None) -> float:
        """
        Calculate the average degree of the graph.
        Calculated on filtered graph (connected nodes only).

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.

        Returns:
            Average degree
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        
        # Use filtered graph for degree calculation
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0:
            return 0.0
        degrees = dict(filtered_graph.degree())
        return sum(degrees.values()) / len(degrees) if degrees else 0.0

    def get_average_path_length(
        self, G: Optional[nx.Graph] = None, use_largest_component: bool = True
    ) -> float:
        """
        Calculate the average shortest path length.
        Handles disconnected graphs by using the largest connected component.

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.
            use_largest_component: If True, calculate on the largest connected component.
                                If False, returns 0.0 for disconnected graphs.

        Returns:
            Average shortest path length
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        
        # Use filtered graph, then get largest component
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0 or filtered_graph.number_of_edges() == 0:
            return 0.0

        # Get largest connected component
        if not nx.is_connected(filtered_graph):
            if use_largest_component:
                filtered_graph = self.get_largest_connected_component(filtered_graph)
            else:
                logger.warning("Graph is disconnected, cannot calculate average path length")
                return 0.0

        if filtered_graph.number_of_nodes() < 2:
            return 0.0

        try:
            return nx.average_shortest_path_length(filtered_graph)
        except Exception as e:
            logger.warning(f"Error calculating average path length: {e}")
            return 0.0

    def get_node_betweenness(
        self, G: Optional[nx.Graph] = None, normalized: bool = True, use_cache: bool = True
    ) -> Dict[int, float]:
        """
        Calculate betweenness centrality for all nodes.
        Calculated on FULL graph first (paper methodology).

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.
            normalized: If True, return normalized betweenness (0-1 scale).
                        If False, return unnormalized betweenness (raw counts).
            use_cache: If True, use cached betweenness values if available.

        Returns:
            Dictionary of node betweenness centralities.
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        
        # Use cache if available and normalized matches
        if use_cache and self._betweenness_cache is not None and normalized:
            return self._betweenness_cache
        
        # Calculate on FULL graph (paper methodology)
        betweenness = nx.betweenness_centrality(graph, normalized=normalized)
        
        # Cache normalized betweenness
        if normalized and use_cache:
            self._betweenness_cache = betweenness
        
        return betweenness

    def get_edge_betweenness(
        self, G: Optional[nx.Graph] = None, normalized: bool = True, use_cache: bool = True
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate edge betweenness centrality for all edges.
        Calculated on FULL graph first (paper methodology).

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.
            normalized: If True, return normalized betweenness (0-1 scale).
                        If False, return unnormalized betweenness (raw counts).
            use_cache: If True, use cached edge betweenness values if available.

        Returns:
            Dictionary of edge betweenness centralities.
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        
        # Use cache if available and normalized matches
        if use_cache and self._edge_betweenness_cache is not None and normalized:
            return self._edge_betweenness_cache
        
        # Calculate on FULL graph (paper methodology)
        edge_betweenness = nx.edge_betweenness_centrality(graph, normalized=normalized)
        
        # Cache normalized edge betweenness
        if normalized and use_cache:
            self._edge_betweenness_cache = edge_betweenness
        
        return edge_betweenness

    def get_average_betweenness(
        self, G: Optional[nx.Graph] = None, normalized: bool = False
    ) -> float:
        """
        Calculate the average betweenness centrality.
        For Table 1: unnormalized betweenness on FULL graph, averaged over ALL nodes.

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.
            normalized: If True, return normalized betweenness (0-1 scale).
                        If False, return unnormalized betweenness (raw counts).
                        Default False for Table 1 to match paper.

        Returns:
            Average betweenness centrality
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        if graph.number_of_nodes() == 0:
            return 0.0

        # Calculate on FULL graph (paper methodology for Table 1)
        betweenness = self.get_node_betweenness(graph, normalized=normalized, use_cache=True)
        
        # Average over ALL nodes (including isolated nodes with betweenness=0)
        return sum(betweenness.values()) / len(betweenness) if betweenness else 0.0

    def get_average_closeness(self, G: Optional[nx.Graph] = None) -> float:
        """
        Calculate the average closeness centrality.
        For disconnected graphs, calculates for each component separately.
        Calculated on filtered graph (connected nodes only).

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.

        Returns:
            Average closeness centrality
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        
        # Use filtered graph for closeness calculation
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0:
            return 0.0

        # For disconnected graphs, calculate for each component separately
        if not nx.is_connected(filtered_graph):
            closeness_values = []
            for component in nx.connected_components(filtered_graph):
                subgraph = filtered_graph.subgraph(component)
                if subgraph.number_of_nodes() > 1:
                    component_closeness = nx.closeness_centrality(subgraph)
                    closeness_values.extend(component_closeness.values())
            return sum(closeness_values) / len(closeness_values) if closeness_values else 0.0
        else:
            closeness = nx.closeness_centrality(filtered_graph)
            return sum(closeness.values()) / len(closeness) if closeness else 0.0

    def get_modularity(self, G: Optional[nx.Graph] = None) -> float:
        """
        Calculate modularity using the Louvain community detection algorithm.
        Calculated on filtered graph (connected nodes only).

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.

        Returns:
            Modularity value
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        
        # Use filtered graph for modularity calculation
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0 or filtered_graph.number_of_edges() == 0:
            return 0.0

        try:
            # Detect communities using Louvain algorithm
            partition = community_louvain.best_partition(filtered_graph)
            # Calculate modularity
            return community_louvain.modularity(partition, filtered_graph)
        except Exception as e:
            logger.warning(f"Error calculating modularity: {e}")
            return 0.0

    def get_clustering_coefficient(self, G: Optional[nx.Graph] = None) -> float:
        """
        Calculate the average clustering coefficient of the graph.
        Calculated on filtered graph (connected nodes only).

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.

        Returns:
            Average clustering coefficient
        """
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        
        # Use filtered graph for clustering calculation
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0:
            return 0.0

        try:
            return nx.average_clustering(filtered_graph)
        except Exception as e:
            logger.warning(f"Error calculating clustering coefficient: {e}")
            return 0.0

    def calculate_all_properties(
        self, G: Optional[nx.Graph] = None, filter_isolated: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all network properties for a given graph.
        
        Methodology (matching paper):
        - Betweenness: Calculated on FULL graph (unnormalized), averaged over ALL nodes
        - Other properties: Calculated on filtered graph (connected nodes only)
        - Filtering happens AFTER calculations

        Args:
            G: Optional NetworkX graph. If None, uses the graph stored in the instance.
            filter_isolated: If True, filter out isolated nodes for reporting.
                           Note: Calculations still use full graph for betweenness.

        Returns:
            Dictionary with all network properties
        """
        graph_for_calculation = G if G is not None else self.graph
        if graph_for_calculation is None:
            raise ValueError("No graph provided or set in the instance.")

        # Get filtered graph for properties that need it
        filtered_graph = self.get_filtered_graph(graph_for_calculation)

        if filtered_graph.number_of_nodes() == 0:
            return {
                "connected_nodes": 0,
                "degree": 0.0,
                "average_path": 0.0,
                "betweenness": 0.0,
                "closeness": 0.0,
                "modularity": 0.0,
                "clustering": 0.0,
            }

        # Betweenness: Calculate on FULL graph (unnormalized), average over ALL nodes
        # This matches the paper's Table 1 methodology
        avg_betweenness = self.get_average_betweenness(graph_for_calculation, normalized=False)

        return {
            "connected_nodes": self.get_connected_nodes_count(graph_for_calculation),
            "degree": self.get_average_degree(graph_for_calculation),
            "average_path": self.get_average_path_length(graph_for_calculation, use_largest_component=True),
            "betweenness": avg_betweenness,  # Unnormalized, from full graph, averaged over all nodes
            "closeness": self.get_average_closeness(graph_for_calculation),
            "modularity": self.get_modularity(graph_for_calculation),
            "clustering": self.get_clustering_coefficient(graph_for_calculation),
        }


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    # Example usage of NetworkAnalyzer
    # Assuming you have an adjacency matrix file at 'data/interim/extracted/Data/3.AdjacencyMatrices/Adj_Matrix_Female_ICD_age_1.csv'
    # file_path = Path("data/interim/extracted/Data/3.AdjacencyMatrices/Adj_Matrix_Female_ICD_age_1.csv")
    # G = NetworkAnalyzer.load_adjacency_matrix(file_path)
    # analyzer = NetworkAnalyzer(G)
    # properties = analyzer.calculate_all_properties()
    # logger.info(f"Connected nodes: {properties['connected_nodes']}, Degree: {properties['degree']}")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

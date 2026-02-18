"""
Network analysis methodology and data integration.

This analysis implements the paper's network metric calculations:

Network Metrics Computed:
1. Node-level metrics: degree, betweenness centrality, closeness centrality
2. Graph-level metrics: average path length, modularity, clustering coefficient
3. Integration with clinical data: prevalence, mortality rates, disease descriptions

Data Sources:
- Adjacency matrices: Comorbidity network structure (16 networks: 2 sexes × 8 age groups)
- ICD-10 codes: Disease identifiers and descriptions
- Prevalence data: Disease frequency in population (1997-2014)
- Mortality data: In-hospital mortality rates by disease

Methodology:
- Betweenness centrality calculated on complete graph before filtering
- Isolated nodes (degree = 0) removed from most analyses
- Normalization applied for cross-network comparability
- Community detection using Louvain algorithm (maximizes modularity)

Paper References:
- Table 1: Network properties across demographic groups
- Supplementary materials: Detailed network metrics
- Methods section: Metric definitions and calculations
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from community import community_louvain
from loguru import logger
import networkx as nx
import numpy as np
import pandas as pd
import typer

from tapas.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, DATA_DIR, AGE_GROUPS

app = typer.Typer()


class NetworkAnalyzer:
    """Class for analyzing network properties from graphs.

    This class implements the paper's methodology:
    - Calculations are performed on the FULL graph first
    - Filtering (removing isolated nodes) happens AFTER calculations
    """

    def __init__(self, graph: Optional[nx.Graph] = None):
        """
        Initialize NetworkAnalyzer with an optional graph.
        """
        self.graph = graph
        self._betweenness_cache: Optional[Dict[int, float]] = None
        self._edge_betweenness_cache: Optional[Dict[Tuple[int, int], float]] = None
        self._filtered_graph_cache: Optional[nx.Graph] = None

    @staticmethod
    def _resolve_paths(sex: str, age_group_id: int) -> Dict[str, Path]:
        """
        Internal helper to resolve all file paths for a specific Sex/Age group.
        Handles the fallback logic between INTERIM and root DATA directories.
        """
        # Base Data Path Logic for adjacency matrices and prevalence
        base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
        if not base_data_path.exists():
            base_data_path = DATA_DIR

        # File definitions
        adj_filename = f"Adj_Matrix_{sex}_ICD_age_{age_group_id}.csv"
       
        mort_filename = f"mortality_diag_{sex}.csv"
        
        # ICD diagnosis files are in INTERIM_DATA_DIR, not in extracted/Data
        icd_codes_path = INTERIM_DATA_DIR / "ICD10_Diagnoses.csv"
        icd_eng_path = INTERIM_DATA_DIR / "DiagAll_Eng.csv"
        
        # Mortality files are in INTERIM_DATA_DIR/mortality
        mortality_path = INTERIM_DATA_DIR /"extracted" / "Data" / mort_filename
        print("DEBUG: mortality diag: ",mortality_path)
        paths = {
            "adjacency": base_data_path / "3.AdjacencyMatrices" / adj_filename,
            "icd_codes": icd_codes_path,
            "icd_eng": icd_eng_path,
            "prevalence": base_data_path / "1.Prevalence" / "Prevalence_Sex_Age_Year_ICD.csv",
            "mortality": mortality_path,
        }
        
        # Fallback for mortality if not in the interim/mortality folder
        if not paths["mortality"].exists():
            paths["mortality"] = DATA_DIR / mort_filename

        return paths

    @staticmethod
    def load_adjacency_matrix(file_path: Path, threshold: Optional[float] = None) -> nx.Graph:
        """
        Load a graph from an adjacency matrix CSV file.
        
        Args:
            file_path: Path to the CSV.
            threshold: If provided, values < threshold are set to 0, 
                       and values >= threshold are set to 1 (unweighted binary graph).
                       If None, loads raw weights (weighted graph).
        """
        try:
            adj_matrix = np.loadtxt(file_path, delimiter=" ")
            
            if threshold is not None:
                # Binarize the matrix based on threshold for robustness testing
                adj_matrix = (adj_matrix >= threshold).astype(float)
                
            G = nx.from_numpy_array(adj_matrix)
            if G.is_directed():
                G = G.to_undirected()
            return G
        except Exception as e:
            logger.error(f"Error loading graph from {file_path}: {e}")
            raise

    @classmethod
    def load_node_metrics(cls, sex: str, age_group_id: int, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Loads graph, calculates node metrics, and merges with metadata.
        Supports thresholding for robustness analysis.
        """
        age_label = AGE_GROUPS[age_group_id]
        paths = cls._resolve_paths(sex, age_group_id)

        if not paths["adjacency"].exists():
            # logger.warning(f"Adjacency file missing: {paths['adjacency']}")
            return pd.DataFrame()

        # 1. Load Graph & Calculate Metrics
        try:
            G = cls.load_adjacency_matrix(paths["adjacency"], threshold=threshold)
            analyzer = cls(G)
            graph_obj = analyzer.graph
            
            # Calculate metrics (Topology based on the loaded graph G)
            betweenness = analyzer.get_node_betweenness(normalized=True)
            degrees = dict(graph_obj.degree())
        except Exception as e:
            logger.error(f"Graph processing error for {sex} {age_label}: {e}")
            return pd.DataFrame()

        # 2. Load Metadata (ICD, Prev, Mort)
        try:
            icd_df = pd.read_csv(paths["icd_codes"])
            # Handle column name mapping: Id->diagnose_id, Code->icd_code, ShortDescription->descr
            if 'Id' in icd_df.columns:
                icd_df = icd_df.rename(columns={'Id': 'diagnose_id', 'Code': 'icd_code', 'ShortDescription': 'descr'})
            
            # Prevalence (2014 specific)
            prev_df = pd.read_csv(paths["prevalence"]) if paths["prevalence"].exists() else pd.DataFrame()
            if not prev_df.empty:
                prev_subset = prev_df[
                    (prev_df['Age_Group'] == age_label) & 
                    (prev_df['sex'] == sex) & 
                    (prev_df['year'] == 2014)
                ]
                prev_dict = prev_subset.set_index('icd_code')['p'].to_dict()
            else:
                prev_dict = {}

            # Mortality
            mort_df = pd.read_csv(paths["mortality"]) if paths["mortality"].exists() else pd.DataFrame()
            if not mort_df.empty and 'age_10' in mort_df.columns:
                mort_subset = mort_df[mort_df['age_10'] == age_group_id]
                mort_dict = dict(zip(mort_subset['icd_code'], mort_subset['mortality']))
            else:
                mort_dict = {}

        except Exception as e:
            logger.error(f"Metadata loading error: {e}")
            return pd.DataFrame()

        # 3. Build DataFrame
        results = []
        for node_idx in graph_obj.nodes():
            degree = degrees.get(node_idx, 0)
            
            if degree > 0:
                diagnose_id = node_idx + 1 # 0-indexed to 1-indexed
                icd_row = icd_df[icd_df['diagnose_id'] == diagnose_id]
                
                if not icd_row.empty:
                    icd_code = icd_row.iloc[0]['icd_code']
                    descr = icd_row.iloc[0]['descr']
                    
                    results.append({
                        'Sex': sex,
                        'Age_Group': age_group_id, # Int for logic
                        'Age_Range': age_label,    # Str for display
                        'ICD_Code': icd_code,
                        'Description_GER': descr,
                        'Degree': degree,
                        'Betweenness': betweenness.get(node_idx, 0),
                        'Prevalence': prev_dict.get(icd_code, 0),
                        'Mortality': mort_dict.get(icd_code, 0)
                    })
        
        return pd.DataFrame(results)

    @classmethod
    def load_edge_metrics(cls, sex: str, age_group_id: int, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Loads graph, calculates edge metrics.
        Supports thresholding for robustness analysis.
        """
        age_label = AGE_GROUPS[age_group_id]
        paths = cls._resolve_paths(sex, age_group_id)

        if not paths["adjacency"].exists():
            return pd.DataFrame()

        # 1. Load Graph & Calc Edge Betweenness
        try:
            G = cls.load_adjacency_matrix(paths["adjacency"], threshold=threshold)
            analyzer = cls(G)
            edge_betweenness = analyzer.get_edge_betweenness(normalized=True)
            graph_obj = analyzer.graph
        except Exception as e:
            logger.error(f"Graph error {sex} {age_label}: {e}")
            return pd.DataFrame()

        # 2. Metadata
        try:
            icd_df = pd.read_csv(paths["icd_codes"])
            # Handle column name mapping: Id->diagnose_id, Code->icd_code, ShortDescription->descr
            if 'Id' in icd_df.columns:
                icd_df = icd_df.rename(columns={'Id': 'diagnose_id', 'Code': 'icd_code', 'ShortDescription': 'descr'})
            icd_dict = dict(zip(icd_df['diagnose_id'] - 1, icd_df['icd_code']))
            descr_dict = dict(zip(icd_df['diagnose_id'] - 1, icd_df['descr']))
            
            mort_df = pd.read_csv(paths["mortality"]) if paths["mortality"].exists() else pd.DataFrame()
            if not mort_df.empty:
                mort_subset = mort_df[mort_df['age_10'] == age_group_id]
                mort_dict = dict(zip(mort_subset['icd_code'], mort_subset['mortality']))
            else:
                mort_dict = {}
        except Exception as e:
            logger.error(f"Metadata error: {e}")
            return pd.DataFrame()

        # 3. Build DataFrame
        results = []
        for u, v in graph_obj.edges():
            bet = edge_betweenness.get((u, v))
            if bet is None:
                bet = edge_betweenness.get((v, u), 0)
            
            icd1, icd2 = icd_dict.get(u), icd_dict.get(v)
            if not icd1 or not icd2:
                continue
                
            mort1 = mort_dict.get(icd1, 0)
            mort2 = mort_dict.get(icd2, 0)
            
            results.append({
                'Sex': sex,
                'Age_Group': age_group_id,
                'Age_Range': age_label,
                'ICD_Code_1': icd1,
                'ICD_Code_2': icd2,
                'Description_1': descr_dict.get(u, ''),
                'Description_2': descr_dict.get(v, ''),
                'Edge_Betweenness': bet,
                'Mortality_1': mort1,
                'Mortality_2': mort2,
                'Mortality_Diff': abs(mort1 - mort2)
            })
            
        return pd.DataFrame(results)

    @staticmethod
    def add_english_descriptions(df: pd.DataFrame, base_path_override: Optional[Path] = None) -> pd.DataFrame:
        """Helper to add English descriptions to a DataFrame containing 'ICD_Code'."""
        # Determine path
        if base_path_override:
             eng_path = base_path_override / 'ICD10_Diagnoses_All_ENG.csv'
        else:
             base_data_path = INTERIM_DATA_DIR / "extracted" / "Data"
             if not base_data_path.exists():
                 base_data_path = DATA_DIR
             eng_path = base_data_path / 'ICD10_Diagnoses_All_ENG.csv'
        
        if not eng_path.exists():
            if 'Description_Eng' not in df.columns and 'Description_GER' in df.columns:
                 df['Description_Eng'] = df['Description_GER']
            return df

        try:
            eng_df = pd.read_csv(eng_path)
            if 'Code' in eng_df.columns and 'ShortDescription' in eng_df.columns:
                icd_to_eng = dict(zip(eng_df['Code'], eng_df['ShortDescription']))
                
                # Check if this is an edge dataframe (two codes) or node dataframe
                if 'ICD_Code' in df.columns:
                    df['Description_Eng'] = df['ICD_Code'].map(icd_to_eng)
                    df['Description_Eng'] = df['Description_Eng'].fillna(df['Description_GER'])
                
                if 'ICD_Code_1' in df.columns:
                    df['Description_Eng_1'] = df['ICD_Code_1'].map(icd_to_eng).fillna(df['Description_1'])
                    df['Description_Eng_2'] = df['ICD_Code_2'].map(icd_to_eng).fillna(df['Description_2'])
        except Exception as e:
            logger.warning(f"Failed to add English descriptions: {e}")

        return df

    # ... [Keep existing methods: get_largest_connected_component, etc.] ...
    @staticmethod
    def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
        if G.number_of_nodes() == 0:
            return nx.Graph()
        if nx.is_connected(G):
            return G
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()

    @staticmethod
    def filter_connected_nodes(G: nx.Graph) -> nx.Graph:
        nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]
        return G.subgraph(nodes_with_edges).copy()

    def get_filtered_graph(self, G: Optional[nx.Graph] = None) -> nx.Graph:
        if self._filtered_graph_cache is not None:
            return self._filtered_graph_cache
        graph = G if G is not None else self.graph
        if graph is None:
            raise ValueError("No graph provided or set in the instance.")
        self._filtered_graph_cache = self.filter_connected_nodes(graph)
        return self._filtered_graph_cache

    def get_connected_nodes_count(self, G: Optional[nx.Graph] = None) -> int:
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        return sum(1 for node in graph.nodes() if graph.degree(node) > 0)

    def get_average_degree(self, G: Optional[nx.Graph] = None) -> float:
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0: return 0.0
        degrees = dict(filtered_graph.degree())
        return sum(degrees.values()) / len(degrees) if degrees else 0.0

    def get_average_path_length(self, G: Optional[nx.Graph] = None, use_largest_component: bool = True) -> float:
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0 or filtered_graph.number_of_edges() == 0: return 0.0
        
        if not nx.is_connected(filtered_graph):
            if use_largest_component:
                filtered_graph = self.get_largest_connected_component(filtered_graph)
            else:
                return 0.0
        if filtered_graph.number_of_nodes() < 2: return 0.0
        try: return nx.average_shortest_path_length(filtered_graph)
        except Exception: return 0.0

    def get_node_betweenness(self, G: Optional[nx.Graph] = None, normalized: bool = True, use_cache: bool = True) -> Dict[int, float]:
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        if use_cache and self._betweenness_cache is not None and normalized:
            return self._betweenness_cache
        betweenness = nx.betweenness_centrality(graph, normalized=normalized)
        if normalized and use_cache: self._betweenness_cache = betweenness
        return betweenness

    def get_edge_betweenness(self, G: Optional[nx.Graph] = None, normalized: bool = True, use_cache: bool = True) -> Dict[Tuple[int, int], float]:
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        if use_cache and self._edge_betweenness_cache is not None and normalized:
            return self._edge_betweenness_cache
        edge_betweenness = nx.edge_betweenness_centrality(graph, normalized=normalized)
        if normalized and use_cache: self._edge_betweenness_cache = edge_betweenness
        return edge_betweenness

    def get_average_betweenness(self, G: Optional[nx.Graph] = None, normalized: bool = False) -> float:
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        if graph.number_of_nodes() == 0: return 0.0
        betweenness = self.get_node_betweenness(graph, normalized=normalized, use_cache=True)
        return sum(betweenness.values()) / len(betweenness) if betweenness else 0.0

    def get_average_closeness(self, G: Optional[nx.Graph] = None) -> float:
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0: return 0.0
        
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
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0 or filtered_graph.number_of_edges() == 0: return 0.0
        try:
            partition = community_louvain.best_partition(filtered_graph)
            return community_louvain.modularity(partition, filtered_graph)
        except Exception as e:
            logger.warning(f"Error calculating modularity: {e}")
            return 0.0

    def get_clustering_coefficient(self, G: Optional[nx.Graph] = None) -> float:
        graph = G if G is not None else self.graph
        if graph is None: raise ValueError("No graph provided")
        filtered_graph = self.get_filtered_graph(graph)
        if filtered_graph.number_of_nodes() == 0: return 0.0
        try: return nx.average_clustering(filtered_graph)
        except Exception: return 0.0

    def calculate_all_properties(self, G: Optional[nx.Graph] = None, filter_isolated: bool = True) -> Dict[str, float]:
        graph_for_calculation = G if G is not None else self.graph
        if graph_for_calculation is None: raise ValueError("No graph provided")
        
        filtered_graph = self.get_filtered_graph(graph_for_calculation)
        if filtered_graph.number_of_nodes() == 0:
            return {"connected_nodes": 0, "degree": 0.0, "average_path": 0.0, "betweenness": 0.0, "closeness": 0.0, "modularity": 0.0, "clustering": 0.0}

        avg_betweenness = self.get_average_betweenness(graph_for_calculation, normalized=False)

        return {
            "connected_nodes": self.get_connected_nodes_count(graph_for_calculation),
            "degree": self.get_average_degree(graph_for_calculation),
            "average_path": self.get_average_path_length(graph_for_calculation, use_largest_component=True),
            "betweenness": avg_betweenness,
            "closeness": self.get_average_closeness(graph_for_calculation),
            "modularity": self.get_modularity(graph_for_calculation),
            "clustering": self.get_clustering_coefficient(graph_for_calculation),
        }

@app.command()
def main():
    logger.info("Features module. Use specialized CLI tools for analysis.")

if __name__ == "__main__":
    app()
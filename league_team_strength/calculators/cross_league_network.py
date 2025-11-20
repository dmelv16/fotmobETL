import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CrossLeagueNetworkAnalyzer:
    """
    Network analysis for inferring league relationships through indirect connections.
    Uses graph algorithms to estimate strength gaps between leagues without direct data.
    """
    
    def __init__(self, connection):
        self.conn = connection
        self.graph = None
    
    def build_league_network(self, season_year: int) -> nx.Graph:
        """
        Build a network graph of league relationships.
        
        Nodes = leagues
        Edges = strength relationships (from transfers or European matches)
        Edge weights = confidence in relationship
        """
        # Get all league network edges
        query = """
            SELECT 
                league_a_id,
                league_b_id,
                strength_gap_estimate,
                gap_confidence,
                total_transfers,
                european_matches
            FROM [dbo].[league_network_edges]
            WHERE season_year = ?
              AND strength_gap_estimate IS NOT NULL
        """
        
        df = pd.read_sql(query, self.conn, params=[season_year])
        
        if len(df) == 0:
            logger.warning(f"No league network data for season {season_year}")
            return nx.Graph()
        
        # Build graph
        G = nx.Graph()
        
        for _, row in df.iterrows():
            league_a = row['league_a_id']
            league_b = row['league_b_id']
            gap = row['strength_gap_estimate']
            confidence = row['gap_confidence']
            sample_size = (row['total_transfers'] or 0) + (row['european_matches'] or 0)
            
            # Add edge with attributes
            G.add_edge(league_a, league_b, 
                      gap=gap, 
                      confidence=confidence,
                      sample_size=sample_size,
                      weight=confidence)  # Weight by confidence for pathfinding
        
        logger.info(f"Built league network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        self.graph = G
        return G
    
    def find_league_gap(self, league_a_id: int, league_b_id: int, 
                       season_year: int) -> Optional[Dict]:
        """
        Find estimated strength gap between two leagues using network paths.
        
        Returns: Dict with gap estimate, confidence, and method
        """
        if self.graph is None:
            self.build_league_network(season_year)
        
        if league_a_id not in self.graph or league_b_id not in self.graph:
            logger.warning(f"Leagues {league_a_id} or {league_b_id} not in network")
            return None
        
        # Check if direct edge exists
        if self.graph.has_edge(league_a_id, league_b_id):
            edge_data = self.graph[league_a_id][league_b_id]
            return {
                'league_a_id': league_a_id,
                'league_b_id': league_b_id,
                'gap_estimate': edge_data['gap'],
                'confidence': edge_data['confidence'],
                'method': 'direct',
                'path_length': 1
            }
        
        # Find shortest path
        try:
            path = nx.shortest_path(self.graph, league_a_id, league_b_id, weight='weight')
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between leagues {league_a_id} and {league_b_id}")
            return None
        
        # Calculate cumulative gap along path
        cumulative_gap = 1.0
        cumulative_confidence = 1.0
        
        for i in range(len(path) - 1):
            node_a = path[i]
            node_b = path[i + 1]
            
            edge_data = self.graph[node_a][node_b]
            
            # Determine direction
            if node_a < node_b:
                # Forward direction
                cumulative_gap *= edge_data['gap']
            else:
                # Reverse direction (invert gap)
                cumulative_gap /= edge_data['gap']
            
            # Confidence decays with path length
            cumulative_confidence *= edge_data['confidence'] * 0.85  # 15% decay per hop
        
        return {
            'league_a_id': league_a_id,
            'league_b_id': league_b_id,
            'gap_estimate': cumulative_gap,
            'confidence': cumulative_confidence,
            'method': 'inferred',
            'path_length': len(path) - 1,
            'path': path
        }
    
    def fill_missing_edges(self, season_year: int, max_path_length: int = 3):
        """
        Fill in missing edges in the network using path inference.
        Only infers edges where path length <= max_path_length.
        """
        if self.graph is None:
            self.build_league_network(season_year)
        
        nodes = list(self.graph.nodes())
        cursor = self.conn.cursor()
        
        inferred_edges = []
        
        for i, league_a in enumerate(nodes):
            for league_b in nodes[i+1:]:
                
                # Skip if direct edge exists
                if self.graph.has_edge(league_a, league_b):
                    continue
                
                # Try to find path
                gap_info = self.find_league_gap(league_a, league_b, season_year)
                
                if gap_info and gap_info['path_length'] <= max_path_length:
                    inferred_edges.append(gap_info)
                    
                    # Save to database
                    cursor.execute("""
                        INSERT INTO [dbo].[league_network_edges] (
                            league_a_id, league_b_id, season_year,
                            strength_gap_estimate, gap_confidence, gap_method
                        ) VALUES (?, ?, ?, ?, ?, 'inferred')
                    """, league_a, league_b, season_year,
                         gap_info['gap_estimate'], gap_info['confidence'])
        
        self.conn.commit()
        logger.info(f"Inferred {len(inferred_edges)} missing edges")
        
        return inferred_edges
    
    def get_league_centrality(self, season_year: int) -> Dict[int, float]:
        """
        Calculate centrality measures for leagues (how connected/important they are).
        
        Returns: Dict of {league_id: centrality_score}
        """
        if self.graph is None:
            self.build_league_network(season_year)
        
        # Use betweenness centrality (how often league is on shortest paths)
        centrality = nx.betweenness_centrality(self.graph, weight='weight')
        
        return centrality
    
    def detect_league_clusters(self, season_year: int) -> Dict[int, int]:
        """
        Detect clusters of closely connected leagues.
        
        Returns: Dict of {league_id: cluster_id}
        """
        if self.graph is None:
            self.build_league_network(season_year)
        
        # Use community detection
        from networkx.algorithms import community
        
        communities = community.greedy_modularity_communities(self.graph, weight='weight')
        
        league_clusters = {}
        for cluster_id, community_set in enumerate(communities):
            for league_id in community_set:
                league_clusters[league_id] = cluster_id
        
        return league_clusters
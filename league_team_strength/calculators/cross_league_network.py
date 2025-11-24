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
    
    def build_league_network(self, season_year: int) -> nx.DiGraph:
        """
        Build a DIRECTED network graph of league relationships.
        
        Nodes = leagues
        Edges = strength relationships (from transfers or European matches)
        Edge direction = from source to destination
        Edge weight = strength gap (dest_strength / source_strength)
        """
        season_year = int(season_year)
        
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
            logger.warning(f"No league edges found for season {season_year}")
            return nx.DiGraph()
        
        # Get leagues that actually had matches this season
        unique_leagues = set(df['league_a_id'].unique()) | set(df['league_b_id'].unique())
        unique_leagues = [int(league_id) for league_id in unique_leagues]
        
        if not unique_leagues:
            return nx.DiGraph()
        
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(unique_leagues))
        cursor.execute(f"""
            SELECT DISTINCT parent_league_id 
            FROM [dbo].[match_details] 
            WHERE YEAR(match_time_utc) = ? 
            AND parent_league_id IN ({placeholders})
        """, season_year, *unique_leagues)
        
        leagues_with_matches = {int(row[0]) for row in cursor.fetchall()}
        
        # Filter out leagues without matches
        df = df[
            df['league_a_id'].isin(leagues_with_matches) & 
            df['league_b_id'].isin(leagues_with_matches)
        ]
        
        if len(df) == 0:
            logger.warning(f"No valid league edges after filtering for season {season_year}")
            return nx.DiGraph()
        
        # CRITICAL: Use DiGraph to maintain directionality
        G = nx.DiGraph()
        
        for _, row in df.iterrows():
            league_a = int(row['league_a_id'])
            league_b = int(row['league_b_id'])
            gap = float(row['strength_gap_estimate'])
            confidence = float(row['gap_confidence'])
            sample_size = int((row['total_transfers'] or 0) + (row['european_matches'] or 0))
            
            # Validate gap
            if gap <= 0 or not np.isfinite(gap):
                logger.warning(f"Invalid gap {gap} for edge {league_a}->{league_b}")
                continue
            
            # Add DIRECTED edge from A to B
            # gap = strength_b / strength_a
            G.add_edge(league_a, league_b,
                      gap=gap,
                      confidence=confidence,
                      sample_size=sample_size,
                      weight=1.0/confidence)  # Weight for pathfinding (lower = better)
            
            # Add reverse edge with inverted gap
            # From B to A: gap = strength_a / strength_b = 1/gap
            G.add_edge(league_b, league_a,
                      gap=1.0/gap,
                      confidence=confidence,
                      sample_size=sample_size,
                      weight=1.0/confidence)
        
        self.graph = G
        logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} directed edges")
        return G
    
    def find_league_gap(self, league_a_id: int, league_b_id: int, 
                       season_year: int, max_path_length: int = 3) -> Optional[Dict]:
        """
        Find estimated strength gap between two leagues using network paths.
        
        Returns gap as: strength_b / strength_a
        
        Args:
            league_a_id: Source league
            league_b_id: Destination league  
            season_year: Season to analyze
            max_path_length: Maximum path length to consider
        """
        league_a_id = int(league_a_id)
        league_b_id = int(league_b_id)
        season_year = int(season_year)
        
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.build_league_network(season_year)
        
        if self.graph.number_of_nodes() == 0:
            return None
        
        if league_a_id not in self.graph or league_b_id not in self.graph:
            return None
        
        # Check if direct edge exists
        if self.graph.has_edge(league_a_id, league_b_id):
            edge_data = self.graph[league_a_id][league_b_id]
            return {
                'league_a_id': league_a_id,
                'league_b_id': league_b_id,
                'gap_estimate': float(edge_data['gap']),
                'confidence': float(edge_data['confidence']),
                'method': 'direct',
                'path_length': 1,
                'path': [league_a_id, league_b_id]
            }
        
        # Find shortest path (by weight = 1/confidence)
        try:
            path = nx.shortest_path(self.graph, league_a_id, league_b_id, weight='weight')
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
        
        # Check path length limit
        path_length = len(path) - 1
        if path_length > max_path_length:
            return None
        
        # Calculate cumulative gap along path
        cumulative_gap = 1.0
        cumulative_confidence = 1.0
        
        for i in range(len(path) - 1):
            source_node = path[i]
            dest_node = path[i + 1]
            
            if not self.graph.has_edge(source_node, dest_node):
                logger.warning(f"Path contains non-existent edge: {source_node}->{dest_node}")
                return None
            
            edge_data = self.graph[source_node][dest_node]
            
            # Since graph is directed, gap is already in correct direction
            cumulative_gap *= edge_data['gap']
            
            # Confidence decays with path length (15% decay per hop)
            cumulative_confidence *= edge_data['confidence'] * (0.85 ** i)
        
        # Sanity check
        if not np.isfinite(cumulative_gap) or cumulative_gap <= 0:
            logger.warning(f"Invalid cumulative gap {cumulative_gap} for path {path}")
            return None
        
        return {
            'league_a_id': league_a_id,
            'league_b_id': league_b_id,
            'gap_estimate': float(cumulative_gap),
            'confidence': float(cumulative_confidence),
            'method': 'inferred',
            'path_length': path_length,
            'path': [int(node) for node in path]
        }
    
    def fill_missing_edges(self, season_year: int, max_path_length: int = 3) -> List[Dict]:
        """
        Fill in missing edges in the network using path inference.
        Only infers edges where path length <= max_path_length.
        """
        season_year = int(season_year)
        
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.build_league_network(season_year)
        
        if self.graph.number_of_nodes() == 0:
            logger.info(f"No league network found for season {season_year}")
            return []
        
        if self.graph.number_of_edges() < 3:
            logger.info(f"Insufficient edges ({self.graph.number_of_edges()}) for inference in season {season_year}")
            return []
        
        nodes = list(self.graph.nodes())
        cursor = self.conn.cursor()
        inferred_edges = []
        
        # Only look at undirected pairs to avoid duplicates
        for i, league_a in enumerate(nodes):
            for league_b in nodes[i+1:]:
                
                # Check if we have BOTH directions
                has_a_to_b = self.graph.has_edge(league_a, league_b)
                has_b_to_a = self.graph.has_edge(league_b, league_a)
                
                # Skip if we already have a direct edge in either direction
                if has_a_to_b or has_b_to_a:
                    continue
                
                # Try to find path from A to B
                gap_info = self.find_league_gap(league_a, league_b, season_year, max_path_length)
                
                if gap_info and gap_info['confidence'] > 0.2:  # Minimum confidence threshold
                    inferred_edges.append(gap_info)
                    
                    # Insert into database
                    cursor.execute("""
                        IF NOT EXISTS (
                            SELECT 1 FROM [dbo].[league_network_edges]
                            WHERE league_a_id = ? AND league_b_id = ? AND season_year = ?
                        )
                        BEGIN
                            INSERT INTO [dbo].[league_network_edges] (
                                league_a_id, league_b_id, season_year,
                                strength_gap_estimate, gap_confidence, gap_method,
                                last_updated
                            ) VALUES (?, ?, ?, ?, ?, 'inferred', GETDATE())
                        END
                    """, (
                        int(league_a), int(league_b), int(season_year),
                        int(league_a), int(league_b), int(season_year),
                        float(gap_info['gap_estimate']), float(gap_info['confidence'])
                    ))
        
        if inferred_edges:
            self.conn.commit()
            logger.info(f"Inferred {len(inferred_edges)} missing edges for season {season_year}")
        else:
            logger.info(f"No new edges inferred for season {season_year}")
        
        return inferred_edges
    
    def get_league_centrality(self, season_year: int) -> Dict[int, float]:
        """
        Calculate centrality measures for leagues (how connected/important they are).
        
        Returns: Dict of {league_id: centrality_score}
        """
        season_year = int(season_year)
        
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.build_league_network(season_year)
        
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Convert to undirected for centrality calculation
        G_undirected = self.graph.to_undirected()
        
        # Use betweenness centrality (how often league is on shortest paths)
        try:
            centrality = nx.betweenness_centrality(G_undirected, weight='weight')
            return {int(k): float(v) for k, v in centrality.items()}
        except:
            logger.warning("Failed to calculate centrality")
            return {}
    
    def detect_league_clusters(self, season_year: int) -> Dict[int, int]:
        """
        Detect clusters of closely connected leagues.
        
        Returns: Dict of {league_id: cluster_id}
        """
        season_year = int(season_year)
        
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.build_league_network(season_year)
        
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected()
        
        try:
            from networkx.algorithms import community
            
            communities = community.greedy_modularity_communities(G_undirected, weight='weight')
            
            league_clusters = {}
            for cluster_id, community_set in enumerate(communities):
                for league_id in community_set:
                    league_clusters[int(league_id)] = cluster_id
            
            return league_clusters
        except Exception as e:
            logger.warning(f"Failed to detect clusters: {e}")
            return {}
    
    def validate_network_consistency(self, season_year: int, tolerance: float = 0.3) -> List[Dict]:
        """
        Check for triangular inconsistencies in the network.
        E.g., if A->B gap is 1.5, B->C gap is 1.2, then A->C should be ~1.8
        
        Returns list of inconsistent triangles.
        """
        season_year = int(season_year)
        
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.build_league_network(season_year)
        
        if self.graph.number_of_nodes() < 3:
            return []
        
        inconsistencies = []
        nodes = list(self.graph.nodes())
        
        # Check all triangles
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes[i+1:], i+1):
                for k, c in enumerate(nodes[j+1:], j+1):
                    
                    # Check if we have all three edges
                    if not (self.graph.has_edge(a, b) and 
                           self.graph.has_edge(b, c) and 
                           self.graph.has_edge(a, c)):
                        continue
                    
                    # Get gaps
                    gap_ab = self.graph[a][b]['gap']
                    gap_bc = self.graph[b][c]['gap']
                    gap_ac = self.graph[a][c]['gap']
                    
                    # Expected: gap_ac = gap_ab * gap_bc
                    expected_gap_ac = gap_ab * gap_bc
                    
                    # Check if consistent within tolerance
                    ratio = gap_ac / expected_gap_ac if expected_gap_ac > 0 else 0
                    
                    if ratio < (1 - tolerance) or ratio > (1 + tolerance):
                        inconsistencies.append({
                            'triangle': (int(a), int(b), int(c)),
                            'gap_ab': float(gap_ab),
                            'gap_bc': float(gap_bc),
                            'gap_ac_actual': float(gap_ac),
                            'gap_ac_expected': float(expected_gap_ac),
                            'ratio': float(ratio),
                            'inconsistency': abs(1 - ratio)
                        })
        
        if inconsistencies:
            logger.warning(f"Found {len(inconsistencies)} inconsistent triangles in season {season_year}")
        
        return inconsistencies
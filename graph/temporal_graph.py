import random
import pandas as pd
from collections import defaultdict

class TemporalGraph:
    """
    node_id=0: padding node
    edge_id=0: padding edge
    """
    def __init__(self,
            graph_df:pd.DataFrame,
            bipartite:bool=False
        ):
        """
        Input:
            graph_df: pd.DataFrame, sorted by event time
            node_ft: np.ndarray, (N+1,node_dim)
            edge_ft: np.ndarray, (E+1,edge_dim)
            node_dim: int
            edge_dim: int
        """
        # set graph_df, adj, adj_t, edge_events
        self.graph_df=graph_df
        self.adj=defaultdict(list)
        self.adj_t=defaultdict(list)
        self.edge_events=[]
        for event in graph_df.itertuples(index=False): # col: [u,i,ts,idx=edge_id]
            src=int(event.u)
            dst=int(event.i)
            t=float(event.ts)
            edge_id=int(event.idx)
            # edge 양방향 저장
            self.adj[dst].append((src,edge_id))
            self.adj_t[dst].append(t)
            self.adj[src].append((dst,edge_id))
            self.adj_t[src].append(t)
            # edge event 저장
            self.edge_events[(src,dst,t,edge_id)]

        # set n_node, n_event, bipartite, max_u, max_t
        self.n_node=max(graph_df["u"].max(),graph_df["i"].max())
        self.n_event=graph_df["idx"].max()
        self.bipartite=bipartite
        self.max_u=graph_df["u"].max()
        self.max_t=graph_df["ts"].max()

    def set_random_seed(self,
            seed:int
        ):
        self.rng=random.Random(seed)

    def get_num_node(self):
        return self.n_node

    def get_num_event(self):
        return self.n_event

    def TR_sampling(self,

        ):
        """
        """
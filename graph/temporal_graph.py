import random
import pandas as pd
import bisect
import torch
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
            self.edge_events.append((src,dst,t,edge_id))

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

    def compute_TR_step(self,
            Q:set,
            TR_info:dict
        ):
        """
        양방향 그래프를 가정하기 때문에 self.adj로 이웃 노드들을 가져온다.
        self.adj와 self.adj_t는 이미 시간순 오름차순 정렬되어 있다.
        """
        Q_next=set()
        for node in Q:
            idx=bisect.bisect_right(self.adj_t[node],TR_info[node]["last_t"]) # adj_t 중 node의 last_t 보다 큰 시간 값의 첫번째 인덱스
            if idx==len(self.adj_t[node]):
                continue # 해당 값이 없는 경우 리스트 길이 반환하는 점 이용하여 다음 노드로 넘어감
            for (neighbor,_),event_t in zip(self.adj[node][idx:],self.adj_t[node][idx:]):
                if event_t<TR_info[neighbor]["last_t"]:
                    TR_info[neighbor]["r"]=1
                    TR_info[neighbor]["last_t"]=event_t
                    TR_info[neighbor]["hop"]=TR_info[node]["hop"]+1
                    if TR_info[node]["hop"]==0:
                        TR_info[neighbor]["first_t"]=event_t
                    else:
                        TR_info[neighbor]["first_t"]=TR_info[node]["first_t"]
                    Q_next.add(neighbor)
        return Q_next

    def compute_TR(self,
            source:int
        ):
        """
        Compute Temporal Reachability Info using Temporal BFS.
        
        TR_info value:
            r: temporal reachability result, 도달 가능하면 1 아니면 0
            last_t: source -> target path 중 마지막 상호작용 시간
            first_t: source -> target path 중 첫 번째 상호작용 시간
            hop: source -> target path hop 수
        
        Return:
            TR_info: dict 
                key: node id
                value: {
                    "r":int,
                    "last_t":float,
                    "first_t":float,
                    "hop":int
                }
        """
        INF=float('inf')
        TR_info={
            node: {
                "r": 0,
                "last_t":INF,
                "first_t":INF,
                "hop": 0
            }
            for node in range(1,self.n_node+1)
        }
        TR_info[source]={
            "r": 1,
            "last_t":float("-inf"), # 첫 조회 시 event_t=0.0 인 경우도 탐색을 위해서 -inf 값 설정
            "first_t":0.0,
            "hop": 0
        }
        Q=set()
        Q.add(source)
        while Q:
            Q=self.compute_TR_step(
                Q=Q,
                TR_info=TR_info
            )
        return TR_info

    def TR_sampling(self,
            n_sample:int,
            n_path:int
        ):
        """
        n_sample 개의 pos_pair, neg_pair 각각 생성.
            all pos_pair: n_sample개
            all neg_pair: n_sample개
        
        임의의 source node 선택 후 n_path개 만큼의 pos,neg pair 생성.
            pos_pair: n_path개
            neg_pair: n_path개
            pos_pair의 개수 k_pos < n_path 인 경우 다음과 같이 생성.
                pos_pair: k_pos개
                neg_pair: k_pos개 

        n_sample개 다 채워지면 source node 선택 종료.

        선택된 pair sample들의 정보를 pair_info dict로 반환.
        pair_info: dict
            key: (src,dst)
            value:
                r: 1 or 0
                last_t: src->dst path의 마지막 event_t
                first_t: src->dst path의 첫 번째 event_t
                hop: src->dst path의 hop 수
        
        Input:
            n_sample: int, 
            n_path: int,

        Return:
            pos_pair: dict
                key: src, dst
                value: 
                    src: [n_sample,] 
                    dst: [n_sample,] 
            neg_pair: dict
                key: src, dst
                value: 
                    src: [n_sample,] 
                    dst: [n_sample,]
            pair_info: dict
                key: (src,dst)
                value:
                    r:int
                    last_t:float
                    first_t:float
                    hop:int

        쿼리 시점들에 대한 샘플링은 어떻게 수행할 것인가?
        """
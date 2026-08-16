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
            t=float(event.t)
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
        self.max_t=graph_df["t"].max()

    def set_random_seed(self,
            seed:int
        ):
        self.rng=random.Random(seed)

    def get_num_node(self):
        return self.n_node

    def get_num_event(self):
        return self.n_event

    def compute_TR(self,
            source:int,
            query_time:float|None=None,
            max_hop:int|None=None
        ):
        """
        Compute Temporal Reachability using hop-layer Temporal BFS.

        조건:
            - query_time 이전의 이벤트만 고려
            - temporal path는 시간 증가 조건을 만족해야 함
            - 최대 max_hop까지만 탐색
            - 각 node의 shortest temporal hop 계산
            - 같은 shortest hop 내에서는 earliest-arrival path를 대표로 저장

        max_hop 만큼만 step 반복하여 최대 hop 수 조정, None이면 가능한 최대 hop까지
        
        TR_info value:
            r: temporal reachable이면 1, 아니면 0
            last_t: shortest-hop path들 중 earliest-arrival path의 마지막 이벤트 시간
            first_t: shortest-hop path들 중 earliest-arrival path의 첫 번째 이벤트 시간
            hop: source -> target의 shortest temporal hop
        
        Return:
            TR_info: dict 
                key: node id
                value: {
                    "r":int,
                    "hop":int
                    "first_t":float,
                    "last_t":float
                }
        """
        INF=float("inf")
        NEG_INF=float("-inf")

        if max_hop is None:
            max_hop=self.n_node-1

        TR_info={
            n:{"r":0,"hop":INF,"first_t":INF,"last_t":INF}
            for n in range(1,self.n_node+1)
        }
        TR_info[source]={
            "r":1,
            "hop":0,
            "first_t":0.0,
            "last_t":NEG_INF
        }

        best_arrival={source:NEG_INF}
        cur_state={source:NEG_INF}
        cur_first={source:0.0}
        end_cache={}

        for hop in range(1,max_hop+1):
            next_state={}
            next_first={}
            for node,arrival_t in cur_state.items():
                times=self.adj_t[node]
                edges=self.adj[node]
                start_idx=bisect.bisect_right(times,arrival_t)

                if query_time is None:
                    end_idx=len(times)
                elif node in end_cache:
                    end_idx=end_cache[node]
                else:
                    end_idx=bisect.bisect_right(times,query_time)
                    end_cache[node]=end_idx

                for idx in range(start_idx,end_idx):
                    neighbor=edges[idx][0]
                    event_t=times[idx]

                    if best_arrival.get(neighbor,INF)<=event_t:
                        continue
                    if next_state.get(neighbor,INF)<=event_t:
                        continue

                    next_state[neighbor]=event_t
                    next_first[neighbor]=event_t if hop==1 else cur_first[node]

            if not next_state:
                break

            for node,arrival_t in next_state.items():
                best_arrival[node]=arrival_t
                if TR_info[node]["r"]==0:
                    TR_info[node]={
                        "r":1,
                        "hop":hop,
                        "first_t":next_first[node],
                        "last_t":arrival_t
                    }
            cur_state=next_state
            cur_first=next_first
        return TR_info

    def random_TR_sampling(self,
            n_sample:int,
            n_pair:int,
            query_time:float|None=None,
            max_hop:int|None=5
        ):
        """
        각 source를 최대 한 번 사용하여 balanced TR query를 random sampling.

        Input:
            n_sample: 생성할 positive/negative pair의 목표 개수
            n_pair: 각 source에서 생성할 positive/negative pair의 최대 개수
            query_time: query_time 이하의 event만 사용
            max_hop: temporal reachability의 최대 hop

        Return:
            src: [N,] long tensor
            dst: [N,] long tensor
            label: [N,] float tensor
            pos_mask: [N,] bool tensor
            pair_info: 각 pair의 TR 정보
        """
        pos_pairs=[]
        neg_pairs=[]
        pair_info={}
        nodes=list(range(1,self.n_node+1))
        sources=nodes.copy()
        self.rng.shuffle(sources)
        for src in sources:
            if n_sample<=len(pos_pairs):
                break

            TR_info=self.compute_TR(src,query_time,max_hop)
            pos=[n for n in nodes if n!=src and TR_info[n]["r"]==1]
            neg=[n for n in nodes if n!=src and TR_info[n]["r"]==0]

            n=min(n_pair,n_sample-len(pos_pairs),len(pos),len(neg))
            if n==0:
                continue

            for dst in self.rng.sample(pos,n):
                pos_pairs.append((src,dst))
                pair_info[(src,dst)]=TR_info[dst].copy()

            for dst in self.rng.sample(neg,n):
                neg_pairs.append((src,dst))
                pair_info[(src,dst)]=TR_info[dst].copy()

        pairs=pos_pairs+neg_pairs
        label=torch.tensor(
            [1]*len(pos_pairs)+[0]*len(neg_pairs),
            dtype=torch.float32
        )

        return {
            "src":torch.tensor([s for s,_ in pairs],dtype=torch.long),
            "dst":torch.tensor([d for _,d in pairs],dtype=torch.long),
            "label":label,
            "pos_mask":label.bool(),
            "pair_info":[pair_info[p] for p in pairs]
        }




    def new_TR_sampling(self,
            source:torch.Tensor,
            n_sample:int,
            query_time:float|None=None,
            max_hop:int|None=5
        ):
        """
        batch 내의 event들의 src, dst 노드들을 source 로 사용할 경우

        Input:
            source: [n_src,] long tensor
            n_sample: int, 각 src마다 생성할 positve/negative sample 개수
        Return:
        """
        pos_pairs=[]
        neg_pairs=[]
        pair_info={}
        nodes=list(range(1,self.n_node+1))
        for src in source.detach().cpu().tolist():
            TR_info=self.compute_TR(
                source=src,
                query_time=query_time,
                max_hop=max_hop
            )
            pos_nodes=[
                n for n in nodes
                if n!=src and TR_info[n]["r"]==1
            ]
            neg_nodes=[
                n for n in nodes
                if n!=src and TR_info[n]["r"]==0
            ]
            pos_dst=self.rng.sample(
                pos_nodes,
                min(len(pos_nodes),n_sample)
            )
            neg_dst=self.rng.sample(
                neg_nodes,
                min(len(neg_nodes),n_sample)
            )
            for dst in pos_dst:
                pos_pairs.append((src,dst))
                pair_info[(src,dst)]=TR_info[dst].copy()
            for dst in neg_dst:
                neg_pairs.append((src,dst))
                pair_info[(src,dst)]=TR_info[dst].copy()
        pos_pair={
            "src":torch.tensor([s for s,_ in pos_pairs],dtype=torch.long),
            "dst":torch.tensor([d for _,d in pos_pairs],dtype=torch.long)
        }
        neg_pair={
            "src":torch.tensor([s for s,_ in neg_pairs],dtype=torch.long),
            "dst":torch.tensor([d for _,d in neg_pairs],dtype=torch.long)
        }
        return {
            "pos_pair":pos_pair,
            "neg_pair":neg_pair,
            "pair_info":pair_info
        }
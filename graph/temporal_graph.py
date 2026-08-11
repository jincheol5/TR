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

    def compute_TR(self,
            source:int,
            query_time:float,
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
        INF=float('inf')
        if max_hop is None:
            max_hop=self.n_node-1
        TR_info={
            node: {
                "r":0,
                "hop":INF,
                "first_t":INF,
                "last_t":INF
            }
            for node in range(1,self.n_node+1)
        }
        TR_info[source]={
            "r":1,
            "hop":0,
            "first_t":0.0,
            "last_t":float("-inf") # 첫 조회 시 event_t=0.0 인 경우도 탐색을 위해서 -inf 값 설정
        }

        ### Step loop
        # layer state: 현재 hop에서 node에 도달하는 earliest arrival time
        # layer first_t: 현재 hop의 earliest-arrival path의 첫 이벤트 시간
        cur_layer_state={ # 0-hop layer state
            source:float("-inf")
        }
        cur_layer_first_t={ 
            source:0.0
        }
        for hop in range(1,max_hop+1):
            next_layer_state={}
            next_layer_first_t={}
            for node,arrival_t in cur_layer_state.items():
                # arrival_t보다 늦은 첫 이벤트
                start_idx=bisect.bisect_right(
                    self.adj_t[node],
                    arrival_t
                )
                # arrival_t보다 늦은 이벤트가 없는 경우
                if start_idx==len(self.adj_t[node]):
                    continue

                # query_time보다 늦은 첫 이벤트
                end_idx=bisect.bisect_right(
                    self.adj_t[node],
                    query_time
                )
                for (neighbor,_),event_t in zip(
                        self.adj[node][start_idx:end_idx],
                        self.adj_t[node][start_idx:end_idx]
                    ):
                    # 현재 hop에서 neighbor에 더 빨리 도착하는 path만 유지
                    if neighbor in next_layer_state and next_layer_state[neighbor]<=event_t:
                        continue
                    next_layer_state[neighbor]=event_t

                    # 현재 path의 first_t 전달
                    if hop==1:
                        next_layer_first_t[neighbor]=event_t
                    else:
                        next_layer_first_t[neighbor]=cur_layer_first_t[node]

            # next_layer_state가 비어있으면 종료
            if not next_layer_state:
                break

            # 이번 hop에서 처음 발견된 node만 shortest-hop 결과로 TR_info에 저장
            # 현재 hop의 모든 후보를 비교해서 earliest arrival만 남긴 뒤 저장
            for neighbor,event_t in next_layer_state.items():
                if TR_info[neighbor]["r"]==0:
                    TR_info[neighbor]["r"]=1
                    TR_info[neighbor]["hop"]=hop
                    TR_info[neighbor]["first_t"]=next_layer_first_t[neighbor]
                    TR_info[neighbor]["last_t"]=event_t

            # 다음 hop layer로 이동
            cur_layer_state=next_layer_state
            cur_layer_first_t=next_layer_first_t
        return TR_info

    def random_TR_sampling(self,
            n_sample:int,
            query_time:float,
            max_hop:int=5
        ):
        """
        """
        n_pos=n_sample
        n_neg=n_sample

        pos_pairs=[]
        neg_pairs=[]
        pair_info={}

        TR_cache={}
        nodes=list(range(1,self.n_node+1))
        while len(pos_pairs)<n_pos or len(neg_pairs)<n_neg:
            src=self.rng.choice(nodes)
            dst=self.rng.choice(nodes)
            if src==dst:
                continue
            if (src,dst) in pair_info:
                continue
            if src not in TR_cache:
                TR_cache[src]=self.compute_TR(
                    source=src,
                    query_time=query_time,
                    max_hop=max_hop
                )
            info=TR_cache[src][dst]
            if info["r"]==1:
                if len(pos_pairs)>=n_pos:
                    continue
                pos_pairs.append((src,dst))
            else:
                if len(neg_pairs)>=n_neg:
                    continue
                neg_pairs.append((src,dst))
            pair_info[(src,dst)]=info.copy()

        pos_pair={
            "src":torch.tensor(
                [src for src,_ in pos_pairs],
                dtype=torch.long
            ),
            "dst":torch.tensor(
                [dst for _,dst in pos_pairs],
                dtype=torch.long
            )
        }
        neg_pair={
            "src":torch.tensor(
                [src for src,_ in neg_pairs],
                dtype=torch.long
            ),
            "dst":torch.tensor(
                [dst for _,dst in neg_pairs],
                dtype=torch.long
            )
        }
        return pos_pair,neg_pair,pair_info

    def TR_sampling(self,
            n_sample:int,
            n_path:int,
            max_hop:int,
            query_time:float
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
                hop: src->dst path의 hop 수
                first_t: src->dst path의 첫 번째 event_t
                last_t: src->dst path의 마지막 event_t
        
        Input:
            n_sample: int, 
            n_path: int,
            max_hop: int,
            query_time: float
            
        Return:
            pos_pair: dict
                key: src, dst
                value: 
                    src: [n_sample,] long tensor
                    dst: [n_sample,] long tensor
            neg_pair: dict
                key: src, dst
                value: 
                    src: [n_sample,] long tensor
                    dst: [n_sample,] long tensor
            pair_info: dict
                key: (src,dst)
                value:
                    r:int
                    hop:int
                    first_t:float
                    last_t:float
        """
        pos_src=[]
        pos_dst=[]
        neg_src=[]
        neg_dst=[]
        pair_info={}
        nodes=list(range(1,self.n_node+1))
        while len(pos_src)<n_sample:
            source=self.rng.choice(nodes)
            TR_info=self.compute_TR(
                source=source,
                query_time=query_time,
                max_hop=max_hop
            )
            pos_targets=[
                node
                for node in nodes
                if node!=source and TR_info[node]["r"]==1
            ]
            neg_targets=[
                node
                for node in nodes
                if node!=source and TR_info[node]["r"]==0
            ]
            if not pos_targets or not neg_targets:
                continue
            k=min(
                n_path,
                len(pos_targets),
                len(neg_targets),
                n_sample-len(pos_src)
            )
            sampled_pos_tar=self.rng.sample(pos_targets,k)
            sampled_neg_tar=self.rng.sample(neg_targets,k)
            for dst in sampled_pos_tar:
                pos_src.append(source)
                pos_dst.append(dst)
                pair_info[(source,dst)]=TR_info[dst].copy()
            for dst in sampled_neg_tar:
                neg_src.append(source)
                neg_dst.append(dst)
                pair_info[(source,dst)]=TR_info[dst].copy()
        pos_pair={
            "src":torch.tensor(pos_src,dtype=torch.long),
            "dst":torch.tensor(pos_dst,dtype=torch.long)
        }
        neg_pair={
            "src":torch.tensor(neg_src,dtype=torch.long),
            "dst":torch.tensor(neg_dst,dtype=torch.long)
        }
        return pos_pair,neg_pair,pair_info
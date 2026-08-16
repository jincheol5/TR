import random
import bisect
import pandas as pd
import torch
from collections import defaultdict
from utils import DataUtils


class TemporalGraph:
    def __init__(self,
            graph_df:pd.DataFrame,
            bipartite:bool=False
        ):
        self.graph_df=graph_df
        self.adj=defaultdict(list)
        self.adj_t=defaultdict(list)
        self.edge_events=[]

        for event in graph_df.itertuples(index=False):
            src=int(event.u)
            dst=int(event.i)
            t=float(event.t)
            edge_id=int(event.idx)

            self.adj[src].append((dst,edge_id))
            self.adj_t[src].append(t)
            self.adj[dst].append((src,edge_id))
            self.adj_t[dst].append(t)

            self.edge_events.append(
                (src,dst,t,edge_id)
            )

        self.n_node=int(
            max(
                graph_df["u"].max(),
                graph_df["i"].max()
            )
        )
        self.n_event=int(graph_df["idx"].max())
        self.bipartite=bipartite
        self.max_u=int(graph_df["u"].max())
        self.max_t=float(graph_df["t"].max())

    def set_random_seed(self,
            seed:int
        ):
        self.rng=random.Random(seed)

    def compute_TR(self,
            source:int,
            query_time:float|None=None,
            max_hop:int|None=None
        ):
        INF=float("inf")
        NEG_INF=float("-inf")

        if max_hop is None:
            max_hop=self.n_node-1

        TR_info={
            n:{
                "r":0,
                "hop":INF,
                "first_t":INF,
                "last_t":INF
            }
            for n in range(1,self.n_node+1)
        }

        TR_info[source]={
            "r":1,
            "hop":0,
            "first_t":0.0,
            "last_t":NEG_INF
        }

        best_arrival={
            source:NEG_INF
        }
        cur_state={
            source:NEG_INF
        }
        cur_first={
            source:0.0
        }
        end_cache={}

        for hop in range(1,max_hop+1):
            next_state={}
            next_first={}

            for node,arrival_t in cur_state.items():
                times=self.adj_t[node]
                edges=self.adj[node]

                start_idx=bisect.bisect_right(
                    times,
                    arrival_t
                )

                if query_time is None:
                    end_idx=len(times)
                elif node in end_cache:
                    end_idx=end_cache[node]
                else:
                    end_idx=bisect.bisect_right(
                        times,
                        query_time
                    )
                    end_cache[node]=end_idx

                for idx in range(start_idx,end_idx):
                    neighbor=edges[idx][0]
                    event_t=times[idx]

                    if best_arrival.get(
                            neighbor,
                            INF
                        )<=event_t:
                        continue

                    if next_state.get(
                            neighbor,
                            INF
                        )<=event_t:
                        continue

                    next_state[neighbor]=event_t
                    next_first[neighbor]=(
                        event_t
                        if hop==1
                        else cur_first[node]
                    )

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
        pos_pairs=[]
        neg_pairs=[]
        cache={}
        nodes=list(
            range(1,self.n_node+1)
        )

        for _ in range(self.n_node*3):
            if len(pos_pairs)>=n_sample:
                break

            src=self.rng.choice(nodes)

            if src not in cache:
                TR_info=self.compute_TR(
                    source=src,
                    query_time=query_time,
                    max_hop=max_hop
                )

                pos=[
                    n
                    for n in nodes
                    if n!=src
                    and TR_info[n]["r"]==1
                ]
                neg=[
                    n
                    for n in nodes
                    if n!=src
                    and TR_info[n]["r"]==0
                ]

                self.rng.shuffle(pos)
                self.rng.shuffle(neg)

                cache[src]=(
                    TR_info,
                    pos,
                    neg
                )

            TR_info,pos,neg=cache[src]

            n=min(
                n_pair,
                n_sample-len(pos_pairs),
                len(pos),
                len(neg)
            )

            if n==0:
                continue

            pos_pairs.extend(
                (src,dst)
                for dst in pos[-n:]
            )
            neg_pairs.extend(
                (src,dst)
                for dst in neg[-n:]
            )

            del pos[-n:]
            del neg[-n:]

        pairs=pos_pairs+neg_pairs

        label=torch.tensor(
            [1]*len(pos_pairs)
            +[0]*len(neg_pairs),
            dtype=torch.float32
        )

        return {
            "src":torch.tensor(
                [s for s,_ in pairs],
                dtype=torch.long
            ),
            "dst":torch.tensor(
                [d for _,d in pairs],
                dtype=torch.long
            ),
            "label":label,
            "pos_mask":label.bool(),
            "pair_info":[
                cache[s][0][d].copy()
                for s,d in pairs
            ]
        }


def validate_sample(sample):
    n_src=len(sample["src"])
    n_dst=len(sample["dst"])
    n_label=len(sample["label"])
    n_info=len(sample["pair_info"])

    print("===== Shape Check =====")
    print(f"src       : {n_src}")
    print(f"dst       : {n_dst}")
    print(f"label     : {n_label}")
    print(f"pair_info : {n_info}")

    assert n_src==n_dst
    assert n_src==n_label
    assert n_src==n_info

    n_pos=int(
        (sample["label"]==1).sum().item()
    )
    n_neg=int(
        (sample["label"]==0).sum().item()
    )

    print("\n===== Label Distribution =====")
    print(f"Positive : {n_pos}")
    print(f"Negative : {n_neg}")

    n_mismatch=0

    print("\n===== label vs r Check =====")

    for i,(src,dst,label,info) in enumerate(
        zip(
            sample["src"],
            sample["dst"],
            sample["label"],
            sample["pair_info"]
        )
    ):
        label=int(label.item())
        r=int(info["r"])

        if label!=r:
            n_mismatch+=1
            print(
                f"Mismatch | "
                f"index={i}, "
                f"pair=({src.item()},{dst.item()}), "
                f"label={label}, "
                f"r={r}, "
                f"hop={info['hop']}, "
                f"first_t={info['first_t']}, "
                f"last_t={info['last_t']}"
            )

    pairs=list(
        zip(
            sample["src"].tolist(),
            sample["dst"].tolist()
        )
    )
    n_duplicate=len(pairs)-len(set(pairs))

    print("\n===== Result =====")
    print(f"Total samples : {n_label}")
    print(f"Positive      : {n_pos}")
    print(f"Negative      : {n_neg}")
    print(f"Mismatch      : {n_mismatch}")
    print(f"Duplicate     : {n_duplicate}")

    assert n_mismatch==0,(
        "label과 pair_info['r']가 일치하지 않습니다."
    )
    assert n_duplicate==0,(
        "중복된 (src,dst) pair가 존재합니다."
    )

    print("\nValidation Passed!")


if __name__=="__main__":
    # timestamp 오름차순으로 생성한 테스트 temporal graph

    data=DataUtils.preprocess_graph(dataset_name="enron")
    graph_df=data["graph_df"]

    graph=TemporalGraph(
        graph_df=graph_df
    )
    graph.set_random_seed(
        seed=42
    )

    sample=graph.random_TR_sampling(
        n_sample=10,
        n_pair=2,
        query_time=10,
        max_hop=5
    )

    validate_sample(sample)
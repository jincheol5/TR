import os
import pandas as pd
import numpy as np
from typing import Literal

class DataUtils:
    base_path=os.path.join('..','data','temporal_graph')

    @staticmethod
    def preprocess_snap_dataset(
            dataset_name:Literal[
                "CollegeMsg"
                "bitcoin-otc",
                "bitcoin-alpha"
            ]
        ):
        """
        """
        match dataset_name:
            case "CollegeMsg":
                dataset_path=os.path.join(DataUtils.base_path,dataset_name,f"raw_{dataset_name}.txt")
                df=pd.read_csv(
                    dataset_path,
                    header=None,
                    sep=r"\s+",
                    usecols=[0,1,2],
                    names=["u","i","ts"],
                )
            case "bitcoin-otc"|"bitcoin-alpha":
                dataset_path=os.path.join(DataUtils.base_path,dataset_name,f"raw_{dataset_name}.csv")
                df=pd.read_csv(
                    dataset_path,
                    header=None,
                    usecols=[0,1,3],
                    names=["u","i","ts"],
                )

        # 결측값 제거
        df=df.dropna(
            subset=["u","i","ts"]
        ).copy()

        # 자료형 변환
        df["u"]=df["u"].astype(int)
        df["i"]=df["i"].astype(int)
        df["ts"]=df["ts"].astype(float)

        # self-loop 제거
        self_loop_mask=df["u"]==df["i"]
        df=df.loc[~self_loop_mask].copy()

        # 모든 node ID 수집
        node_ids=sorted(
            set(df["u"]).union(df["i"])
        )

        # 기존 node ID -> 1 ~ N 매핑
        node_mapping={
            original_id:mapped_id
            for mapped_id,original_id in enumerate(
                node_ids,
                start=1,
            )
        }

        # node ID 재매핑
        df["u"]=df["u"].map(node_mapping).astype(int)
        df["i"]=df["i"].map(node_mapping).astype(int)

        # edge ID를 1 ~ E로 지정
        df["idx"]=range(1,len(df)+1)

        # 전처리된 파일 저장
        output_path=os.path.join(DataUtils.base_path,dataset_name,f"{dataset_name}.csv")
        df.to_csv(
            output_path,
            index=False,
            header=False,
        )
        print(f"Success to preprocess raw {dataset_name} to new .csv file!")

    @staticmethod
    def preprocess_zenodo_graph(
        dataset_name:Literal[
                "enron",
                "wikipedia",
                "reddit"
            ]
        ):
        """
        """
        bipartite_dataset={
            "CollegeMsg":False,
            "bitcoin-otc":False,
            "bitcoin-alpha":False,
            "enron":False,
            "wikipedia":True,
            "reddit":True
        }
        bipartite=bipartite_dataset[dataset_name]
        graph_path=os.path.join(DataUtils.base_path,dataset_name,f"ml_{dataset_name}.csv")
        node_ft_path=os.path.join(DataUtils.base_path,dataset_name,f"ml_{dataset_name}_node.npy")
        edge_ft_path=os.path.join(DataUtils.base_path,dataset_name,f"ml_{dataset_name}.npy")

        graph_df=pd.read_csv(
            graph_path,
            index_col=0,
        )[["u","i","ts","idx"]].copy()
        node_ft=np.load(node_ft_path)
        edge_ft=np.load(edge_ft_path)

        ### remove self-loop
        mask=graph_df["u"]!=graph_df["i"]
        new_graph_df=graph_df[mask].copy().reset_index(drop=True)

        ### remapping edge id
        # 기존 idx 기준으로 edge feature 선택
        old_edge_ids=new_graph_df["idx"].to_numpy()
        new_edge_ft=edge_ft[old_edge_ids]

        # padding edge feature 다시 추가
        new_edge_ft=np.vstack(
            [
                np.zeros((1,edge_ft.shape[1])),
                new_edge_ft
            ]
        )

        # idx 재부여
        new_graph_df["idx"]=np.arange(1,len(new_graph_df)+1)

        ### remapping node id
        used_node_ids=np.sort(
            pd.concat([new_graph_df["u"],new_graph_df["i"]])
            .unique()
            .astype(int)
        )
        node_id_map={
            old_id:new_id
            for new_id,old_id in enumerate(used_node_ids,start=1)
        }
        new_graph_df["u"]=new_graph_df["u"].map(node_id_map).astype(int)
        new_graph_df["i"]=new_graph_df["i"].map(node_id_map).astype(int)

        # Remap node features with the same node order
        new_node_ft=np.vstack([
            np.zeros((1,node_ft.shape[1]),dtype=node_ft.dtype),
            node_ft[used_node_ids]
        ])

        # 열 순서 최종 보장
        new_graph_df=new_graph_df[
            ["u","i","ts","idx"]
        ]

        # get n_node, node_dim, edge_dim
        n_node=max(new_graph_df["u"].max(),new_graph_df["i"].max())
        node_dim=new_node_ft.shape[1]
        edge_dim=new_edge_ft.shape[1]

        return {
            "graph_df":new_graph_df,
            "n_node":n_node,
            "bipartite":bipartite,
            "max_u":new_graph_df["u"].max(),
            "node_dim":node_dim,
            "edge_dim":edge_dim,
            "node_ft_np":new_node_ft,
            "edge_ft_np":new_edge_ft
        }

    @staticmethod
    def preprocess_snap_graph(
        dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-otc",
                "bitcoin-alpha"
            ]
        ):
        """
        self-loop 제거된 상태
        node_id remapping 된 상태
        node, edge feature 없음

        """
        bipartite_dataset={
            "CollegeMsg":False,
            "bitcoin-otc":False,
            "bitcoin-alpha":False,
            "enron":False,
            "wikipedia":True,
            "reddit":True
        }
        bipartite=bipartite_dataset[dataset_name]
        graph_path=os.path.join(DataUtils.base_path,dataset_name,f"{dataset_name}.csv")
        graph_df=pd.read_csv(graph_path,index_col=0)
        return {
            "graph_df":graph_df,
            "n_node":max(graph_df["u"].max(),graph_df["i"].max()),
            "bipartite":bipartite,
            "max_u":graph_df["u"].max(),
            "node_dim":None,
            "edge_dim":None,
            "node_ft":None,
            "edge_ft":None
        }


    @staticmethod
    def preprocess_graph(
            dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-otc",
                "bitcoin-alpha",
                "enron",
                "wikipedia",
                "reddit"
            ]
        ):
        """
        """
        match dataset_name:
            case "CollegeMsg"|"bitcoin-otc"|"bitcoin-alpha":
                return DataUtils.preprocess_snap_graph(dataset_name=dataset_name)
            case "enron"|"wikipedia"|"reddit":
                return DataUtils.preprocess_zenodo_graph(dataset_name=dataset_name)

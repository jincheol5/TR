import argparse
import torch
from utils import DataUtils,TrainUtils
from graph import TemporalGraph,CTDNE_Graph,ATDGEB_Graph

"""
<< Test >> 
data.temporal_graph
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TemporalGraph.compute_TR
            """
            dataset=DataUtils.preprocess_graph(dataset_name="enron")
            graph_df=dataset["graph_df"]
            graph=TemporalGraph(graph_df=graph_df)
            graph.set_random_seed(seed=1)
            TR_info=graph.compute_TR(
                source=1,
                max_hop=5
            )
            print(graph.get_num_node())

        case 2:
            """
            Test. TemporalGraph.random_TR_sampling
            """
            dataset=DataUtils.preprocess_graph(dataset_name="enron")
            graph_df=dataset["graph_df"]
            graph=TemporalGraph(graph_df=graph_df)
            graph.set_random_seed(seed=1)
            result=graph.random_TR_sampling(
                n_sample=100,
                n_pair=10,
                max_hop=5
            )
            print(result["src"].size())

        case 3:
            """
            Test. CTDNE_Graph.generate_walks
            """
            dataset=DataUtils.preprocess_graph(dataset_name="enron")
            graph_df=dataset["graph_df"]
            train_df,_,_=TrainUtils.split_graph_df(df=graph_df)
            graph=CTDNE_Graph(graph_df=graph_df,train_df=train_df)
            graph.set_random_seed(seed=1)

            # 하이퍼 파라미터
            n_node=graph.get_num_node()
            walk_len=10
            min_walk_len=3
            n_walk=10
            n_window=int(n_walk*n_node*(walk_len-min_walk_len+1))
            edge_sampling=f"uniform"
            neighbor_sampling=f"uniform"

            walks=graph.generate_walks(
                walk_len=walk_len,
                min_walk_len=min_walk_len,
                n_walk=n_walk,
                n_window=n_window,
                edge_sampling=edge_sampling,
                neighbor_sampling=neighbor_sampling
            )
            print(len(walks))
        case 4:
            """
            Test. ATDGEB_Graph.generate_walks
            """
            dataset=DataUtils.preprocess_graph(dataset_name="enron")
            graph_df=dataset["graph_df"]
            train_df,_,_=TrainUtils.split_graph_df(df=graph_df)
            graph=ATDGEB_Graph(graph_df=graph_df,train_df=train_df)
            graph.set_random_seed(seed=1)

            # 하이퍼 파라미터
            k_list=[2,4,6,8,10]
            n_aggr=2
            min_points=3
            walk_len=5
            n_lsbs=3

            walks=graph.generate_walks(
                k_list=k_list,
                n_aggr=n_aggr,
                min_points=min_points,
                walk_len=walk_len,
                n_lsbs=n_lsbs
            )
            print(walks)


if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_fn(**test_config)
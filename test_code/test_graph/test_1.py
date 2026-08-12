import argparse
import torch
from utils import DataUtils
from graph import TemporalGraph

"""
<< Test >> 
data.temporal_graph.TemporalGraph
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. data.temporal_graph.TemporalGraph.random_TR_sampling
            """
            dataset=DataUtils.preprocess_graph(dataset_name="enron")
            graph_df=dataset["graph_df"]
            bipartite=dataset["bipartite"]
            graph=TemporalGraph(
                graph_df=graph_df,
                bipartite=bipartite
            )
            graph.set_random_seed(seed=1)
            source=torch.tensor([1,2,3],dtype=torch.long)
            result=graph.random_TR_sampling(
                source=source,
                n_sample=3,
                max_hop=5
            )
            print(f"pos_pair:")
            print(result["pos_pair"])
            print(f"\nneg_pair:")
            print(result["neg_pair"])
            print(f"\npair_info:")
            print(result["pair_info"])

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
import argparse
import torch
from utils import DataUtils
from graph import CTDNE_Graph
from module import SkipGram

"""
<< Test >> 

"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. skip_gram.SkipGram
            """
            ### data
            dataset=DataUtils.preprocess_graph(dataset_name="enron")
            graph_df=dataset["graph_df"]
            graph=CTDNE_Graph(graph_df=graph_df)
            graph.set_random_seed(seed=1)

            # 하이퍼 파라미터
            n_node=graph.get_num_node()
            walk_len=10
            min_walk_len=3
            n_walk=10
            n_window=int(n_walk*n_node*(walk_len-min_walk_len+1))
            edge_sampling=f"uniform"
            neighbor_sampling=f"uniform"
            epoch=10

            ### model
            skip_gram=SkipGram(
                vector_size=128,
                window_size=min_walk_len
            )

            ### Walk 생성
            walks=graph.generate_walks(
                walk_len=walk_len,
                min_walk_len=min_walk_len,
                n_walk=n_walk,
                n_window=n_window,
                edge_sampling=edge_sampling,
                neighbor_sampling=neighbor_sampling
            )
    
            ### vocabulary 생성
            all_nodes=[[str(node)] for node in range(0,n_node+1)]
            vocab_corpus=walks+all_nodes
            skip_gram.build_vocab(vocab_corpus)
    
            ### Skip-Gram 학습
            skip_gram.train(
                corpus_iterable=walks,
                total_examples=skip_gram.corpus_count,
                epochs=epoch
            )
            print(f"Skip-Gram training is finished!")

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
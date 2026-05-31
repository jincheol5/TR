import argparse
import torch
from modules import TemporalGraph

"""
<< Test >> 
data.TemporalGraphData
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TemporalGraphData.update_graph()
            """
            eventstream=[
                (3,7,1),
                (3,6,2),
                (1,4,3),
                (1,3,4),

                (1,2,5),
                (2,7,5),
                (1,4,7),
                (4,7,7),

                (4,5,8),
                (1,3,9)
            ]
            graph=TemporalGraph(node_dim=2)
            batch_size=4
            i=0
            for idx in range(0,len(eventstream),batch_size):
                batch_events=eventstream[idx:idx+batch_size]
                graph.update_graph(batch_events=batch_events)
                print(f"<< {i+1} batch result >>")
                for node_id in data.node_ft.keys():
                    print(f"node_id: {node_id}")
                    print(f"node_feature: {data.node_ft[node_id]}")
                    print(f"node_neighbor: {data.neighbor[node_id]}")
                    print(f"node_neighbor_t: {data.neighbor_t[node_id]}",end="\n\n")
                i+=1

        case 2:
            """
            Test. TemporalGraph.find_temporal_neighbor()
            """
            eventstream=[
                (3,7,1),
                (3,6,2),
                (1,4,3),
                (1,3,4),
                (1,2,5),
                (2,7,5),
                (1,4,7),
                (4,7,7),
                (4,5,8),
                (1,3,9)
            ]
            graph=TemporalGraph(node_dim=2)
            graph.update_graph(batch_events=eventstream)
            tar_list=[2,3,7]
            cut_t_list=[3.0,5.0,8.0]
            for i in range(3):
                neighbor,neighbor_t=graph.find_temporal_neighbor(tar=tar_list[i],cut_time=cut_t_list[i])
                print(f"tar: {tar_list[i]}")
                print(f"neighbor: {neighbor}")
                print(f"neighbor_t:{neighbor_t}",end="\n\n")

        case 3:
            """
            Test. TemporalGraph.get_batch_data_for_embedding()
            """
            eventstream=[
                (3,7,1),
                (3,6,2),
                (1,4,3),
                (1,3,4),
                (1,2,5),
                (2,7,5),
                (1,4,7),
                (4,7,7),
                (4,5,8),
                (1,3,9)
            ]
            graph=TemporalGraph(node_dim=2)
            graph.update_graph(batch_events=eventstream)
            batch_tar=torch.tensor([2,3,7],dtype=torch.long)
            batch_t=torch.tensor([3.0,5.0,8.0],dtype=torch.float32)

            batch_data=graph.get_data_for_embedding(batch_tar=batch_tar,batch_t=batch_t)
            batch_tar_ts=batch_data["batch_tar_ts"] # [B,]
            batch_n=batch_data["batch_n"] # [B,N]
            batch_n_t=batch_data["batch_n_t"] # [B,N] 
            batch_n_ts=batch_data["batch_n_ts"] # [B,N]
            batch_n_mask=batch_data["batch_n_mask"] # [B,N]

            print(f"batch_tar_ts: {batch_tar_ts}")
            print(f"batch_n: {batch_n}")
            print(f"batch_n_t: {batch_n_t}")
            print(f"batch_n_ts: {batch_n_ts}")
            print(f"batch_n_mask: {batch_n_mask}")

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
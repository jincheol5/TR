import argparse
import torch
from modules import Memory
from modules import TimeEncoder,GRUMemoryUpdater

"""
<< Test >> 
memory_modules.GRUMemoryUpdater
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. GRUMemoryUpdater.update_memory
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
            memory=Memory(memory_dim=4)
            time_encoder=TimeEncoder(time_dim=4)
            module=GRUMemoryUpdater(
                mem_dim=4,
                msg_dim=4,
                time_dim=4,
                time_encoder=time_encoder,
                memory=memory,
                msg_fn="mlp",
                aggr_fn="last"
            )

            batch_size=4
            for idx in range(0,len(eventstream),batch_size):
                batch_events=eventstream[idx:idx+batch_size]
                memory.update_memory(batch_events=batch_events)
                src,tar,event_t=zip(*batch_events)
                src=torch.tensor(src,dtype=torch.long) # [B,]
                tar=torch.tensor(tar,dtype=torch.long) # [B,]
                event_t=torch.tensor(event_t,dtype=torch.float32) # [B,]
                
                nodes=torch.concat(
                    [src,tar],
                    dim=0
                )
                pre_nodes_memory=module.memory.get_batch_memory(batch_node=nodes)
                print(f"pre memory: {pre_nodes_memory}")
                
                module.update_memory(
                    src=src,
                    tar=tar,
                    event_t=event_t
                )
                
                updated_nodes_memory=module.memory.get_batch_memory(batch_node=nodes)
                print(f"updated memory: {updated_nodes_memory}")



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
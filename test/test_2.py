import argparse
import torch
from modules import Memory

"""
<< Test >> 
data.MemoryData
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. MemoryData.update_memory_data()
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
            data=Memory(memory_dim=2)
            batch_size=4
            i=0
            for idx in range(0,len(eventstream),batch_size):
                batch_events=eventstream[idx:idx+batch_size]
                data.update_memory_data(batch_events=batch_events)
                print(f"<< {i+1} batch result >>")
                for node in data.memory.keys():
                    print(f"node_id: {node}")
                    print(f"node_memory_feature: {data.memory[node]}")
                    print(f"node_interact_t: {data.interact_t[node]}",end="\n\n")
                i+=1

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
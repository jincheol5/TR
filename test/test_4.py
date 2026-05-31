import argparse
import torch
from modules import TemporalGraph,Memory
from modules import TimeEncoder,GRUMemoryUpdater
from modules import IdentityEmbedding,TimeProjectionEmbedding,GraphAttentionEmbedding,GraphSumEmbedding

"""
<< Test >> 
memory_modules.EmbeddingModule
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. GraphAttentionEmbedding
            """
            # parameter
            node_dim=4
            mem_dim=4
            msg_dim=12 # msg_fn="concat"인 경우 mem_dim+mem_dim+time_dim
            latent_dim=4
            output_dim=4
            time_dim=4
            n_head=4
            n_layer=3
            msg_fn="concat"
            aggr_fn="last"
            is_memory=True
            batch_size=4
            device=kwargs["device"]

            # data
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
            graph=TemporalGraph(node_dim=node_dim,device=device)
            memory=Memory(mem_dim=mem_dim,device=device)
            time_encoder=TimeEncoder(time_dim=time_dim)
            memory_updater=GRUMemoryUpdater(
                mem_dim=mem_dim,
                msg_dim=msg_dim,
                time_dim=time_dim,
                time_encoder=time_encoder,
                memory=memory,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )

            embedding_module=GraphAttentionEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                time_dim=time_dim,
                graph=graph,
                memory=memory,
                time_encoder=time_encoder,
                n_head=n_head,
                n_layer=n_layer,
                is_memory=is_memory
            )
            
            # load to device
            time_encoder=time_encoder.to(device)
            memory_updater=memory_updater.to(device)
            embedding_module=embedding_module.to(device)

            for idx in range(0,len(eventstream),batch_size):
                batch_events=eventstream[idx:idx+batch_size]

                # update data
                graph.update_graph(batch_events=batch_events)
                memory.update_memory(batch_events=batch_events)

                ### execute memory, embed module
                src,tar,event_t=zip(*batch_events)
                src=torch.tensor(src,dtype=torch.long,device=device) # [B,]
                tar=torch.tensor(tar,dtype=torch.long,device=device) # [B,]
                event_t=torch.tensor(event_t,dtype=torch.float32,device=device) # [B,]

                # 1. execute memory module
                memory_updater.update_memory(
                    src=src,
                    tar=tar,
                    event_t=event_t
                )

                # 2. execute embed module
                updated_batch_tar_ft=embedding_module.compute_embedding(
                    batch_tar=tar,
                    batch_t=event_t,
                    n_layer=3
                )

                print(f"Test Module: GraphAttentionEmbedding")
                print(f"updated batch_tar_feature:")
                print(updated_batch_tar_ft)

        case 2:
            """
            Test. IdentityEmbedding
            """
            # parameter
            node_dim=4
            mem_dim=4
            msg_dim=12 # msg_fn="concat"인 경우 mem_dim+mem_dim+time_dim
            latent_dim=4
            output_dim=4
            time_dim=4
            msg_fn="concat"
            aggr_fn="last"
            is_memory=True
            batch_size=4
            device=kwargs["device"]

            # data
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
            memory=Memory(mem_dim=mem_dim,device=device)
            time_encoder=TimeEncoder(time_dim=time_dim)
            memory_updater=GRUMemoryUpdater(
                mem_dim=mem_dim,
                msg_dim=msg_dim,
                time_dim=time_dim,
                time_encoder=time_encoder,
                memory=memory,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )

            embedding_module=IdentityEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                memory=memory
            )

            # load to device
            time_encoder=time_encoder.to(device)
            memory_updater=memory_updater.to(device)
            embedding_module=embedding_module.to(device)
            
            for idx in range(0,len(eventstream),batch_size):
                batch_events=eventstream[idx:idx+batch_size]

                # update data
                memory.update_memory(batch_events=batch_events)

                ### execute memory, embed module
                src,tar,event_t=zip(*batch_events)
                src=torch.tensor(src,dtype=torch.long,device=device) # [B,]
                tar=torch.tensor(tar,dtype=torch.long,device=device) # [B,]
                event_t=torch.tensor(event_t,dtype=torch.float32,device=device) # [B,]

                # 1. execute memory module
                memory_updater.update_memory(
                    src=src,
                    tar=tar,
                    event_t=event_t
                )

                # 2. execute embed module
                updated_batch_tar_ft=embedding_module.compute_embedding(
                    batch_tar=tar
                )

                print(f"Test Module: IdentityEmbedding")
                print(f"updated batch_tar_feature:")
                print(updated_batch_tar_ft)

        case 3:
            """
            Test. TimeProjectionEmbedding
            """
            # parameter
            node_dim=4
            mem_dim=4
            msg_dim=12 # msg_fn="concat"인 경우 mem_dim+mem_dim+time_dim
            latent_dim=4
            output_dim=4
            time_dim=4
            msg_fn="concat"
            aggr_fn="last"
            is_memory=True
            batch_size=4
            device=kwargs["device"]

            # data
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
            memory=Memory(mem_dim=mem_dim,device=device)
            time_encoder=TimeEncoder(time_dim=time_dim)
            memory_updater=GRUMemoryUpdater(
                mem_dim=mem_dim,
                msg_dim=msg_dim,
                time_dim=time_dim,
                time_encoder=time_encoder,
                memory=memory,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )

            embedding_module=TimeProjectionEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                memory=memory
            )

            # load to device
            time_encoder=time_encoder.to(device)
            memory_updater=memory_updater.to(device)
            embedding_module=embedding_module.to(device)
            
            for idx in range(0,len(eventstream),batch_size):
                batch_events=eventstream[idx:idx+batch_size]

                # update data
                memory.update_memory(batch_events=batch_events)

                ### execute memory, embed module
                src,tar,event_t=zip(*batch_events)
                src=torch.tensor(src,dtype=torch.long,device=device) # [B,]
                tar=torch.tensor(tar,dtype=torch.long,device=device) # [B,]
                event_t=torch.tensor(event_t,dtype=torch.float32,device=device) # [B,]

                # 1. execute memory module
                memory_updater.update_memory(
                    src=src,
                    tar=tar,
                    event_t=event_t
                )

                # 2. execute embed module
                updated_batch_tar_ft=embedding_module.compute_embedding(
                    batch_tar=tar,
                    batch_t=event_t
                )

                print(f"Test Module: TimeProjectionEmbedding")
                print(f"updated batch_tar_feature:")
                print(updated_batch_tar_ft)

        case 4:
            """
            Test. GraphSumEmbedding
            """
            # parameter
            node_dim=4
            mem_dim=4
            msg_dim=12 # msg_fn="concat"인 경우 mem_dim+mem_dim+time_dim
            latent_dim=4
            output_dim=4
            time_dim=4
            n_layer=3
            msg_fn="concat"
            aggr_fn="last"
            is_memory=True
            batch_size=4
            device=kwargs["device"]

            # data
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
            graph=TemporalGraph(node_dim=node_dim,device=device)
            memory=Memory(mem_dim=mem_dim,device=device)
            time_encoder=TimeEncoder(time_dim=time_dim)
            memory_updater=GRUMemoryUpdater(
                mem_dim=mem_dim,
                msg_dim=msg_dim,
                time_dim=time_dim,
                time_encoder=time_encoder,
                memory=memory,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )

            embedding_module=GraphSumEmbedding(
                node_dim=node_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                output_dim=output_dim,
                time_dim=time_dim,
                graph=graph,
                memory=memory,
                time_encoder=time_encoder,
                n_layer=n_layer,
                is_memory=is_memory
            )

            # load to device
            time_encoder=time_encoder.to(device)
            memory_updater=memory_updater.to(device)
            embedding_module=embedding_module.to(device)
            
            for idx in range(0,len(eventstream),batch_size):
                batch_events=eventstream[idx:idx+batch_size]

                # update data
                graph.update_graph(batch_events=batch_events)
                memory.update_memory(batch_events=batch_events)

                ### execute memory, embed module
                src,tar,event_t=zip(*batch_events)
                src=torch.tensor(src,dtype=torch.long,device=device) # [B,]
                tar=torch.tensor(tar,dtype=torch.long,device=device) # [B,]
                event_t=torch.tensor(event_t,dtype=torch.float32,device=device) # [B,]

                # 1. execute memory module
                memory_updater.update_memory(
                    src=src,
                    tar=tar,
                    event_t=event_t
                )

                # 2. execute embed module
                updated_batch_tar_ft=embedding_module.compute_embedding(
                    batch_tar=tar,
                    batch_t=event_t,
                    n_layer=3
                )

                print(f"Test Module: GraphSumEmbedding")
                print(f"updated batch_tar_feature:")
                print(updated_batch_tar_ft)

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    parser.add_argument("--device",type=str,default="cpu")
    args=parser.parse_args()
    test_config={
        "test_num":args.test_num,
        "device":args.device
    }
    test_fn(**test_config)
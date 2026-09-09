import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import TGN_Graph,DyGFormer_Graph
from model import TGAT,TGN,DyGFormer,ReaCH_TGN
from model_train import GNNModelTrainer,ReaCH_TGN_Trainer

"""
<< Test >> 
model_train.ModelTrainer

Test Model:
    - TGAT
    - TGN
    - DyGFormer
    - ReaCH-TGN
"""
def test_fn(**kwargs):
    ### set dataset and graph
    data=DataUtils.preprocess_graph_dataset(dataset_name=f"enron")
    graph_df=data["graph_df"]
    node_ft=data["node_ft"]
    edge_ft=data["edge_ft"]
    node_dim=data["node_dim"]
    edge_dim=data["edge_dim"]
    graph=TGN_Graph(
        graph_df=graph_df,
        node_ft=node_ft,
        edge_ft=edge_ft,
        node_dim=node_dim,
        edge_dim=edge_dim
    )
    seed=1
    graph.set_random_seed(seed=seed)

    ### TR sample 관련 파라미터
    batch_size=200
    max_hop=5
    n_pair=10
    sampling=kwargs["sampling"]

    ### 모델 관련 파라미터
    n_layer=1
    n_neighbor=10
    n_head=4

    ### 학습 관련 파라미터
    time_dim=32
    latent_dim=32
    embed_dim=32
    epoch=100
    lr=0.0005
    optimizer=f"adam"
    early_stop=True
    patience=10

    ### set data_loader
    train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
    train_dataset=TemporalGraphDataset(df=train_df)
    val_dataset=TemporalGraphDataset(df=val_df)
    test_dataset=TemporalGraphDataset(df=test_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    ### load SR, TR result
    train_SR_result=DataUtils.load_SR_result(
        dataset_name=f"enron",
        max_hop=max_hop,
        batch_size=batch_size,
        purpose="train"
    )
    train_TR_result=DataUtils.load_TR_result(
        dataset_name=f"enron",
        max_hop=max_hop,
        batch_size=batch_size,
        purpose="train"
    )
    val_TR_result=DataUtils.load_TR_result(
        dataset_name=f"enron",
        max_hop=max_hop,
        batch_size=batch_size,
        purpose="val"
    )
    test_TR_result=DataUtils.load_TR_result(
        dataset_name=f"enron",
        max_hop=max_hop,
        batch_size=batch_size,
        purpose="test"
    )

    ### set val, test sample_list using random sampling
    val_sample_list=TrainUtils.get_TR_sample_list(
        n_pair=n_pair,
        data_loader=val_loader,
        TR_result=val_TR_result,
        sampling=f"random"
    )
    test_sample_list=TrainUtils.get_TR_sample_list(
        n_pair=n_pair,
        data_loader=test_loader,
        TR_result=test_TR_result,
        sampling=f"random"
    )

    match kwargs["model_name"]:
        case "TGAT":
            """
            Test Model: TGAT
            """
            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_pair":n_pair,
                "sampling":sampling,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "time_dim":time_dim,
                "latent_dim":latent_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
            model=TGAT(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
            model=GNNModelTrainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                SR_result=train_SR_result,
                TR_result=train_TR_result,
                **model_config
            )
            evaluate_result=GNNModelTrainer.evaluate(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC of {kwargs['model_name']} using {kwargs['sampling']} TR Sampling: {evaluate_result['acc']}")

        case "TGN":
            ### TGN 학습 관련 파라미터
            msg_dim=32
            mem_dim=32

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_pair":n_pair,
                "sampling":sampling,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "time_dim":time_dim,
                "latent_dim":latent_dim,
                "msg_dim":msg_dim,
                "mem_dim":mem_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
            model=TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )   
            model=GNNModelTrainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                SR_result=train_SR_result,
                TR_result=train_TR_result,
                **model_config
            )
            evaluate_result=GNNModelTrainer.evaluate(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC of {kwargs['model_name']} using {kwargs['sampling']} TR Sampling: {evaluate_result['acc']}")

        case "DyGFormer":
            ### set DyGFormer graph
            graph=DyGFormer_Graph(
                graph_df=graph_df,
                node_ft=node_ft,
                edge_ft=edge_ft,
                node_dim=node_dim,
                edge_dim=edge_dim
            )
            seed=1
            graph.set_random_seed(seed=seed)

            ### DyGFormer 모델 관련 파라미터
            max_seq_len=10
            patch_size=5

            ### DyGFormer 학습 관련 파라미터
            co_dim=32
            common_dim=32

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_pair":n_pair,
                "sampling":sampling,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "max_seq_len":max_seq_len,
                "patch_size":patch_size,
                "latent_dim":latent_dim,
                "time_dim":time_dim,
                "co_dim":co_dim,
                "common_dim":common_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
            model=DyGFormer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                co_dim=co_dim,
                common_dim=common_dim,
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                patch_size=patch_size,
                graph=graph,
                n_neighbor=n_neighbor,
                n_layer=n_layer,
                n_head=n_head
            )
            model=GNNModelTrainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                SR_result=train_SR_result,
                TR_result=train_TR_result,
                **model_config
            )
            evaluate_result=GNNModelTrainer.evaluate(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC of {kwargs['model_name']} using {kwargs['sampling']} TR Sampling: {evaluate_result['acc']}")

        case "ReaCH-TGN":
            ### ReaCH-TGN 학습 관련 파라미터
            msg_dim=32
            mem_dim=32

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_pair":n_pair,
                "sampling":sampling,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "time_dim":time_dim,
                "latent_dim":latent_dim,
                "msg_dim":msg_dim,
                "mem_dim":mem_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
            model=ReaCH_TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
            model=ReaCH_TGN_Trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                SR_result=train_SR_result,
                TR_result=train_TR_result,
                **model_config
            )
            evaluate_result=ReaCH_TGN_Trainer.evaluate(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC of {kwargs['model_name']} using {kwargs['sampling']} TR Sampling: {evaluate_result['acc']}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=["TGAT","TGN","DyGFormer","ReaCH-TGN"],
        default=f"TGAT"
    )
    parser.add_argument("--sampling",
        type=str,
        choices=["random","hard"],
        default=f"random"
    )
    args=parser.parse_args()
    test_config={
        "model_name":args.model_name,
        "sampling":args.sampling
    }
    test_fn(**test_config)
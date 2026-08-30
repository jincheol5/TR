import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import TemporalGraph

"""
Compute TR result and save to .pt
"""
def main(**kwargs):
    data=DataUtils.preprocess_graph_dataset(
        dataset_name=kwargs["dataset_name"]
    )
    graph_df=data["graph_df"]
    train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)

    train_dataset=TemporalGraphDataset(df=train_df)
    val_dataset=TemporalGraphDataset(df=val_df)
    test_dataset=TemporalGraphDataset(df=test_df)

    train_loader=DataLoader(dataset=train_dataset,batch_size=kwargs["batch_size"],shuffle=False)
    val_loader=DataLoader(dataset=val_dataset,batch_size=kwargs["batch_size"],shuffle=False)
    test_loader=DataLoader(dataset=test_dataset,batch_size=kwargs["batch_size"],shuffle=False)

    graph=TemporalGraph(graph_df=graph_df)

    train_TR_result=TrainUtils.get_TR_result(
        graph=graph,
        data_loader=train_loader,
        max_hop=kwargs["max_hop"]
    )
    train_TR_label=train_TR_result["TR_label"]
    train_TR_hop=train_TR_result["TR_hop"]
    train_TR_last_t=train_TR_result["TR_last_t"]
    train_TR_first_t=train_TR_result["TR_first_t"]

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "CollegeMsg",
            "bitcoin-otc",
            "bitcoin-alpha",
            "enron"
        ],
        default=f"CollegeMsg"
    )
    parser.add_argument("--max_hop",type=int,default=5)
    parser.add_argument("--batch_size",type=int,default=200)
    parser.add_argument("--perpose",
        type=str,
        choices=[
            "train",
            "val",
            "test"
        ],
        default=f"train"
    )
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name,
        "max_hop":args.max_hop,
        "batch_size":args.batch_size,
        "perpose":args.perpose
    }
    main(**app_config)

import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset

def test_fn(**kwargs):
    data=DataUtils.preprocess_graph_dataset(
        dataset_name=kwargs["dataset_name"]
    )
    graph_df=data["graph_df"]
    train_df,_,_=TrainUtils.split_graph_df(df=graph_df)
    train_dataset=TemporalGraphDataset(df=train_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=200,shuffle=False)
    TR_result=DataUtils.load_TR_result(
        dataset_name=kwargs["dataset_name"],
        max_hop=5,
        batch_size=200,
        purpose="train"
    )
    TR_label=TR_result["TR_label"]
    TR_hop=TR_result["TR_hop"]
    TR_last_T=TR_result["TR_last_t"]

    match kwargs['test_num']:
        case "random":
            """
            Test. Random TR Sampling
            """
            batches=list(train_loader)
            first_batch_event=batches[0]
            last_batch_event=batches[-1]


if __name__=="__main__":
    """
    Execute test_fn
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
    parser.add_argument("--sampling",
            type=str,
            choices=[
                "random",
                "hard"
            ],
            default=f"random"
        )
    args=parser.parse_args()
    test_config={
        "dataset_name":args.dataset_name,
        "sampling":args.sampling
    }
    test_fn(**test_config)
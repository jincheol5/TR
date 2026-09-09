import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset,TR_Sampling

def test_fn(**kwargs):
    data=DataUtils.preprocess_graph_dataset(
        dataset_name=kwargs["dataset_name"]
    )
    graph_df=data["graph_df"]
    train_df,_,_=TrainUtils.split_graph_df(df=graph_df)
    train_dataset=TemporalGraphDataset(df=train_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=200,shuffle=False)
    SR_result=DataUtils.load_SR_result(
        dataset_name=kwargs["dataset_name"],
        max_hop=5,
        batch_size=200,
        purpose="train"
    )
    SR_label=SR_result["SR_label"]
    SR_hop=SR_result["SR_hop"]

    TR_result=DataUtils.load_TR_result(
        dataset_name=kwargs["dataset_name"],
        max_hop=5,
        batch_size=200,
        purpose="train"
    )
    TR_label=TR_result["TR_label"]
    TR_hop=TR_result["TR_hop"]
    TR_last_t=TR_result["TR_last_t"]

    match kwargs["sampling"]:
        case "random":
            """
            Test. Random TR Sampling
            """
            batches=list(train_loader)
            first_batch_event=batches[0]
            last_batch_event=batches[-1]

            sources=[1,2,3]
            first_result=TR_Sampling.random_TR_sampling(
                sources=sources,
                n_pair=10,
                query_time=first_batch_event[2].max().item(),
                TR_label=TR_label[0]
            )
            print(f"first sample size: {first_result['src'].size(0)}")
            print(f"first positive sample size: {first_result['pos_mask'].sum().item()}")
            print(f"first negative sample size: {first_result['src'].size(0)-first_result['pos_mask'].sum().item()}")

            last_result=TR_Sampling.random_TR_sampling(
                sources=sources,
                n_pair=10,
                query_time=last_batch_event[2].max().item(),
                TR_label=TR_label[-1]
            )
            print(f"last sample size: {last_result['src'].size(0)}")
            print(f"last positive sample size: {last_result['pos_mask'].sum().item()}")
            print(f"last negative sample size: {last_result['src'].size(0)-last_result['pos_mask'].sum().item()}")

        case "hard":
            batches=list(train_loader)
            first_batch_event=batches[0]
            last_batch_event=batches[-1]

            sources=[1,2,3]
            first_result=TR_Sampling.hard_TR_sampling(
                sources=sources,
                n_pair=10,
                start_query_time=first_batch_event[2].min().item(),
                end_query_time=first_batch_event[2].max().item(),
                SR_label=SR_label[0],
                SR_hop=SR_hop[0],
                TR_label=TR_label[0],
                TR_hop=TR_hop[0],
                TR_last_t=TR_last_t[0]
            )
            print(f"first sample size: {first_result['src'].size(0)}")
            print(f"first positive sample size: {first_result['pos_mask'].sum().item()}")
            print(f"first negative sample size: {first_result['src'].size(0)-first_result['pos_mask'].sum().item()}")

            last_result=TR_Sampling.hard_TR_sampling(
                sources=sources,
                n_pair=10,
                start_query_time=last_batch_event[2].min().item(),
                end_query_time=last_batch_event[2].max().item(),
                SR_label=SR_label[-1],
                SR_hop=SR_hop[-1],
                TR_label=TR_label[-1],
                TR_hop=TR_hop[-1],
                TR_last_t=TR_last_t[-1]
            )
            print(f"last sample size: {last_result['src'].size(0)}")
            print(f"last positive sample size: {last_result['pos_mask'].sum().item()}")
            print(f"last negative sample size: {last_result['src'].size(0)-last_result['pos_mask'].sum().item()}")

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
        default=f"enron"
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
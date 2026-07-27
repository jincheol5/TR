import argparse
from utils import DataUtils

"""
<< Test >> 
CTDNE
"""
def main(**kwargs):
    DataUtils.preprocess_snap_dataset(dataset_name=kwargs["dataset_name"])

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str,default=f"CollegeMsg")
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name
    }
    main(**app_config)
import argparse
from utils import GraphGenerator

"""
<< Test >> 
utils.graph_utils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. GraphGenerator.generate_graph
            """
            generator=GraphGenerator()
            try:
                ladder_graph=generator.generate_graph(graph_type="ladder",num_nodes=20,num_times=5)
            except:
                print(f"Generate error: ladder_graph")
            try:
                grid_graph=generator.generate_graph(graph_type="grid",num_nodes=20,num_times=5)
            except:
                print(f"Generate error: grid_graph")
            try:
                tree_graph=generator.generate_graph(graph_type="tree",num_nodes=20,num_times=5)
            except:
                print(f"Generate error: tree_graph")
            try:
                erdos_renyi_graph=generator.generate_graph(graph_type="erdos_renyi",num_nodes=20,num_times=5)
            except:
                print(f"Generate error: erdos_renyi_graph")
            try:
                barabasi_albert_graph=generator.generate_graph(graph_type="barabasi_albert",num_nodes=20,num_times=5)
            except:
                print(f"Generate error: barabasi_albert_graph")
            try:
                community_graph=generator.generate_graph(graph_type="community",num_nodes=20,num_times=5)
            except:
                print(f"Generate error: community_graph")
            try:
                caveman_graph=generator.generate_graph(graph_type="caveman",num_nodes=20,num_times=5)
            except:
                print(f"Generate error: caveman_graph")

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
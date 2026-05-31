import random
import numpy as np
import networkx as nx
from typing_extensions import Literal
from tqdm import tqdm

class GraphGenerator:
    @staticmethod
    def set_edge_time_attr(graph:nx.DiGraph,num_times:int):
        for src,tar in graph.edges():
            timestamp_num=np.random.randint(1,num_times+1)
            timestamps=np.random.uniform(0.2,1.0,size=timestamp_num)
            timestamps=np.sort(timestamps).tolist() 
            graph[src][tar]['t']=timestamps

    @staticmethod
    def remove_self_loop(graph:nx.Graph):
        graph.remove_edges_from(nx.selfloop_edges(graph))

    @staticmethod
    def remove_random_edges(graph:nx.DiGraph,ratio:float=0.3):
        edges=[(u,v) for u,v in graph.edges() if u!=v]
        num_edges=len(edges)
        num_remove=int(num_edges*ratio)
        remove_edges=random.sample(edges,num_remove)
        graph.remove_edges_from(remove_edges)
    
    @staticmethod
    def generate_graph(
            graph_type:Literal['ladder','grid','tree','erdos_renyi','barabasi_albert','community','caveman'],
            num_nodes:Literal[20,50,100,500,1000],
            num_times:int=5
        ):
        """
        <<Generate 7-type graphs>>
        1. ladder graph
        2. 2D grid graph
        3. tree graph
        4. Erdos-Renyi graph
        5. Barabasi-Albert graph
        6. 4-community graph
        7. 4-caveman graph

        Input: 
            graph_type
            num_nodes
            num_times
        Output:
            graph
        """
        match graph_type:
            case 'ladder':
                if num_nodes%2!=0:
                    raise ValueError("ladder graph requires an even number of nodes.")
                graph=nx.ladder_graph(num_nodes//2)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'grid':
                side_length=int(np.ceil(np.sqrt(num_nodes)))
                graph=nx.grid_2d_graph(side_length,side_length)
                graph=nx.convert_node_labels_to_integers(graph)
                graph=graph.subgraph(range(num_nodes)).copy()
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'tree':
                graph=nx.random_tree(num_nodes)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'erdos_renyi':
                p=min(np.log2(num_nodes)/num_nodes,0.5)
                graph=nx.erdos_renyi_graph(num_nodes,p)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                if num_nodes>=500:
                    GraphGenerator.remove_random_edges(graph=graph,ratio=0.7)
                else:
                    GraphGenerator.remove_random_edges(graph=graph,ratio=0.5)
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'barabasi_albert':
                if num_nodes<=4:
                    raise ValueError("barabasi_albert graph requires more than 4 number of nodes.")
                m=random.choice([4,5])
                graph=nx.barabasi_albert_graph(num_nodes,m)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                GraphGenerator.remove_random_edges(graph=graph,ratio=0.7)
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'community':
                if num_nodes<4:
                    raise ValueError("4-Community graph requires at least 4 nodes.")
                community_size=num_nodes//4
                remaining_nodes=num_nodes%4
                communities=[nx.erdos_renyi_graph(community_size,0.1) for _ in range(4)]
                graph=nx.disjoint_union_all(communities)
                for i in range(remaining_nodes):
                    graph.add_node(graph.number_of_nodes())
                nodes=list(graph.nodes())
                for i in range(len(nodes)):
                    for j in range(i+1,len(nodes)):
                        if (i//community_size)!=(j//community_size):
                            if random.random()<0.01:
                                graph.add_edge(i,j)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                if num_nodes>=500:
                    GraphGenerator.remove_random_edges(graph=graph,ratio=0.7)
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
            case 'caveman':
                if num_nodes<4:
                    raise ValueError("4-Caveman graph requires at least 4 nodes.")
                clique_size=num_nodes//4
                remaining_nodes=num_nodes%4
                graph=nx.caveman_graph(4,clique_size)
                for i in range(remaining_nodes):
                    graph.add_node(graph.number_of_nodes())
                edges_to_remove=[edge for edge in graph.edges() if random.random()<0.8]
                graph.remove_edges_from(edges_to_remove)
                num_shortcuts=int(0.025*num_nodes)
                for _ in range(num_shortcuts):
                    u,v=random.sample(list(graph.nodes()),2)
                    if not graph.has_edge(u,v):
                        graph.add_edge(u,v)
                GraphGenerator.remove_self_loop(graph=graph)
                graph=graph.to_directed()
                if num_nodes>=500:
                    GraphGenerator.remove_random_edges(graph=graph,ratio=0.7)
                GraphGenerator.set_edge_time_attr(graph=graph,num_times=num_times)
        return graph

class GraphUtils:
    @staticmethod
    def get_eventstream(graph:nx.DiGraph):
        """
        Input:
            networkx DiGraph
        Output:
            sorted tuple list: (src,tar,ts)
        """
        eventstream=[]
        for u,v,data in graph.edges(data=True):
            time_list=data['t']
            for timestamp in time_list:
                eventstream.append((int(u),int(v),float(timestamp)))
        eventstream=sorted(eventstream,key=lambda x:x[2])
        return eventstream




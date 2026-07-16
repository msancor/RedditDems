import graph_tool.all as gt
import numpy as np
import pandas as pd
from scipy import integrate
import time

DATA_PATH = "data_shared/supplementary_data/"

def directed_disparity_filter(G:gt.Graph, edge_property:gt.PropertyMap, alpha:float=0.05):
    """
    This function filters the edges of a graph based on the Disparity Filter algorithm.
    Args:
        G (Graph): The weighted and directed graph to be filtered.
        edge_property (PropertyMap): The edge property map containing the weights.
        alpha (float): The significance level of the test.
    Returns:
        Graph: The filtered graph.
    """
    new_edge_list = []
    initial_time = time.time()
    for u in G.iter_vertices():
        #Here we print every 100 vertices and we print the time passed till now
        if u % 100 == 0:
            print(f"Processing vertex {u} - Time passed: {time.time()-initial_time  :.2f} seconds")

        k_out = G.get_out_degrees([u])[0]
        k_in = G.get_in_degrees([u])[0]

        if k_out > 1:
            sum_out_weight = sum(np.absolute(edge_property[(u,v)]) for v in G.iter_out_neighbors(u))
            for v in G.iter_out_neighbors(u):
                w = edge_property[(u,v)]
                p_ij_out = float(np.absolute(w))/sum_out_weight
                alpha_ij_out = 1 - (k_out-1) * integrate.quad(lambda x: (1-x)**(k_out-2), 0, p_ij_out)[0]
                if alpha_ij_out < alpha:
                    new_edge_list.append((u, v, w))

        elif k_out == 1 and [G.get_in_degrees([v])[0] for v in G.iter_out_neighbors(u)][0] == 1:
            v = G.get_out_neighbors(u)[0]
            w = edge_property[(u,v)]
            new_edge_list.append((u, v, w))

        if k_in > 1:
            sum_in_weight = sum(np.absolute(edge_property[(v,u)]) for v in G.iter_in_neighbors(u))
            for v in G.iter_in_neighbors(u):
                w = edge_property[(v,u)]
                p_ij_in = float(np.absolute(w))/sum_in_weight
                alpha_ij_in = 1 - (k_in-1) * integrate.quad(lambda x: (1-x)**(k_in-2), 0, p_ij_in)[0]
                if alpha_ij_in < alpha:
                    new_edge_list.append((v, u, w))

    filtered_graph = gt.Graph(directed=True)
    eweight = filtered_graph.new_ep("double")  
    filtered_graph.add_edge_list(list(set(new_edge_list)), eprops=[eweight])
    return filtered_graph, eweight

if __name__ == "__main__":
    #Here we read the graph
    print("Reading graph data...")
    graph = pd.read_csv(f"{DATA_PATH}/attention_flow_graph_sampled.tsv", names=["subreddit_i", "subreddit_j", "weight"])
    #Here we convert to a list of tuples
    edges = [tuple(x) for x in graph.values]
    
    print("Creating the graph...")
    #Here we create the graph
    complete_graph = gt.Graph(directed=True)
    eweight = complete_graph.new_ep("double")
    vertex_property_map = complete_graph.add_edge_list(edges, hashed=True, eprops=[eweight])
    
    print("Filtering the graph...")
    filtered_graph, new_weights = directed_disparity_filter(complete_graph, eweight, alpha=0.01)
    print("Saving the filtered graph...")
   
    with open(f"{DATA_PATH}/filtered_attention_flow_graph_0.01_sampled.tsv", "w") as f:
        for s,t,w in filtered_graph.iter_edges([new_weights]):
            source = vertex_property_map[s]
            target = vertex_property_map[t]
            f.write(f"{source}\t{target}\t{w}\n")
    
    print("Done!")



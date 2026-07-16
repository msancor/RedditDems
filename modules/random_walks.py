import graph_tool.all as gt
import pandas as pd
import heapq as hq
import json
from typing import Tuple, List
import time
import warnings
import sys
warnings.filterwarnings("ignore")

MAIN_DATA_PATH = "data_shared/main_data/"
DATA_PATH = "data_shared/supplementary_data/"
REDDIT_DATA_PATH = "data_shared/reddit/"

def load_data(file_path:str)->pd.DataFrame:
    """
    This function loads the graph including the weights and demographic scores of the edges.
    The graph is represented as a pandas dataframe.

    Parameters:
    file_path (str): The path to the file containing the graph.

    Returns:
    pandas.DataFrame: The graph represented as a pandas dataframe.
    """
    #We import the demographic scores of the edges
    demographic_scores = pd.read_csv(file_path, sep=",", names=["subreddit_i", "subreddit_j", "age_score", "gender_score", "affluence_score", "partisan_score"])
    
    #We import the communities detected by the infomap algorithm
    communities = pd.read_csv(f"{REDDIT_DATA_PATH}/communities_infomap_0.01.csv", sep=",", header=0)
    #Here we filter the demographic graph to only include the subreddits that we have in the communities_infomap.csv file
    demographic_scores = demographic_scores[demographic_scores["subreddit_i"].isin(communities["id"])]
    demographic_scores = demographic_scores[demographic_scores["subreddit_j"].isin(communities["id"])]
    
    #We import the induced graph by the infomap communities
    infomap_edges = pd.read_csv(f"{REDDIT_DATA_PATH}/edges_infomap.csv", sep=",", header=0)
    #Now we join the demographic graph with the induced graph by the infomap communities
    demographic_scores = demographic_scores.merge(infomap_edges, left_on=["subreddit_i", "subreddit_j"], right_on=["source", "target"], how="inner")
    #We drop the columns that are not needed
    demographic_scores = demographic_scores.drop(columns=["source", "target"])
    
    return demographic_scores

def load_communities()->dict:
    """
    Load the communities from the file.

    Returns:
    dict: A dictionary with the communities.
    """
    #Here we define a list of community seed subreddits we want to analyze
    seeds = ["conspiracy", "spirituality", "Conservative", "environment", "socialism"]
    #Here we load the communities from the file using pandas
    communities = pd.read_csv(f"{DATA_PATH}/seed_communities.csv", sep=",", header=0)
    #We get the communities for the seeds
    communities_id = dict(communities[communities["id"].isin(seeds)][["community", "id"]].values)
    #Now we rename the communities to the seed names
    communities["community"] = communities["community"].map(communities_id)
    #Here we create a dictionary with keys the communities and values the list of subreddits in the community
    communities_dict = communities.groupby("community")["id"].apply(list).to_dict()
    #We return the communities
    return communities_dict

def extract_network(demographic_scores:pd.DataFrame, demographic_axis:str, sign:str=None)->Tuple[gt.Graph, gt.EdgePropertyMap, gt.VertexPropertyMap]:
    """
    This function extracts a network from the demographic scores of the edges.
    The network is represented as a graph-tool graph.

    Parameters:
    demographic_scores (pd.DataFrame): The demographic scores of the edges.
    demographic_axis (str): The demographic axis to be used for the network.
    sign (str): The sign of the demographic axis to be used for the network.

    Returns:
    Tuple[gt.Graph, gt.EdgePropertyMap, gt.VertexPropertyMap]: The graph-tool graph, the edge property map and the vertex property map.
    """
    if demographic_axis == "none":
        extracted_df = demographic_scores[['subreddit_i', 'subreddit_j', 'weight']]
    else:
        #Here we mantain only the age_score and weight columns along with the subreddit_i and subreddit_j columns
        extracted_df = demographic_scores[['subreddit_i', 'subreddit_j', f'{demographic_axis}_score', 'weight']]
        #We create a new column by multiplying the age_score by the weight
        extracted_df[f'{demographic_axis}_weighted'] = extracted_df[f'{demographic_axis}_score'] * extracted_df['weight']
        #We only keep the columns that we need
        extracted_df = extracted_df[['subreddit_i', 'subreddit_j', f'{demographic_axis}_weighted']]
        
        #We filter the dataframe based on the sign
        if sign == "positive":
            extracted_df = extracted_df[extracted_df[f'{demographic_axis}_weighted'] > 0]
        elif sign == "negative":
            extracted_df = extracted_df[extracted_df[f'{demographic_axis}_weighted'] < 0]
            #We take the absolute value of the negative scores
            extracted_df[f'{demographic_axis}_weighted'] = extracted_df[f'{demographic_axis}_weighted'].abs()
        else:
            raise ValueError("The sign parameter must be either 'positive' or 'negative'")
    
    #Here we create a list of unique subreddits from the demographic scores
    unique_subreddits = list(set(demographic_scores['subreddit_i'].values) | set(demographic_scores['subreddit_j'].values))
    #We do the same for the extracted dataframe
    unique_subreddits_extracted = list(set(extracted_df['subreddit_i'].values) | set(extracted_df['subreddit_j'].values))
    #We substract tht extracted subreddits from the unique subreddits
    unique_subreddits = list(set(unique_subreddits) - set(unique_subreddits_extracted))
    
    #Here we convert to a list of tuples
    edges = [tuple(x) for x in extracted_df.values]
    #Here we create the graph
    complete_graph = gt.Graph(directed=True)
    eweight = complete_graph.new_ep("double")
    vertex_property_map = complete_graph.add_edge_list(edges, hashed=True, eprops=[eweight])

    #We add the vertices that are not connected by an edge
    for subreddit in unique_subreddits:
        added_vertex = complete_graph.add_vertex()
        vertex_property_map[added_vertex] = subreddit

    return complete_graph, eweight, vertex_property_map

def transpose_graph_with_props(complete_graph:gt.Graph, eweight:gt.EdgePropertyMap, names:gt.VertexPropertyMap)->Tuple[gt.Graph, gt.EdgePropertyMap, gt.VertexPropertyMap]:
    """
    This function transposes a directed graph and copies the edge weights and vertex properties.
    Parameters:
    complete_graph (gt.Graph): The directed graph to be transposed.
    eweight (gt.EdgePropertyMap): The edge property map containing the weights.
    names (gt.VertexPropertyMap): The vertex property map containing the names.

    Returns:
    Tuple[gt.Graph, gt.EdgePropertyMap, gt.VertexPropertyMap]: The transposed graph, the edge property map with weights, and the vertex property map with names.
    """
    if not complete_graph.is_directed():
        raise ValueError("Graph must be directed to transpose.")

    # Create transpose graph with same number of vertices
    g_t = gt.Graph(directed=True)
    g_t.add_vertex(complete_graph.num_vertices())

    # Create new edge property for weights on transpose graph
    eweight_t = g_t.new_edge_property(eweight.value_type())

    # Copy the vertex property map to the transpose graph
    vprop_t = g_t.new_vertex_property(names.value_type())
    for v in complete_graph.vertices():
        vprop_t[v] = names[v]

    # Add reversed edges and copy edge weights
    for e in complete_graph.edges():
        src = int(e.source())
        tgt = int(e.target())
        e_t = g_t.add_edge(tgt, src)  # reversed edge
        eweight_t[e_t] = eweight[e]

    return g_t, eweight_t, vprop_t

def ppr(graph:gt.Graph, weights:gt.EdgePropertyMap, names:gt.VertexPropertyMap, communities:dict, community_name:str, restart_prob:float=0.0)->gt.VertexPropertyMap:
    """
    This function calculates the Personalized PageRank for a given community.

    Parameters:
    graph (gt.Graph): The graph.
    weights (gt.EdgePropertyMap): The edge property map containing the weights.
    names (gt.VertexPropertyMap): The vertex property map containing the names.
    communities (dict): The communities dictionary.
    community_name (str): The name of the community.
    restart_prob (float): The restart probability.

    Returns:
    gt.VertexPropertyMap: The personalized pagerank.
    """
    #Here we make a personalization vector for the pagerank algorithm
    p = graph.new_vertex_property("double")

    for u in graph.iter_vertices():
        if names[u] in communities[community_name]:
            p[u] = restart_prob
        else:
            p[u] = 1.0 - restart_prob

    #now we normalize the personalization vector
    p.a /= p.a.sum()

    #Here we calculate the personalized pagerank
    with gt.openmp_context(nthreads=20, schedule="guided"):
        pr = gt.pagerank(graph, weight=weights, pers=p)
    return pr

def get_gateways(graph:gt.Graph, weights:gt.EdgePropertyMap, names:gt.VertexPropertyMap, communities:dict, community_name:str, k:int=None)->List[Tuple[str, float]]:
    """
    This function calculates the gateways for a given community.

    Parameters:
    graph (gt.Graph): The graph.
    weights (gt.EdgePropertyMap): The edge property map containing the weights.
    names (gt.VertexPropertyMap): The vertex property map containing the names.
    communities (dict): The communities dictionary.
    community_name (str): The name of the community.
    k (int): The number of gateways to be returned.

    Returns:
    List[Tuple[str, float]]: A list of tuples containing the gateway subreddit and the score.
    """
    #Here we calculate the personalized pagerank
    pr = ppr(graph, weights, names, communities, community_name, restart_prob=0.0)
    #Here we get the top k gateways for the community
    top_k = []
    for u in graph.iter_vertices():
        if names[u] in communities[community_name]:
            negative_pr_value = (-1)*pr[u]
            hq.heappush(top_k, (negative_pr_value, names[u]))

    sorted_top_k = [hq.heappop(top_k) for _ in range(len(top_k))][:k]
    return [(name, -score) for score, name in sorted_top_k]

def get_bridges(graph:gt.Graph, weights:gt.EdgePropertyMap, names:gt.VertexPropertyMap, communities:dict, community_name:str, k:int=None)->List[Tuple[str, float]]:
    """
    This function calculates the bridges from a given community.

    Parameters:
    graph (gt.Graph): The graph.
    weights (gt.EdgePropertyMap): The edge property map containing the weights.
    names (gt.VertexPropertyMap): The vertex property map containing the names.
    communities (dict): The communities dictionary.
    community_name (str): The name of the community.
    k (int): The number of bridges to be returned.

    Returns:
    List[Tuple[str, float]]: A list of tuples containing the bridge subreddit and the score.
    """
    #We transpose the graph to get the bridges
    graph_t, weights_t, names_t = transpose_graph_with_props(graph, weights, names)
    #Here we calculate the personalized pagerank
    pr = ppr(graph_t, weights_t, names_t, communities, community_name, restart_prob=1.0)
    #Here we get the top k bridges for the community
    top_k = []
    for u in graph_t.iter_vertices():
        if names_t[u] not in communities[community_name]:
            negative_pr_value = (-1)*pr[u]
            hq.heappush(top_k, (negative_pr_value, names_t[u]))

    sorted_top_k = get_k_exact_subreddits(top_k, communities, k)
    return sorted_top_k

def get_k_exact_subreddits(top_k:List[Tuple[str, float]], communities:dict, k:int)->List[Tuple[str, float]]:
    """
    This function returns the top k subreddits that are in the communities.

    Parameters:
    top_k (List[Tuple[str, float]]): The top k subreddits.
    communities (dict): The communities dictionary.
    k (int): The number of subreddits to be returned.

    Returns:
    List[Tuple[str, float]]: The top k subreddits that are in the communities.
    """
    counter = 0
    sorted_top_k = []
    all_subreddits = [subreddit for subreddit_list in communities.values() for subreddit in subreddit_list]
    for _ in range(len(top_k)):
        if counter == k:
            break
        negative_pr_value, subreddit = hq.heappop(top_k)
        if subreddit in all_subreddits:
            sorted_top_k.append((subreddit, -negative_pr_value))
            counter += 1
    return sorted_top_k
def community_reachability(graph:gt.Graph, weights:gt.EdgePropertyMap, names:gt.VertexPropertyMap, communities:dict, community_name:str)->List[Tuple[str, float]]:
    """
    This function calculates the community reachability for a given source community to the other communities.

    Parameters:
    graph (gt.Graph): The graph.
    weights (gt.EdgePropertyMap): The edge property map containing the weights.
    names (gt.VertexPropertyMap): The vertex property map containing the names.
    communities (dict): The communities dictionary.
    community_name (str): The name of the community.

    Returns:
    List[Tuple[str, float]]: A list of tuples containing the community and the reachability score.
    """
    #We transpose the graph
    graph_t, weights_t, names_t = transpose_graph_with_props(graph, weights, names)
    #Here we calculate the personalized pagerank
    pr = ppr(graph_t, weights_t, names_t, communities, community_name, restart_prob=1.0)
    #Here we define a list of community seed subreddits we want to analyze
    seeds = ["conspiracy", "spirituality", "Conservative", "environment", "socialism"]
    #We take out the source community
    seeds.remove(community_name)
    #Here we calculate the reachability for each community
    reachability = {(community_name, target): sum(pr[u] for u in graph.iter_vertices() if names[u] in communities[target]) for target in seeds}
    #We renormalize the reachability scores
    normalized_reachability = {key: value/sum(reachability.values()) for key, value in reachability.items()}
    #We convert the dictionary to a list of tuples
    normalized_reachability = [(key[1], value) for key, value in normalized_reachability.items()]
    return normalized_reachability

if __name__ == "__main__":
    arguments = sys.argv[1]
    if arguments == "full":
        FILE_PATH = f"{DATA_PATH}/attention_flow_graph_scores.tsv"
    elif arguments == "sparse":
        FILE_PATH = f"{DATA_PATH}/attention_flow_graph_scores_sparse.tsv"
    seeds = ["conspiracy", "spirituality", "Conservative", "environment", "socialism"]
    axis = ["none", "age", "gender", "affluence", "partisan"]
    sign = ["positive", "negative"]
    communities = load_communities()
    demographic_scores = load_data(FILE_PATH)

    results_dict = {}
    for community_name in seeds:
        results_dict[community_name] = {}
        for demographic_axis in axis:
            print(f"Processing {community_name} community with demographic axis {demographic_axis}")
            start_time = time.time()
            results_dict[community_name][demographic_axis] = {}
            if demographic_axis == "none":
                graph, weights, names = extract_network(demographic_scores, demographic_axis)
                gateways = get_gateways(graph, weights, names, communities, community_name)
                bridges = get_bridges(graph, weights, names, communities, community_name)
                reachability = community_reachability(graph, weights, names, communities, community_name)
                results_dict[community_name][demographic_axis]["none"] = {"gateways": gateways, "bridges": bridges, "reachability": reachability}
            else:
                for s in sign:
                    graph, weights, names = extract_network(demographic_scores, demographic_axis, s)
                    gateways = get_gateways(graph, weights, names, communities, community_name)
                    bridges = get_bridges(graph, weights, names, communities, community_name)
                    reachability = community_reachability(graph, weights, names, communities, community_name)
                    results_dict[community_name][demographic_axis][s] = {"gateways": gateways, "bridges": bridges, "reachability": reachability}

            print(f"Elapsed time: {time.time() - start_time}")
    #We save the results to a file
    with open(f"{MAIN_DATA_PATH}/gateways_bridges_results_{arguments}_inverted_final.json", "w") as f:
        json.dump(results_dict, f)

                

from collections import Counter
import igraph as ig
import pandas as pd
import numpy as np
import json

DATA_PATH = "data_shared/supplementary_data/"
REDDIT_DATA_PATH = "data_shared/reddit/"

print("Reading graph data...")
#Here we read the graph from a pandas dataframe
df = pd.read_csv(f"{DATA_PATH}/filtered_attention_flow_graph_0.01_sampled.tsv", sep='\t', names=['source', 'target', 'weight'])

#Here we create the graph using the igraph library
g = ig.Graph.TupleList(df.values, directed=True, edge_attrs=['weight'])

print("Creating the graph...")
#Here we obtain the largest connected component using weakly connected components
components = g.components("weak")
largest_component = components.giant()

#Here we print the number of nodes and edges in the largest connected component
print('Original graph: nodes = {}, edges = {}'.format(g.vcount(), g.ecount()))
print('Giant component: nodes = {}, edges = {}'.format(largest_component.vcount(), largest_component.ecount()))

print("Running the Infomap algorithm...")
#Then we perform community detection
communities = largest_component.community_infomap(edge_weights='weight', trials=1000)

#Here we create a pandas dataframe with the community assignment for each node name i.e., id, community
df_communities = pd.DataFrame({'id': largest_component.vs['name'], 'community': communities.membership})
#Here we save the dataframe to a file with the headers
df_communities.to_csv(f"{REDDIT_DATA_PATH}/communities_infomap_0.01.csv", sep=',', index=False, header=True)

print(f"Results saved to {REDDIT_DATA_PATH}/communities_infomap_0.01.csv")

#Here we print the number of communities, avg size per community, modularity and codelength
#Here we count the number of nodes per community
community_sizes = Counter(communities.membership)
#We print the results
print('Number of communities: {}'.format(len(community_sizes)))
print('Average size per community: {}'.format(np.mean(list(community_sizes.values()))))
print('Modularity: {}'.format(communities.modularity))
print('Codelength: {}'.format(communities.codelength))

#We save the distribution of community sizes to a json file
with open(f"{DATA_PATH}/community_sizes.json", 'w') as f:
    json.dump(community_sizes, f)

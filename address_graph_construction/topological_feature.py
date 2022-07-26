def central_expansion(node_embedding,edge_unindex):
    G = nx.Graph()
    G.add_edges_from(edge_unindex)
    closeness = nx.algorithms.closeness_centrality(G)
    betweenness = nx.algorithms.betweenness_centrality(G)
    pagerank = nx.pagerank_numpy(G)

    node_id = list(node_embedding.keys())
    for k in node_id:
        node_embedding[k].extend([closeness[k], betweenness[k], pagerank[k],G.degree(k)])
    return node_embedding 
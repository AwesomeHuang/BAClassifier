import pickle
import datetime
from utils import save,txs_dividing,graph_logging
from address_preparing import address_preparing
from transaction import transaction
from single import single
from multi import multi
from config import argparser
from preprocess import preprocess
from collections import defaultdict
from tqdm import tqdm

def building(txs,theta,tx_path):

    preprocess(txs,tx_path)          
    tx_embedding = transaction(txs) 
    single_embedding,single_edge = single(txs)
    multi_embedding,multi_edge = multi(txs,theta)
    edges = single_edge + multi_edge
    embedding = {}
    embedding.update(tx_embedding)
    embedding.update(single_embedding)
    embedding.update(multi_embedding)
    return embedding,edges

def main():
    args = argparser()
    marked_address = address_preparing(args.label_path,args.size,args.relation_path,args.tx_path,args.dataset_path)
    node_num = 0
    graph_num = 0
    address_graph = defaultdict(list)
    address_num = 0
    for add,label in tqdm(marked_address.items()):
        if address_num==1024:
            break
        address_num+=1
        tx_groups = txs_dividing(add,args.relation_path,args.size) 
        for txs in tx_groups:
            start=datetime.datetime.now()
            embedding,edges = building(txs,args.theta,args.tx_path)
            graph_num += 1 
            address_graph[add].append(graph_num)
            save(embedding,edges,label,graph_num,node_num,args.dataset_path)
            node_num += len(embedding)
            end=datetime.datetime.now()
            graph_logging(graph_num,end,start)
            
    f = open('address_graph.pkl', 'wb')
    pickle.dump(address_graph, f)

    
if __name__ == "__main__":
    main()
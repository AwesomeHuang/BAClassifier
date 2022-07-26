import json
import pandas as pd
import math

def read(path):
    f = open(path,'r')
    data = json.load(f)
    f.close()
    return data

def save(embedding,edges,label,indicator,node_num,dataset_path):
    nodes = list(embedding.keys())
    with open(dataset_path + 'BTC_node_attributes.txt','a') as f:
        for node in nodes: 
            n = str(embedding[node])
            n = n.lstrip('[').rstrip(']')
            f.write(n + '\n')  
        f.close()
    with open(dataset_path + 'BTC_A.txt','a') as f:
        for edge in edges: 
            e = str([nodes.index(edge[0])+node_num+1,
                nodes.index(edge[1])+node_num+1])
            e = e.lstrip('[').rstrip(']')
            f.write(e + '\n')
        f.close()
    with open(dataset_path + 'BTC_node_labels.txt','a') as f:
        for node in nodes: 
            if node.startswith('single'):
                f.write(str(1) + '\n')          
            elif node.startswith('multi'):
                f.write(str(2) + '\n')
            else:
                f.write(str(3) + '\n')
        f.close()
    with open(dataset_path + 'BTC_graph_labels.txt','a') as f:
        f.write(str(label) + '\n')
        f.close()
    with open(dataset_path + 'BTC_graph_indicator.txt','a') as f:
        for node in nodes: 
            f.write(str(indicator) + '\n')
        f.close()
    return node_num 

def occurrences_counts(txs,p):
    occurrences = {}
    for tx in txs:
        f = open('tx_embedding/'+tx+'.json')
        tx_data = json.load(f)
        for address in list(tx_data[p].keys()):
            if address in occurrences:
                occurrences[address]+=1
            else:
                occurrences[address]=1
    return occurrences

def txs_dividing(address,relation_path,size):

    file1 = relation_path + address + '_in_addressRelation.csv'
    file2 = relation_path + address + '_out_addressRelation.csv'
    
    data_in = pd.read_csv(file1,header=None,engine='python')
    data_in.columns=['addr','val','tx','block']
    data_in.drop(columns = ['addr','val'],inplace = True)
    data_out = pd.read_csv(file2,header=None,engine='python')
    data_out.columns=['addr','val','tx','block']
    data_out.drop(columns = ['addr','val'],inplace = True)
    data = pd.concat([data_in,data_out])
    data.sort_values("block",inplace=True) # sort the txs according to block 
    data.drop_duplicates(subset=None, keep='first', inplace=True)# delete recurring transactions
    txs = list(data['tx']) 
    return dividing(txs,size)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def dividing(txs,size):

    '''
         ignore the remain txs
    '''

    return list(chunks(txs, size))[:int(math.floor(len(txs)/size))]

def graph_logging(graph_indicator,end,start):

    loggraph = {}
    loggraph['graph_num'] = graph_indicator 
    graphtime = 'Running time: %s '%(end-start)
    loggraph['Time'] = graphtime
    fw = open("construct_log/graph_log.txt",'a')
    fw.write(str(loggraph)+'\n')
    fw.close()


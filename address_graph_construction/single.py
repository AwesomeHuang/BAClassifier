import json
from utils import read,occurrences_counts
from feature import statistics

def single(txs):

    #Count the number of occurrences of each address in this transaction group
    embedding = {}
    edge = []
    for tx in txs:
        filepath = 'tx_embedding/' + tx +'.json'
        tx_data = read(filepath)

        # calculate the embedding for aggregated single addresses and delete them
        for p in ['in','out']:
            if p =="in":
                occurrences = occurrences_counts(txs,p)
                values,aggregated_num = single_aggregator(tx_data[p],occurrences)
                if len(values)!=0:
                    embedding['single_'+tx+'_'+p] = statistics(values) + [aggregated_num] 
                    edge.append(['single_'+tx+'_'+p,tx])
            else:
                occurrences = occurrences_counts(txs,p)
                values,aggregated_num = single_aggregator(tx_data[p],occurrences)
                if len(values)!=0:
                    embedding['single_'+tx+'_'+p] = statistics(values) + [aggregated_num] 
                    edge.append([tx,'single_'+tx+'_'+p])        
    return embedding,edge

def single_aggregator(tx_data,occurrences):
    values = []
    aggregated_num = 0 # the num of aggregated addresses
    for address in list(tx_data.keys()):
        if occurrences[address]==1:
            aggregated_num+=1
            values.extend(tx_data.pop(address))
    return values,aggregated_num
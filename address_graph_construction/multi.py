import torch
import numpy as np
import torch.nn as nn
import sys
from utils import read
from collections import defaultdict,Counter
from feature import statistics

'''
input transactions
output address_tx: address1:tx1,tx2,tx3...
        values: address1:value1,value2,value3...
        
'''

def connection(txs,p):
    address_tx = defaultdict(list)
    values = defaultdict(list)
    for tx in txs:
        data = read('tx_embedding/'+tx+'.json')
        for address,value in data[p].items():
            # single deleted remain multi
            address_tx[address].append(tx)
            values[address].extend(value)

    for key,val in address_tx.items():
        address_tx[key] = tuple(val)
        
    return address_tx,values

def adjacency_matrix(txs,address_tx,addresses):

    index_ad = []
    index_tx = []
    degree = []

    for address in addresses:
        degree.append(len(address_tx[address])) #return the number of txs of each address
        index_ad.extend([addresses.index(address)]*len(address_tx[address]))
        index_tx.extend([txs.index(tx) for tx in address_tx[address]])

    index = torch.LongTensor([index_ad,index_tx])
    index_transpose = torch.LongTensor([index_tx,index_ad])
    one = torch.ones(len(index_ad)).float()
    ad_tx = torch.sparse.FloatTensor(index, one, torch.Size([len(addresses),len(txs)])).to_dense()
    tx_ad = torch.sparse.FloatTensor(index_transpose, one, torch.Size([len(txs),len(addresses)])).to_dense()

    return ad_tx,tx_ad,np.array(degree)

def similarity_matrix(ad_tx,tx_ad,degree,theta):

    similarity = ad_tx.mm(tx_ad).float() # ad_tx*tx_ad = ad*ad 
    degree =torch.diag(torch.tensor(1./degree)).float()
    normalized = similarity.mm(degree)#normalizing the similarity of addresses
    filtered =  normalized - torch.ones([len(degree),len(degree)])*theta
    relu = nn.ReLU()
    similarity = relu(filtered)

    return similarity

def aggregate_parameter(num,theta):

    if num >1200:
        theta = 0.9
    else: 
        theta = 0
        
    beta = 0
    if num>800 and num<=1200:
        beta = num/40
    if num>1200 and num <=2500:
        beta = num/12
    if num>2500 and num<=4500:
        beta = num/3
    if num>4500 and num<=7500:
        beta = num/2.4
    if num>7500 and num<10000:
        beta = num/2
    if num>=10000:
        beta = num/1.8
    return theta,beta


def multi_aggregator_1(address_tx,values,pos):
    
    more_txs_adds,once_txs_adds = transverse(address_tx)
    address_tx,values = deduplicating(more_txs_adds,once_txs_adds,values,pos)
    return address_tx,values


def transverse(address_tx):

    adds_vals = defaultdict(list)
    txset = list(address_tx.values())
    once_txs = [item for item,count in Counter(txset).items() if count ==1 ]
    more_txs = set(txset)-set(once_txs)
    more_txs_adds = defaultdict(list)
    once_txs_adds = defaultdict(list)
    for add,tx in address_tx.items():
        if tx in more_txs:
            more_txs_adds[tx].append(add)
        else:
            once_txs_adds[tx].append(add)
    return more_txs_adds,once_txs_adds

def deduplicating(more_txs_adds,once_txs_adds,raw_values,pos):

    values = {}
    address_tx = {}
    i = 0
    for tx,adds in more_txs_adds.items():
        i += 1
        address_tx['Deduplication_'+pos+str(i)] = set(tx)
        values['Deduplication_'+pos+str(i)] = []
        for add in adds:
            values['Deduplication_'+pos+str(i)].extend(raw_values[add])

    for tx,add in once_txs_adds.items():
        address_tx[add[0]] = set(tx)
        values[add[0]] = raw_values[add[0]]

    return address_tx,values


def multi_aggregator_2(similarity,beta):
    num = len(similarity) # the number of addresses in the transaction group
    aggregated_index = []
    aggregated_num = {} # the num of aggregated address
    for i in range(num):
        nonzero = len(torch.nonzero(similarity[i]))# remain address
        if nonzero>beta:
            aggregated_num[i] = nonzero
            aggregated_index.append(i)

    return aggregated_index,aggregated_num

def multi(txs,theta):

    edge = [] # the connection of address with tx
    embedding = {}
    
    for pos in ['in','out']:
        if pos =="in":
            address_tx,values = connection(txs,pos)
            address_tx,values = multi_aggregator_1(address_tx,values,pos) 
            addresses = list(address_tx.keys())
            ad_tx,tx_ad,degree = adjacency_matrix(txs,address_tx,addresses)
            num = len(ad_tx)
            theta,beta = aggregate_parameter(num,theta)
            similarity = similarity_matrix(ad_tx,tx_ad,degree,theta)
            aggregated_index,aggregated_num = multi_aggregator_2(similarity,beta)
            for add in aggregated_index:
                _,index = similarity[add].sort(descending = True)
                address_id = index.tolist()[:len(torch.nonzero(similarity[add]))]# nonzero element of similarity
                aggregated_values = []
                for i in address_id:
                    aggregated_values.extend(values[addresses[i]])
                embedding['multi_'+ pos + str(add)] = statistics(aggregated_values)+[aggregated_num[add]]
                for tx in address_tx[addresses[add]]:
                    edge.append(['multi_'+ pos + str(add),tx]) 
        else:
            address_tx,values = connection(txs,pos)
            address_tx,values = multi_aggregator_1(address_tx,values,pos) 
            addresses = list(address_tx.keys())
            ad_tx,tx_ad,degree = adjacency_matrix(txs,address_tx,addresses)
            num = len(ad_tx)
            theta,beta = aggregate_parameter(num,theta)
            similarity = similarity_matrix(ad_tx,tx_ad,degree,theta)
            aggregated_index,aggregated_num = multi_aggregator_2(similarity,beta)
            for add in aggregated_index:
                _,index = similarity[add].sort(descending = True)
                address_id = index.tolist()[:len(torch.nonzero(similarity[add]))]# nonzero element of similarity
                aggregated_values = []
                for i in address_id:
                    aggregated_values.extend(values[addresses[i]])
                embedding['multi_'+ pos + str(add)] = statistics(aggregated_values)+[aggregated_num[add]]
                for tx in address_tx[addresses[add]]:
                    edge.append([tx,'multi_'+ pos + str(add)]) 
    return embedding,edge

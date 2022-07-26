from utils import read
from feature import statistics

def transaction(txs):    
    embedding = {}
    for tx in txs:
        filepath = 'tx_embedding/'+tx +'.json'
        tx_data = read(filepath)
        values = [value[0] for value in list(tx_data['in'].values()) + list(tx_data['out'].values())]
        aggregated_num = 0 # no address or tx aggregated
        embedding[tx] = statistics(values) +[aggregated_num]
    return embedding
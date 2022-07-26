import shutil
import os
import json
from collections import defaultdict
from utils import read

'''

    preproess the data, keep the useful info
    input: the raw transaction json file
    output: {
        in: address1:value1, address2:value2 ...   the inputs addresses with transfer amount
        out: address1:value1, address2:value2 ...  the outputs addresses with transfer amount
    }

'''

def preprocess(txs,tx_path):
    if os.path.exists('tx_embedding'):
        shutil.rmtree('tx_embedding')
    os.mkdir('tx_embedding')
    for tx in txs:
        data={} 
        data['in'] = defaultdict(list)
        data['out'] = defaultdict(list)

        # read file
        path = tx_path + tx +'_tx.json'
        tx_info = read(path)

        # process file
        for inputs in tx_info['inputs']:
            if inputs['prev_addresses']!=[]: #ignore the coinbase
                address = inputs['prev_addresses'][0]
                value = inputs['prev_value']/float(100000000) # transfer the satoshi to the standard btc
                data['in'][address].append(value)             
        for outputs in tx_info['outputs']:
            if outputs['addresses']!=[]: #ignore the coinbase
                address = outputs['addresses'][0]
                value = outputs['value']/float(100000000)  # transfer the satoshi to the standard btc
                data['out'][address].append(value) 
        filepath = 'tx_embedding/'+tx+'.json'

        # save file
        f = open(filepath,'w+')
        json.dump(data,f)
        f.close()
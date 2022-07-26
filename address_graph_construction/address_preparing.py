import os
import pickle
import csv
import pandas as pd
from tqdm import tqdm


def integrity_checking(address,size,relation_path,tx_path):
    file1 = relation_path + address + '_in_addressRelation.csv'
    file2 = relation_path + address + '_out_addressRelation.csv'

    # check the addressRelation file is vaild or not
    if (os.path.getsize(file1)==0) or (os.path.getsize(file2)==0):
        return False
    else: 
        data_in = pd.read_csv(file1,header=None)
        data_out = pd.read_csv(file2,header=None)
        data_in.columns=['add','val','tx','block']
        data_out.columns=['add','val','tx','block']
        data_all = pd.concat([data_in,data_out], axis=0)
        txs = list(set(data_all['tx']))       
        """ check num of tx file larger than the size of tx_group """
        if len(txs)<size:
            return False

        """ check tx file whether exist """
        for tx in txs:
            if(not os.path.exists(tx_path + tx +'_tx.json')):
                return False

        return True

def marking(label_path):
    data = pd.read_csv(label_path+'Address_Saved_Labeld_Marked_Modified.csv',header=None)
    data.columns = ['add','label','info']
    marked_address = {}
    for i in set(data['add']):
        label = list(data[data['add']==i]['label'])[0]   
        if label=='EXCHANGE':
            marked_address[i] = 0        
        if label=='GAMBLING_WEBSITE':
            marked_address[i] = 1               
        if label=='MINING_POOL':
            marked_address[i] = 2        
        if label in ['RANSOMWARE','MARKETPLACE','FAUCET',
        'MIXER','WALLET','OLDHISTORY','SERVICE']:
            marked_address[i] = 3  
    
    return marked_address         

def address_checking(marked_address,size,relation_path,tx_path):
    print('addresses checking ...')
    for address in tqdm(list(marked_address.keys())):
        if not integrity_checking(address,size,relation_path,tx_path):
            del marked_address[address]  #delete the invaild addresses
    print('checking finish !')
    print('The number of valid addresses : ',len(marked_address))
    return marked_address

def address_preparing(label_path,size,relation_path,tx_path,dateset_path):

    if(os.path.exists(dateset_path+'marked_address.pkl')):
        print('marked_address are already prepared')
        f = open(dateset_path+'marked_address.pkl', 'rb')
        marked_address = pickle.load(f)
        f.close()  
        return marked_address
    else:
        print('marked_address start to prepare')
        marked_address = marking(label_path)
        marked_address = address_checking(marked_address,size,relation_path,tx_path)
        f = open(dateset_path+'marked_address.pkl', 'wb')
        pickle.dump(marked_address, f)
        f.close()
        print('marked_address are already prepared')
        return marked_address    
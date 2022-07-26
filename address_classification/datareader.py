import os
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import pandas as pd
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def read_data(id_path,emb_path,label_path,graph_address):

    testid = open(id_path,'rb')
    testemb = open(emb_path,'rb')
    readdataid = CPU_Unpickler(testid).load()
    readdataemb = CPU_Unpickler(testemb).load()


    data = pd.read_csv(label_path,header=None)
    data.columns = ['add','type','info']
    address_label = {}
    for i in set(data['add']):
        labeldi = list(data[data['add']==i]['type'])[0]   
        if labeldi=='EXCHANGE':
            address_label[i] = 0        
        if labeldi=='GAMBLING_WEBSITE':
            address_label[i] = 1               
        if labeldi=='MINING_POOL':
            address_label[i] = 2        
        if labeldi in ['RANSOMWARE','MARKETPLACE','FAUCET','MIXER','WALLET','OLDHISTORY','SERVICE']:
            address_label[i] = 3 

    fr1 = open(graph_address,'r')
    dic1= eval(fr1.read()) 
    add_label = {}
    for add in list(dic1.keys()):
        add_label[add] = address_label[add]

    batchnum = len(readdataemb)
    allemb = []
    for i in range(batchnum):
        for read in readdataemb[i]:
            allemb.append(read.tolist())
    hidden_dim = len(allemb[1])
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    temp = {}
    addemb = []
    add_labels =[]

    for add in list(dic1.keys()):
        gids = dic1[add]
        temp = []
        for gid in gids:     
            temp.append(np.array(allemb[gid-1]).tolist())
        addemb.append(list(temp))
        add_labels.append(add_label[add])



    X_train,X_test,y_train,y_test = train_test_split(addemb,add_labels,test_size=0.2)

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for i in range(len(X_train)):
        X = torch.tensor(X_train[i]).to(DEVICE)
        Y = torch.tensor([y_train[i]]).to(DEVICE)
        train_data.append(X)
        train_label.append(Y)

    for i in range(len(X_test)):
        X = torch.tensor(X_test[i]).to(DEVICE)
        Y = torch.tensor([y_test[i]]).to(DEVICE)
        test_data.append(X)
        test_label.append(Y)

    return train_data,train_label,test_data,test_label,hidden_dim
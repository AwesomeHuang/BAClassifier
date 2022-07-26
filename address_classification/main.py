from datareader import read_data
from model.model import ModelB
from cal_recall_pre_f1 import cal_precision_and_recall
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.utils.data as Data
import argparse
import pickle


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4,help ='The number of address types')
    parser.add_argument('--epochs', type=int, default=20,help = 'the epochs of training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,help = 'the learning rate of training')
    parser.add_argument('--weight_decay', type=float, default=0.001,help = 'the weight decay of training')
    parser.add_argument('--id_path', type=str, default='data/graph_id.pkl',help = 'the id of graph')
    parser.add_argument('--emb_path', type=str, default='data/graph_embedding.pkl',help = 'the embedding of each graph')
    parser.add_argument('--label_path', type=str, default='data/Address_Saved_Labeld_Marked_Modified.csv',help = 'the label of address')
    parser.add_argument('--graph_address', type=str, default='data/graph_address.txt',help = 'address and its corresponding graph')
    args = parser.parse_args()
    return args


def ACC(Y_test,Y_pred,n):
    
    acc = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)
        
    return acc

def pre(Y_test,Y_pred,n):
    
    pre = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:,i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)
        
    return pre


def main():

    DEVICE = torch.device("cpu")
    args = argparser()
    num_classes = args.num_classes
    epochs = args.epochs   
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    id_path = args.id_path
    emb_path = args.emb_path
    label_path = args.label_path
    graph_address = args.graph_address
    
    train_data,train_label,test_data,test_label,hidden_dim = read_data(id_path,emb_path,label_path,graph_address)
    model = ModelB(hidden_dim,num_classes).to(DEVICE)

    
    print("Device:", DEVICE)
    print(model)


    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

    model.train()
    recall_max = 0
    for epoch in range(epochs):
        print("===train===")
        total_loss = 0
        correct = 0
        train_true = []
        train_pred = []
        for i in range(len(train_data)):
            optimizer.zero_grad()
            logits = model(train_data[i])
            loss = criterion(logits, train_label[i])
            train_pred.append(logits.max(1)[1].item())
            train_true.append(train_label[i].item())

            loss.backward()
            optimizer.step()
            train_acc = torch.eq(
                logits.max(1)[1], train_label[i]).float().mean()
            correct = train_acc + correct
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(
            epoch, loss.item(), correct/len(train_data)))
        train_accuracy = accuracy_score(train_true,train_pred)
        train_precision, train_recall, train_f1 = precision_recall_fscore_support(train_true,train_pred,average='macro')[:-1]
        print("train_accuracy: ", train_accuracy)
        print("train_precision: ",train_precision)
        print("train_recall: ",train_recall)
        print("ftrain_1: ",train_f1)
        target_names = ["EXCHANGE","GAMBLING_WEBSITE","MINING_POOL","other"]
        print("train_ACC:",ACC(train_true, train_pred,4))
        print("train_pre:",pre(train_true, train_pred,4))
        print(classification_report(train_true, train_pred, target_names=target_names,digits=4))





        model.eval()
        print("===test===")
        y_pred = []
        y_true = []
        with torch.no_grad():
            correct = 0
            for i in range(len(test_data)):
                logits = model(test_data[i])
                test_logits = logits
                y_pred.append(test_logits.max(1)[1].item())
                y_true.append(test_label[i].item())
                test_acc = torch.eq(
                    test_logits.max(1)[1], test_label[i]
                ).float().mean()
                correct = test_acc + correct
        print("TestAcc {:.9f}".format(correct/len(test_data)))
        accuracy = accuracy_score(y_true,y_pred)
        precision, recall, f1 = precision_recall_fscore_support(y_true,y_pred,average='macro')[:-1]
        print("accuracy: ", accuracy)
        print("precision: ",precision)
        print("recall: ",recall)
        print("f1: ",f1)
        target_names = ["EXCHANGE","GAMBLING_WEBSITE","MINING_POOL","other"]
        print("ACC:",ACC(y_true, y_pred,4))
        print("pre:",pre(y_true, y_pred,4))
        print(classification_report(y_true, y_pred, target_names=target_names,digits=4))
        if recall > recall_max:
            recall_max = recall_max
            recall_max_result = y_pred
            
    f = open('result.pkl','wb')
    pickle.dump(recall_max_result,f)
    f.close()
    print("\n")
        
if __name__ == "__main__":
    main()
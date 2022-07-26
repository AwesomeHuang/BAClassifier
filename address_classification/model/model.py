import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F



class Convolution(nn.Module):
    def __init__(self,hidden_dim):
        super(Convolution, self).__init__()
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self,input_feature):
        output = torch.mm(input_feature, self.weight)
        return output
    
    
class ModelB(nn.Module):
    def __init__(self,hidden_dim,num_classes=4):

        super(ModelB, self).__init__()

        self.cnn = Convolution(hidden_dim)
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(hidden_dim , hidden_dim// 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, num_classes)
        self.bilstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, bidirectional=False,batch_first=False)

 
        
        
        
    def forward(self, input_feature):

        output_lstm, (hn, cn)= self.bilstm(input_feature.unsqueeze(1))
        output_last = output_lstm[:,-1,:]

        fc1 = F.relu(self.fc1(output_last[-1,:]))
        fc2 = F.relu(self.fc2(fc1))
        logits = self.fc3(fc2)
        
        return logits.unsqueeze(0)
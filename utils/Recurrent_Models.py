from unicodedata import bidirectional
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm



class RNNClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim,out,device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNNCell(input_dim, hidden_dim)  # RNN Cell
        self.fc = nn.Linear(hidden_dim, out)            # fully connected layer: maps last hidden vector to model prediction
        self.activation = nn.Sigmoid()                # coz binary classification
        self.device=device
    
    def forward(self, x):

        hidden = self.init_hidden(x)
        
        ############################# 
        time_steps=x.shape[1]                 # shape of x is (batches,time_Steps,features)
        
        for i in range(0,time_steps):
            inputs=x[:,i]                     # (batch,features) shape
            hidden = self.rnn(inputs,hidden)
            
        out = self.fc(hidden)                 # take the hidden vector corresponding to last time step
        ###########################
        
        return self.activation(out)
    
    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim)
        return h0.to(self.device)



class LSTMClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim,out,device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out)
        self.activation = nn.Sigmoid()
        self.device=device
        self.drop=nn.Dropout(0.75)

    def forward(self, x):
        hidden,cell = self.init_hidden(x)
    
        time_steps=x.shape[1]              #shape of x is (batches,time_Steps,features)
        
        for i in range(0,time_steps):
            inputs=x[:,i]                  # (batch,features) shape
            hidden,cell = self.rnn(inputs,(hidden,cell))
        hidden=self.drop(hidden)
        out = self.fc(hidden)              
        #############################
        
        return self.activation(out)

    # function to get embedding or latent representation
    def emb(self, x):
        hidden,cell = self.init_hidden(x)
        time_steps=x.shape[1]              #shape of x is (batches,time_Steps,features)
        
        for i in range(0,time_steps):
            inputs=x[:,i]                  # (batch,features) shape
            hidden,cell = self.rnn(inputs,(hidden,cell))
        hidden=self.drop(hidden)
        return hidden

    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim)
        c0 = torch.zeros(x.size(0), self.hidden_dim)
        return h0.to(self.device),c0.to(self.device)



# Phenotyping or latent patient disorder predictyion model
class Emb_pheno(nn.Module):
    
    def __init__(self, emb_dim, out,device):
        super().__init__()
        self.emb_dim = emb_dim
        self.fc = nn.Linear(emb_dim, out)            
        self.activation = nn.Sigmoid()                
        self.device=device
        self.drop=nn.Dropout(0.15) 

    def forward(self, x):
        out = self.fc(x)                
        return self.activation(out)



class Ethnicity(nn.Module):
    
    def __init__(self, emb_dim,out,device):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, out)            # fully connected layer: maps last hidden vector to model prediction
        self.activation1 = nn.Sigmoid()    
        self.device=device

    def forward(self, x):
        x=self.activation1(self.fc1(x))          
        return x
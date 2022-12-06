import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score


def get_loaders(path='./data/',batch=64):
    T=np.load(path+'Data.npy')
    L=np.load(path+'Labels.npy')

    # A=torch.transpose(T, 2, 1) 
    # T=torch.transpose(A,3,2)

    X_train, X_test, y_train, y_test = train_test_split(T, L, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    train_dataset= TensorDataset(torch.tensor(X_train),torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=False)

    val_dataset= TensorDataset(torch.tensor(X_val),torch.tensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch,shuffle=False)

    test_dataset= TensorDataset(torch.tensor(X_test),torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch,shuffle=False)
    return train_loader, val_loader,test_loader


def get_loaders_quantum(path='./data/'):
    T=np.load(path+'encoded_data_8bits.npy')
    L=np.load(path+'Labels.npy')

    X_train, X_test, y_train, y_test = train_test_split(T, L,test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=42)

    print(X_train.shape)

    train_dataset= TensorDataset(torch.tensor(X_train),torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64,shuffle=False)

    val_dataset= TensorDataset(torch.tensor(X_val),torch.tensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=64,shuffle=False)

    test_dataset= TensorDataset(torch.tensor(X_test),torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64,shuffle=False)



    return train_loader, val_loader,test_loader



def get_device():
    if torch.cuda.is_available():
       device=torch.device('cuda:0')
    else:
       device=torch.device('cpu')   
    return device



def prediction_binary(model,loader,loss_fn,device):
    P=[]
    L=[]
    model.eval()
    val_loss=0
    for i,batch in enumerate(loader):
        data,labels=batch
        data=data.to(torch.float32).to(device)
        labels=labels.to(torch.float32).to(device)
        
        if len(labels.shape)>1:
            labels=labels[:,0]
        pred=model(data)[:,0]
        loss=loss_fn(pred,labels)
        val_loss=val_loss+loss.item()

        P.append(pred.cpu().detach().numpy())
        L.append(labels.cpu().detach().numpy())
        
    val_loss=val_loss/len(loader)
    P=np.concatenate(P)  
    L=np.concatenate(L)
    auc=roc_auc_score(L,P)
    return val_loss,auc
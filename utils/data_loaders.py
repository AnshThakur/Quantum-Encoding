
### Loaders for MIMIC-III


import torch
from torch.utils.data import TensorDataset, DataLoader
import _pickle as cPickle
import numpy as np
import pandas as pd



# Mortality Prediction data
def get_loaders_IHM(path='./Data',batch=64,sampler=True,shuffle=False):
    
    with open(path+'/train_raw.p', 'rb') as f:
        x= cPickle.load(f)

    T=x[0]
    L=x[1]
    print(T.shape)
    print(np.where(np.array(L)==1)[0].shape[0])
    train_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    
    if sampler==True:
        class_sample_count = np.array([len(np.where(L == t)[0]) for t in np.unique(L)])
        weight = 1. / class_sample_count
        # weight[1]=weight[1]*2
        samples_weight = np.array([weight[t] for t in L])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight)) 
        train_loader = DataLoader(train_dataset, batch_size=batch,sampler=sampler,shuffle=shuffle)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=shuffle)    
       
    with open(path+'/val_raw.p', 'rb') as f:
        x= cPickle.load(f)

    T=x[0]
    L=x[1]
    val_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    val_loader = DataLoader(val_dataset, batch_size=batch)
    print(T.shape)
    print(np.where(np.array(L)==1)[0].shape[0])
    with open(path+'/test_raw.p', 'rb') as f:
        x= cPickle.load(f)

    test = x["data"][0]
    test_labels = np.array(x["data"][1])
    print(test.shape) 
    print(np.where(test_labels==1)[0].shape[0])

    test_dataset= TensorDataset(torch.tensor(test),torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch)     
    
    return train_loader,val_loader,test_loader



     
def collate_batch(batch):
  
    label_list, data_list, = [], []
  
    for (_data,_label) in batch:
        label_list.append(_label)
        data = torch.tensor(_data, dtype=torch.float32)
        data_list.append(data)
  
    label_list = torch.tensor(label_list, dtype=torch.float32)
  
    data_list = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0.0)
  
    return data_list,label_list



## Penotyping or 25 disorder prediction 
def get_loaders_pheno(path='./Data',batch=64,shuffle=False):
    
    with open(path+'/train_raw.p', 'rb') as f:
        x= cPickle.load(f)


    T=x[0]
    L=pd.read_csv(path+'/new_pheno_train.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()


    train_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=shuffle)    
       
    with open(path+'/val_raw.p', 'rb') as f:
        x= cPickle.load(f)

    T=x[0]
    L=pd.read_csv(path+'/new_pheno_val.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()
    val_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    val_loader = DataLoader(val_dataset, batch_size=batch)

    with open(path+'/test_raw.p', 'rb') as f:
        x= cPickle.load(f)

    test = x["data"][0]
    L=pd.read_csv(path+'/new_pheno_test.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()
    test_labels = L

    test_dataset= TensorDataset(torch.tensor(test),torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch)     
    
    return train_loader,val_loader,test_loader


     

### Quantum Encoded data for mortality prediction
def get_loaders_IHM_quantum(path='./Data',batch=64,sampler=True,shuffle=False):
    
    with open(path+'/train_raw.p', 'rb') as f:
        x= cPickle.load(f)
     
    # T=np.load(path+'/temporal_train_v2.npy')
    T=np.load(path+'/BE_layers_2_train_v2.npy')
    # T=np.load(path+'/BE_layers_2_train_v2.npy')
    T=np.rollaxis(T,2,1)

    L=x[1]
    print(T.shape)
    train_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    
    if sampler==True:
        class_sample_count = np.array([len(np.where(L == t)[0]) for t in np.unique(L)])
        weight = 1. / class_sample_count
        # weight[1]=weight[1]*2
        samples_weight = np.array([weight[t] for t in L])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight)) 
        train_loader = DataLoader(train_dataset, batch_size=batch,sampler=sampler,shuffle=shuffle)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=shuffle)    
       
    with open(path+'/val_raw.p', 'rb') as f:
        x= cPickle.load(f)

    # T=np.load(path+'/temporal_val_v2.npy')
    # T=np.load(path+'/BE_layers_2_val_v2.npy')
    T=np.load(path+'/BE_layers_2_val_v2.npy')
    L=x[1]
    T=np.rollaxis(T,2,1) 

    val_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    val_loader = DataLoader(val_dataset, batch_size=batch)

    with open(path+'/test_raw.p', 'rb') as f:
        x= cPickle.load(f)

    # test=np.load(path+'/temporal_test_v2.npy')
    # test = np.load(path+'/BE_layers_2_test_v2.npy')
    test=np.load(path+'/BE_layers_2_test_v2.npy')
    test_labels = np.array(x["data"][1])
    test=np.rollaxis(test,2,1)

    test_dataset= TensorDataset(torch.tensor(test),torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch)     
    
    return train_loader,val_loader,test_loader
     



## quantum encoded data for phenotyping
def get_loaders_pheno_quantum(path='./Data',batch=64,shuffle=False):

    T=np.load(path+'/BE_layers_2_train_v2.npy')
    # T=np.load(path+'/temporal_train_v2.npy')
    T=np.rollaxis(T,2,1)    
    print(T.shape)
    L=pd.read_csv(path+'/new_pheno_train.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()

    train_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=shuffle)    
       

    T=np.load(path+'/BE_layers_2_val_v2.npy')
    # T=np.load(path+'/temporal_val_v2.npy')
    T=np.rollaxis(T,2,1)  

    L=pd.read_csv(path+'/new_pheno_val.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()
    val_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    val_loader = DataLoader(val_dataset, batch_size=batch)

    with open(path+'/test_raw.p', 'rb') as f:
        x= cPickle.load(f)

    test = np.load(path+'/BE_layers_2_test_v2.npy')
    
    # test=np.load(path+'/temporal_test_v2.npy')
    test=np.rollaxis(test,2,1)      
    
    L=pd.read_csv(path+'/new_pheno_test.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()
    test_labels = L

    test_dataset= TensorDataset(torch.tensor(test),torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch)     
    
    return train_loader,val_loader,test_loader    




# Random projection encoded data for IHM
def get_loaders_IHM_RP(path='./Data',batch=64,sampler=True,shuffle=False):
    
    with open(path+'/train_raw.p', 'rb') as f:
        x= cPickle.load(f)
     
    # T=np.load(path+'/BE_layers_2_train_v2.npy')
    T=np.load(path+'/RP_train.npy')
    # T=np.rollaxis(T,2,1)

    L=x[1]
    print(T.shape)
    train_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    
    if sampler==True:
        class_sample_count = np.array([len(np.where(L == t)[0]) for t in np.unique(L)])
        weight = 1. / class_sample_count
        # weight[1]=weight[1]*2
        samples_weight = np.array([weight[t] for t in L])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight)) 
        train_loader = DataLoader(train_dataset, batch_size=batch,sampler=sampler,shuffle=shuffle)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=shuffle)    
       
    with open(path+'/val_raw.p', 'rb') as f:
        x= cPickle.load(f)

    # T=np.load(path+'/BE_layers_2_val_v2.npy')
    T=np.load(path+'/RP_val.npy')
    L=x[1]
    # T=np.rollaxis(T,2,1) 

    val_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    val_loader = DataLoader(val_dataset, batch_size=batch)

    with open(path+'/test_raw.p', 'rb') as f:
        x= cPickle.load(f)

    # test = np.load(path+'/BE_layers_2_test_v2.npy')
    test=np.load(path+'/RP_test.npy')
    test_labels = np.array(x["data"][1])
    # test=np.rollaxis(test,2,1)

    test_dataset= TensorDataset(torch.tensor(test),torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch)     
    
    return train_loader,val_loader,test_loader



# Random projection encoded data for phenotyping
def get_loaders_pheno_RP(path='./Data',batch=64,shuffle=False):

    T=np.load(path+'/RP_train.npy')  
    print(T.shape)
    L=pd.read_csv(path+'/new_pheno_train.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()

    train_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    train_loader = DataLoader(train_dataset, batch_size=batch,shuffle=shuffle)    
       

    
    T=np.load(path+'/RP_val.npy')
    L=pd.read_csv(path+'/new_pheno_val.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()
    val_dataset= TensorDataset(torch.tensor(T),torch.tensor(L))
    val_loader = DataLoader(val_dataset, batch_size=batch)

    with open(path+'/test_raw.p', 'rb') as f:
        x= cPickle.load(f)

    test=np.load(path+'/RP_test.npy')
    
    # test=np.load(path+'/temporal_test_v2.npy')
    # test=np.rollaxis(test,2,1)      
    
    L=pd.read_csv(path+'/new_pheno_test.csv')
    L=L.iloc[:,3:]
    L=L.to_numpy()
    test_labels = L

    test_dataset= TensorDataset(torch.tensor(test),torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch)     
    
    return train_loader,val_loader,test_loader    
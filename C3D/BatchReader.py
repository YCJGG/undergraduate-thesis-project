import numpy as np
import torch.utils.data as data
import torch
class DatasetProcessing(data.Dataset):
    def __init__(self, data_path):
        #self.feature_path = data_path
        ffp = open(data_path,'r')
        fp = list(ffp)
        self.fullfeature_path = [x.strip().split(' ')[0] for x in fp]
        self.partialfeature_path = [x.strip().split(' ')[1] for x in fp]
        self.labels = [int(x.strip().split(' ')[2]) for x in fp]
        ffp.close()
        
    def __getitem__(self, index):
        ff = np.load(self.fullfeature_path[index])
        pf = np.load(self.partialfeature_path[index])
        label = torch.LongTensor([self.labels[index]])
        return ff, pf, label, index
    def __len__(self):
        return len(self.fullfeature_path)
#DatasetProcessing('test.list')

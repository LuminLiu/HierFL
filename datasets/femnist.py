# Leveraging the dataset from LEAF benchmark
# Each client holds a dataset written by itself

from options import args_parser
import json
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from torchvision.transforms import transforms


class Dataset_femnist(Dataset):
    def __init__(self, data, labels, transforms = None):
        self.labels = labels
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        if self.transforms is not None:
            x = self.transforms(x)
            y = self.transforms(y)
        return x, y

def read_data_femnist(train_data_dir, test_data_dir):
    """
    Adapted from the code in LEAF project
    :param train_data_dir:
    :param test_data_dir:
    :return: trainloaders and testloaders for each of the client,
            trainloader and testloader for the whole dataset
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    clients = list(train_data.keys())
    return clients, train_data, test_data

def generate_femnist_dataloader(num_clients, clients, train_data, test_data, batch_size):
    """

    :param num_clients:
    :param clients:
    :param traindata:
    :param testdata:
    :return: two lists of traindataloaders and testdataloaders
    and trainloader and testloader of the whole combined dataset
    """
    train_dataloaders = [0] * num_clients
    test_dataloaders = [0] * num_clients
    train_datasets_all = []
    test_datasets_all = []
    for i in range(num_clients):
        cid = clients[i]
        cdata_train = train_data[cid]
        # Y_train = [np.asarray([y]) for y in cdata_train['y'][0:200] ]
        # X_train = [np.asarray(x).reshape((-1, 28,28)) for x in cdata_train['x'][0:200]]
        # tensor_x_train = torch.stack([torch.Tensor(i) for i in X_train])  # transform to torch tensors
        # tensor_y_train = torch.stack([torch.Tensor(i) for i in Y_train])
        # train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
        Y_train = torch.from_numpy(np.asarray(cdata_train['y'][0:200]))
        X_train = torch.from_numpy(np.asarray(cdata_train['x'][0:200]).reshape((-1,1, 28, 28)))
        train_dataset = Dataset_femnist(data=X_train, labels=Y_train)
        train_datasets_all.append(train_dataset)
        train_dataloaders[i] = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # test data loader
        cdata_test = test_data[cid]
        Y_test = torch.from_numpy(np.asarray(cdata_test['y'][0:70]))
        X_test = torch.from_numpy(np.asarray(cdata_test['x'][0:70]).reshape((-1,1, 28, 28)))
        test_dataset = Dataset_femnist(data=X_test, labels=Y_test)
        # Y_test = [np.asarray([y]) for y in cdata_test['y'][0:70]]
        # X_test = [np.asarray(x).reshape((-1,28, 28)) for x in cdata_test['x'][0:70]]
        # tensor_x_test = torch.stack([torch.Tensor(i) for i in X_test])  # transform to torch tensors
        # tensor_y_test = torch.stack([torch.Tensor(i) for i in Y_test])
        # test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
        test_datasets_all.append(test_dataset)
        test_dataloaders[i] = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print(len(train_datasets_all))
    print(len(test_datasets_all))
    v_train_dataset = ConcatDataset(train_datasets_all)
    v_test_dataset = ConcatDataset(test_datasets_all)
    v_train_dataloader = DataLoader(dataset= v_train_dataset, batch_size= batch_size, shuffle= True)
    v_test_dataloader = DataLoader(dataset=v_test_dataset, batch_size= batch_size, shuffle=False)

    return train_dataloaders, test_dataloaders, v_train_dataloader, v_test_dataloader

def main():
    TRAIN_DATA_DIR = f'./data/FEMNIST/keep300-sample-user/train'
    TEST_DATA_DIR = f'./data/FEMNIST/keep300-sample-user/test'
    clients, train_data, test_data = read_data_femnist(train_data_dir=TRAIN_DATA_DIR,
                                                       test_data_dir=TEST_DATA_DIR)
    print(clients)
    print(len(clients))
    print(len(train_data))
    print(len(test_data))
    trainloaders, testloaders, v_trainloader, v_testloader = generate_femnist_dataloader(10, clients, train_data,
                                                                                         test_data, 10)
    # for dataloader in trainloaders:
    #     print(f'The number of samples in traindataloader is {dataloader.dataset.data.shape[0]}')
    # for dataloader in testloaders:
    #     print(f'The number of samples in test dataloader is {dataloader.dataset.data.shape[0]}')
    # print(f'The number of samples in v_train dataloader is {v_trainloader.dataset.cummulative_sizes[-1]}')
    # print(f'The number of samples in v_test_dataloader is {v_testloader.dataset.cummulative_sizes[-1]}')
    pass


if __name__ == '__main__':
    main()
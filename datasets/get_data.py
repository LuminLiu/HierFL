# Interface between the dataset and client
# For artificially partitioned dataset, params include num_clients, dataset
# For federated datasets and multitask datasets, params include dataset

FEMNIST_TRAIN_DATA_DIR = f'./data/FEMNIST/keep300-sample-user/train'
FEMNIST_TEST_DATA_DIR = f'./data/FEMNIST/keep300-sample-user/test'

from datasets.femnist import read_data_femnist, generate_femnist_dataloader
from datasets.cifar_mnist import get_dataset, show_distribution
def get_dataloaders(args):
    """
    :param args:
    :return: A list of trainloaders, a list of testloaders, a concatenated trainloader and a concatenated testloader
    """
    if args.dataset in ['mnist', 'cifar10']:
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataset(dataset_root='data',
                                                                                       dataset=args.dataset,
                                                                                       args = args)
    elif args.dataset == 'femnist':
        clients, train_data, test_data = read_data_femnist(train_data_dir= FEMNIST_TRAIN_DATA_DIR,
                                                           test_data_dir=FEMNIST_TEST_DATA_DIR)
        train_loaders, test_loaders, v_train_loader, v_test_loader = generate_femnist_dataloader(10, clients, train_data,
                                                                                               test_data, 10)
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders, test_loaders, v_train_loader, v_test_loader
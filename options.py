import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    #dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'cifar10',
        help = 'name of the dataset: mnist, cifar10'
    )
    parser.add_argument(
        '--model',
        type = str,
        default = 'cnn',
        help='name of model. mnist: logistic, lenet; cifar10: cnn_tutorial, cnn_complex'
    )
    parser.add_argument(
        '--input_channels',
        type = int,
        default = 3,
        help = 'input channels. mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type = int,
        default = 10,
        help = 'output channels'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 10,
        help = 'batch size when trained on client'
    )
    parser.add_argument(
        '--num_communication',
        type = int,
        default=1,
        help = 'number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=1,
        help='number of local update (tau_1)'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type = int,
        default=1,
        help = 'number of edge aggregation (tau_2)'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.001,
        help = 'learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '1',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 0,
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )
    #setting for federeated learning
    parser.add_argument(
        '--iid',
        type = int,
        default = 0,
        help = 'distribution of the data, 1,0, -2(one-class)'
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default=0,
        help='distribution of the data under edges, 1 (edgeiid),0 (edgeniid) (used only when iid = -2)'
    )
    parser.add_argument(
        '--frac',
        type = float,
        default = 1,
        help = 'fraction of participated clients'
    )
    parser.add_argument(
        '--num_clients',
        type = int,
        default = 10,
        help = 'number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type = int,
        default= 1,
        help= 'number of edges'
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 1,
        help = 'random seed (defaul: 1)'
    )
    parser.add_argument(
        '--dataset_root',
        type = str,
        default = 'data',
        help = 'dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= 0,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default = 2,
        help='under artificial non-iid distribution, the classes per client'
    )
    parser.add_argument(
        '--gpu',
        type = int,
        default=0,
        help = 'GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type = int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )
    # parser.add_argument(
    #     '--aggregation',
    #     default='average',
    #     type=str,
    #     help='aggregation algorithm'
    # )

    # parser.add_argument(
    #     '--test_size',
    #     type = int,
    #     default=1000,
    #     help='valid when iid=-4, control the size of the test set'
    # )

    ## options for the finetine

    # parser.add_argument(
    #     '--fine_tune_epoch',
    #     default=50,
    #     type=int,
    #     help = 'num of epochs of finetuning the model'
    # )
    # parser.add_argument(
    #     '--fine_tune_sample',
    #     default=100,
    #     type= int,
    #     help='number of samples of finetuning the model'
    # )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args

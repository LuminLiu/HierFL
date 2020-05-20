# Hierarchical_FL
Implementation of HierFAVG algorithm in [Client-Edge-Cloud Hierarchical Federated Learning](https://arxiv.org/abs/1905.06641) with Pytorch.

For running HierFAVG with mnist and lenet:
```
python3 hierfavg 
--dataset mnist 
--model lenet 
--num_clients 50 
--num_edges 5 
--frac 1 
--num_local_update 60 
--num_edge_aggregation 1 
--num_communication 100
--batch_size 20 
--iid 0
--edgeiid 1
--show_dis 1
--lr 0.01
--lr_decay 0.995
--lr_decay_epoch 1
--momentum 0
--weight_decay 0
```
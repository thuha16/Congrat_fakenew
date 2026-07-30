from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import pickle

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, roc_curve
from torch.nn import ReLU, Sigmoid, Softmax, Dropout
import sklearn.metrics as sm
import numpy as np

import math
import numpy as np

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    f = lambda x: torch.exp(x)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Congrat(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout_rate=0.5):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                # news <-> entity relation
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # news <-> topic relation
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                
                # entity <-> entity relation
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),

                # kg entities <-> news relation
                ('kg_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # entity <-> kg entity relation
                ('kg_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
            }, aggr='sum')
            self.convs.append(conv)

        self.convs1 = torch.nn.ModuleList()
        for i in range(num_layers):
            conv1 = HeteroConv({
                # news <-> entity relation
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # news <-> topic relation
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                
                # entity <-> entity relation
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),

                # kg entities <-> news relation
                ('kg1_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg1_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # entity <-> kg entity relation
                ('kg1_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs1.append(conv1)

        self.convs2 = torch.nn.ModuleList()
        for i in range(num_layers):
            conv2 = HeteroConv({
                # news <-> entity relation
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # news <-> topic relation
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                
                # # entity <-> entity relation
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),

                # kg entities <-> news relation
                ('kg_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('kg1_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg1_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # entity <-> kg entity relation
                ('kg_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('kg1_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs2.append(conv2)

        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)


        self.lin1 = Linear(64, out_channels)
        self.lin2 = Linear(hidden_channels*3, out_channels)
        self.dropout = Dropout(p=dropout_rate)

    def l2_norm(self,input,axis = 1):
        norm = torch.norm(input,2,axis,True)
        output = torch.div(input,norm)
        return output

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def forward(self, x_dict, edge_index_dict):

        for conv in self.convs:
            x_aug1 = conv(x_dict, edge_index_dict)
            x_aug1 = {key: x.relu() for key, x in x_aug1.items()}
        # out = self.sigmoid(self.lin1(x_dict["news"]))
        # # aug_one = self.lin(x_aug1['news']) 
    
        for conv in self.convs1:
            x_aug2 = conv(x_dict, edge_index_dict)
            x_aug2 = {key: x.relu() for key, x in x_aug2.items()}


        for conv in self.convs2:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        aug_one = self.projection(x_aug1['news'])
        aug_two = self.projection(x_aug2['news'])
        aug_three = self.projection(x_dict['news'])
        news_embeddings = torch.cat([aug_one,aug_two], dim=1)
        news_embeddings = torch.cat([news_embeddings,aug_three], dim=1)
        
        # Áp dụng Dropout trước khi qua lớp Linear cuối cùng để chống Overfitting
        news_embeddings = self.dropout(news_embeddings)
        
        return aug_one, aug_two, aug_three, self.lin2(news_embeddings) 
    

def train(model, data, args):
    import copy
    if args.dataset == 'Liar':
        criterion = FocalLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out1, out2, out3, out = model(data.x_dict, data.edge_index_dict)
        mask = data['news'].train_mask

        # cal cl loss
        kg1_cl_loss = (semi_loss(out1, out3) + semi_loss(out3, out1)) * 0.5
        kg2_cl_loss = (semi_loss(out2, out3) + semi_loss(out3, out2)) * 0.5
        kg1_cl_loss = kg1_cl_loss.mean()
        kg2_cl_loss = kg2_cl_loss.mean()
        clf_loss = criterion(out[mask], data['news'].y[mask])
        
        loss = args.alpha * (kg1_cl_loss + kg2_cl_loss) / 2  + clf_loss
        loss.backward()
        optimizer.step()
        
        # Early Stopping theo dõi Validation Loss
        model.eval()
        with torch.no_grad():
            _, _, _, val_out = model(data.x_dict, data.edge_index_dict)
            val_mask = data['news'].val_mask
            val_loss = criterion(val_out[val_mask], data['news'].y[val_mask]).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                # Phục hồi model tốt nhất khi Early Stopping kích hoạt
                model.load_state_dict(best_model_state)
                break

def test(model, data, args):
    # _, _, out = model(data.x_dict, data.edge_index_dict)
    _, _, _, out = model(data.x_dict, data.edge_index_dict)
    pred = out[data['news'].test_mask].argmax(dim=1).cpu()

    y = data['news'].y[[data['news'].test_mask]].cpu()
    # pred_list = out[data['news'].test_mask].tolist()
    # predict = []
    def softmax(p):
        e_x = torch.exp(p)
        partition_x = e_x.sum(1, keepdim=True)
        return e_x / partition_x
    predict = softmax(out[data['news'].test_mask])
    col, row = predict.shape
    # print(col)
    pred_list = []
    for i in range(col):
        pred_list.append(predict[i][1].cpu().tolist())
    pred_list = torch.Tensor(pred_list)

    # print("pred_list is", pred_list)
    # print('label is', y)

  
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, )
    # 修改
    f1 = f1_score(y, pred)
    recall = recall_score(y, pred,)
    # f1_1 = f1_score(y, pred, average='weighted')
    # f1_2  = f1_score(y, pred, pos_label=0)

    auc = roc_auc_score(y, pred,)
    # Trích xuất thống kê Dataset
    num_news = data['news'].num_nodes
    num_entities = data['entities'].num_nodes
    num_topics = data['topic'].num_nodes
    num_fake = int(data['news'].y.sum().item())
    num_real = num_news - num_fake
    
    print(f"Testing Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},F1: {f1:.4f}")
    with open("./Para_analysis.txt", "a+", encoding="utf8") as f:
        f.write(f"Dataset Stats: #News={num_news}, #Fake={num_fake}, #Real={num_real}, #Entities={num_entities}, #Topics={num_topics} | ")
        f.write(f"epoch: {args.epochs}; hidden_channels: {args.hidden_channels} ; Acc:{acc:.4f}; Precision: {precision:.4f}; Recall: {recall:.4f}; F1: {f1:.4f} \n")
        
    return acc, precision, recall, f1

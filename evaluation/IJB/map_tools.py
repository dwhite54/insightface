from __future__ import print_function
from tqdm.auto import tqdm
import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torch.optim.lr_scheduler import StepLR
import math


import sklearn
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

def cosine_distance(x, y):
    return np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

def flatten_singular_values(M):
    u, _, vh = np.linalg.svd(M, full_matrices=False)
    return u @ vh
    
def fit_map(x, y, decay_coef=0.0):
    inv_mapper = sklearn.linear_model.Ridge(fit_intercept=False, normalize=False, alpha=decay_coef)
    inv_mapper.fit(x, y)
    return inv_mapper.coef_.T

def fit_procrustes_map(x, y, is_wahba=False):
    U, s, Vh = np.linalg.svd(x.T @ y)
    if is_wahba:
        m = np.ones(s.shape)
        m[-1] = np.linalg.det(U) * np.linalg.det(Vh)
        return (U[:, :s.shape[0]] * m) @ Vh[:s.shape[0], :]
    return U[:, :s.shape[0]] @ Vh[:s.shape[0], :]
#     variance_sum = np.sum(s)
#     variance_thresholded = variance_sum * explained_variance_proportion
#     S = np.eye(s.shape[0])
#     running_variance_sum = 0.0
#     i = s.shape[0] - 1
#     while i > 0:
#         variance_sum -= s[i]
#         if variance_sum <= variance_thresholded:
#             break
#         i -= 1
#     S[i:] = 0
#     print(i)
#     M = U @ S @ Vh
    
#     if do_plot:
#         plt.plot(s, label='original')
#         plt.plot(np.ones(s.shape[0]), label='procrustes')
#         plt.plot(S.diagonal(), label='thresholded_{}'.format(explained_variance_proportion))
#         plt.legend()
#         plt.yscale('log')
#         plt.show()
    
#     return M
    

# class Mapping(nn.Module):
#     def __init__(self, x_dims, y_dims):
#         super(Mapping, self).__init__()
#         self.M = nn.Linear(x_dims, y_dims, bias=False)

#     def forward(self, x):
#         return self.M(x)

# def fit_rot_map(x, y, test_idx, 
#                 BATCH_SIZE=2**15, 
#                 LR=50.0, 
#                 MOMENTUM=0.9, 
#                 LOG_INTERVAL=10, 
#                 EPOCHS=200, 
#                 LAMBDA=10.0, 
#                 GAMMA=1.0):
#     device = torch.device("cuda")
#     train_dataset = TensorDataset(torch.FloatTensor(x[:test_idx]), torch.FloatTensor(y[:test_idx]))
#     test_dataset = TensorDataset(torch.FloatTensor(x[test_idx:]), torch.FloatTensor(y[test_idx:]))
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     model = Mapping(x.shape[-1], y.shape[-1]).to(device)
#     optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
#     dist_loss = nn.MSELoss()
#     rot_loss = nn.MSELoss()
#     scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

#     I = torch.eye(y.shape[-1], dtype=torch.float32).to(device)
#     losses = np.empty((EPOCHS, 3))
#     train_loss, test_loss, rotation_loss = 0, 0, 0
#     for epoch in tqdm(range(1, EPOCHS + 1)):
#         # train
#         model.train()
#         train_loss = 0
#         MMt = model.M.weight @ torch.t(model.M.weight)
#         rotation_loss = F.mse_loss(MMt, I, reduction='sum')
        
#         for batch_idx, (emb1, emb2) in enumerate(train_loader):
#             emb1, emb2 = emb1.to(device), emb2.to(device)
#             optimizer.zero_grad()
#             emb1_mapped = model(emb1)
#             dloss = dist_loss(emb1_mapped, emb2) 
#             MMt = model.M.weight @ torch.t(model.M.weight)
#             rloss = rot_loss(MMt, I)
#             loss = dloss + LAMBDA * rloss
#             loss.backward()
#             optimizer.step()
#             train_loss += F.mse_loss(emb1_mapped, emb2, reduction='sum').item()

#         train_loss /= len(train_loader.dataset) * BATCH_SIZE

#         # test
#         model.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for emb1, emb2 in test_loader:
#                 emb1, emb2 = emb1.to(device), emb2.to(device)
#                 emb1_mapped = model(emb1)
#                 test_loss += F.mse_loss(emb1_mapped, emb2, reduction='sum').item()

#         test_loss /= len(test_loader.dataset) * BATCH_SIZE
        
#         scheduler.step()
        
#         if math.isnan(train_loss):
#             return None            
        
#         losses[epoch-1, 0] = train_loss
#         losses[epoch-1, 1] = rotation_loss
#         losses[epoch-1, 2] = test_loss

#         if epoch % LOG_INTERVAL == 0:
#             print('[*] Epoch: {}, Train loss: {:.4e}, Test loss: {:.4e}, Rot loss: {:.4e}, LR : {:.2e}'.format(epoch, train_loss, test_loss, rotation_loss, optimizer.param_groups[0]['lr']))

#     # plot loss
#     plt.plot(losses[:, 0], label='Train')
#     plt.plot(losses[:, 2], label='Test')
#     plt.legend()
#     plt.title('Loss during training')
#     plt.ylabel('MSE')
#     plt.xlabel('Epoch')
#     plt.show()
    
#     plt.plot(losses[:, 1], label='Rotation')
#     plt.show()
    
#     M = model.M.weight.cpu().detach().numpy().T
    
# #     # plot mapping
    
# #     mapping = model.M.weight.cpu().detach().numpy()
# #     plt.imshow((mapping @ mapping.T) - np.eye(mapping.shape[0]))
# #     plt.colorbar()
# #     plt.title('M @ M.T - I')
# #     plt.show()
#     Mr = flatten_singular_values(M)
    
#     plt.plot(np.linalg.svd(M)[1], label='before')
#     plt.plot(np.linalg.svd(Mr)[1], label='after')
#     plt.title('Singular values of M')
#     plt.legend()
#     plt.show()
#     return Mr # M

def plot_roc(x, y, labels, map_dict):
    #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    for name, M in map_dict.items():
        scores = cosine_distance(x @ M, y)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)
        best_threshold_idx = np.argmin(np.abs(fpr - 1e-3))
        print(name, 'best threshold', thresholds[best_threshold_idx])
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label='{} Map ROC (AUC = {:.4f})'.format(name, roc_auc), lw=2, alpha=.8)

    x_labels = [10 ** (-ii) for ii in range(1, 8)[::-1]]
    plt.xlim([10 ** -6, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(linestyle="--", linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.0, 1.0, 8, endpoint=True))
    plt.xscale("log")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cosine Distance ROC')
    plt.legend(loc="lower right")
    plt.show()

def experiment(x1, x2, y, test_idx, BATCH_SIZE=2**15, 
                LR=100.0, 
                MOMENTUM=0.9, 
                LOG_INTERVAL=1, 
                EPOCHS=50, 
                LAMBDA=10.0, GAMMA=1.0):
    M, Mr = fit_rot_map(x1, x2, test_idx, BATCH_SIZE, LR, MOMENTUM, LOG_INTERVAL, EPOCHS, LAMBDA, GAMMA)
    map_dict = {
        'Identity': np.eye(x1.shape[-1]),
        'LstSq': fit_map(x1[:test_idx], x2[:test_idx]),
        'Rotation_pre': M,
        'Rotation_post': Mr
    }
    plot_roc(x1[test_idx:], x2[test_idx:], y[test_idx:], map_dict)
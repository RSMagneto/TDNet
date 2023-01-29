import torch
import torch.nn as nn
from opt_einsum import contract
from torch import einsum

def covariance(X):
    return contract('ji...,jk...->ik...', X, X.conj())

def logdet(X):
    sgn, logdet = torch.slogdet(X)
    return sgn * logdet

def compute_loss_1d(V, y, eps=0.1):
    m, C, T = V.shape
    I = torch.eye(C).unsqueeze(-1).to(V.device)
    alpha = C / (m * eps)
    cov = alpha * covariance(V) + I
    loss_expd = logdet(cov.permute(2, 0, 1)).sum() / (2 * T)
    loss_comp = 0.
    for j in y.unique():
        a = (y == int(j))[:, 0]
        V_j = V[(y == int(j))[:, 0]]
        m_j = V_j.shape[0]
        alpha_j = C / (m_j * eps) 
        cov_j = alpha_j * covariance(V_j) + I
        loss_comp += m_j / m * logdet(cov_j.permute(2, 0, 1)).sum() / (2 * T)
    return loss_expd - loss_comp

class MCRloss(nn.Module):
    def __init__(self, batchSize, size):
        super(MCRloss, self).__init__()

        self.batchSize = batchSize
        self.pool = nn.AdaptiveAvgPool2d(size)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, ms_fea, pan_fea, all_fea):
        label = torch.cat([torch.zeros(32, 1), torch.ones(32, 1), torch.ones(32, 1) * 2], 0).int()
        ms_fea = self.conv1(self.pool(ms_fea)).flatten(2)
        pan_fea = self.conv2(self.pool(pan_fea)).flatten(2)
        all_fea = self.conv3(self.pool(all_fea)).flatten(2)
        fea = torch.cat([ms_fea, pan_fea, all_fea], 1).permute(1, 2, 0)
        mrcloss = compute_loss_1d(fea, label, 1)

        return mrcloss



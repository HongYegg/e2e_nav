import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class Gat_Fusion(nn.Module):
    def __init__(self, in_features, mid_features1, hidden, mid_feature2, out_features, dropout, alpha, concat=True):
        super(Gat_Fusion, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.alpha = alpha

        # gnn
        self.fcv = nn.Linear(in_features, mid_features1)
        nn.init.xavier_uniform_(self.fcv.weight)
        self.fck = nn.Linear(in_features, mid_features1)
        nn.init.xavier_uniform_(self.fck.weight)
        self.fcq = nn.Linear(in_features, mid_features1)
        nn.init.xavier_normal_(self.fcq.weight)
        self.fcout1 = nn.Linear(mid_features1, hidden)

        self.fcv2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_uniform_(self.fcv2.weight)
        self.fck2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_uniform_(self.fck2.weight)
        self.fcq2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_normal_(self.fcq2.weight)
        self.fcout2 = nn.Linear(mid_feature2, out_features)

        self.finalMLP = nn.Linear(out_features, out_features)

    def forward(self, h, adj, label=None):
        # label: [batch_size, 1, 12]
        # att1: [batch_size, 12, 12]
        Wh1 = F.relu(self.fcv(h))
        q1 = F.relu(self.fcq(h))
        k1 = F.relu(self.fck(h)).permute(0, 2, 1)
        att1 = F.softmax(torch.mul(torch.bmm(q1, k1), adj) - 9e15 * (1 - adj), dim=2).to(torch.float32)
        # att1 = torch.sigmoid(torch.mul(torch.bmm(q1, k1), adj) - 9e15 * (1 - adj)).to(torch.float32)

        # label = torch.zeros(1, 1, 12)
        # label[0, 0, 0] = 1
        label = label.repeat(1, 12, 1)
        att1 = att1*label

        f1 = torch.bmm(att1, Wh1)
        f2 = self.fcout1(f1)

        # Wh2 = F.relu(self.fcv2(f1))
        # q2 = F.relu(self.fcq2(f1))
        # k2 = F.relu(self.fck2(f1)).permute(0, 2, 1)
        # att2 = F.softmax(torch.mul(torch.bmm(q2, k2), adj) - 9e15 * (1 - adj), dim=2).to(torch.float32)
        # # att2 = torch.sigmoid(torch.mul(torch.bmm(q2, k2), adj) - 9e15 * (1 - adj)).to(torch.float32)
        # f2 = torch.bmm(att2, Wh2)
        # f2 = self.fcout2(f2)

        # print('f2', f2.shape)

        # fusion_1 = torch.cat([f2[:, 0, :], f2[:, 1, :], f2[:, 2, :]], dim=1)
        # fusion_2 = torch.cat([f2[:, 3, :], f2[:, 4, :], f2[:, 5, :]], dim=1)
        # fusion_3 = torch.cat([f2[:, 6, :], f2[:, 7, :], f2[:, 8, :]], dim=1)
        # fusion_4 = torch.cat([f2[:, 9, :], f2[:, 10, :], f2[:, 11, :]], dim=1)
        #
        # fusion_1_4 = torch.stack([fusion_1, fusion_2, fusion_3, fusion_4], dim=1)

        # print('fusion_1_4', fusion_1_4.shape)

        out = self.finalMLP(f2)
        out = out[:, [0, 3, 6, 9], ]

        return out


# f l r
A_same_t = np.array([[1, 0, 0],
                     [1, 0, 0],
                     [1, 0, 0]])

A_3 = np.zeros([12, 12])

A_3[0:3, 0:3] = A_same_t
A_3[3:6, 3:6] = A_same_t
A_3[6:9, 6:9] = A_same_t
A_3[9:12, 9:12] = A_same_t

adj = torch.from_numpy(A_3)

in_fea = torch.randn(8, 12, 256) # batch 3*4 256

out = torch.randn(8, 4, 256) # batch 4 256

gat_fusion = Gat_Fusion(in_features=256, mid_features1=512, hidden=256, mid_feature2=32,
                     out_features=256, dropout=0, alpha=0.2, concat=False)

out_fea = gat_fusion(in_fea, adj)
print(out_fea.shape)


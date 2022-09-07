import sys
import time
import random
import math
import torch
import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from data_load import CARLA_Data
from model_net.model_net import model_all


if __name__ == '__main__':
    batchsize = 24
    lr = 0.00001
    device = 'cuda:0'

    loss_writer = './log/0901'
    writer = SummaryWriter(log_dir=loss_writer)

    train_data = ['',
                  '',
                  '',
                  '']

    train_data = CARLA_Data(root_path=train_data, batch_size=batchsize)
    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)

    A_same_t = np.array([[1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 1],
                         [0, 0, 0, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 1, 1]])

    A_8 = np.zeros([24, 24])
    A_8[0:8, 0:8] = A_same_t
    A_8[8:16, 8:16] = A_same_t
    A_8[16:24, 16:24] = A_same_t
    A_8[0:8, 8:16] = np.eye(8)
    A_8[8:16, 0:8] = np.eye(8)
    A_8[8:16, 16:24] = np.eye(8)
    A_8[16:24, 8:16] = np.eye(8)
    A_orl = A_8

    model_all = model_all().to(device)

    model_all.model_vae_fixed_parameters.load_state_dict(torch.load('./Pre-trained_models/', map_location=device))
    model_all.model_vae.load_state_dict(torch.load('./Pre-trained_models/', map_location=device))
    # model_all.model_adj_rec.load_state_dict(torch.load('./Pre-trained_models/'))
    # model_all.model_feature_rec.load_state_dict(torch.load('./Pre-trained_models/'))
    # model_all.model_nav.load_state_dict(torch.load('./Pre-trained_models/'))

    model_all.load_state_dict(torch.load('./trained_models_/'))

    for p in model_all.model_vae_fixed_parameters.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_all.parameters()), lr=lr)
    model_all.train()

    # criterion_vae = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.MSELoss()

    step = 0

    for epoch in range(100):
        print('epoch: ', epoch)
        for i, data in enumerate(tqdm(dataloader_train), 0):
            step = step + 1

            train_data.rand()
            node_attack = np.array([data['att_nodes'][xxx][0] for xxx in range(8)])

            A_label = A_orl.copy()
            A_label[:, [node_attack[0:4]]] = 0
            A = torch.tensor(A_orl).to(device)
            A_label = torch.tensor(A_label).to(device)

            class_label = np.zeros((8, 2))
            class_label[:, 0] = 1
            class_label[0:4, 0] = 0
            class_label[0:4, 1] = 1
            class_label = torch.tensor(class_label).to(device)

            optimizer.zero_grad()

            features_rec, pred_wp = model_all(data, A)

            # vae loss
            loss_vae = criterion(model_all.dec_clean_imgs, model_all.imgs_clean_batch) + criterion(model_all.dec_clean_lidars, model_all.lidars_clean_batch) + model_all.ll_clean

            # adj rec loss
            class_labels = torch.stack([class_label for xxx in range(model_all.batch)], dim=0)
            loss_c = criterion(model_all.node_class[:, node_attack].float(), class_labels.float())
            A_labels = torch.stack([A_label for xxxx in range(model_all.batch)], dim=0)
            e_clean = torch.sum(torch.mul(model_all.adj_rec, A_labels)) / 20
            e_att = torch.sum(model_all.adj_rec[:, :, node_attack[0:4]]) / 4
            loss_e = torch.exp((- e_clean + e_att) / 24)
            loss_adj_rec = loss_c + loss_e

            # feature rec loss
            loss_feature_rec = criterion(features_rec, model_all.vae_feature_label).to(device, dtype=torch.float32)

            # model nav loss
            gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(device, dtype=torch.float32) for i in
                            range(1, len(data['waypoints']))]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float32)
            loss_nav = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()



            # model all loss
            # loss = loss_vae + loss_adj_rec + loss_feature_rec
            loss = loss_nav
            # print('loss_vae: ', loss_vae)
            # print('loss_adj_rec: ', loss_adj_rec)
            # print('loss_feature_rec: ', loss_feature_rec)
            # print('loss: ', loss)

            writer.add_scalar('train/loss', loss, step)
            # writer.add_scalar('train/loss_vae', loss_vae, step)
            # writer.add_scalar('train/loss_adj_rec', loss_adj_rec, step)
            # writer.add_scalar('train/loss_feature_rec', loss_feature_rec, step)

            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            torch.save(model_all.state_dict(), os.path.join('./trained_models', str(epoch) + 'model.pth'))

    writer.close()


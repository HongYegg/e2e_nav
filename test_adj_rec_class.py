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

from data_load import CARLA_Data
from model_net.model_net import model_all


if __name__ == '__main__':

    # 0:all; 1:model_adj_rec; 2:model_feature_rec;
    train_model = 2

    batchsize = 24
    lr = 0.00001
    device = 'cuda:0'

    train_data_path = ['/home/dataset_ssd/dataset_test1']

    #train_data = CARLA_Data(root_path=train_data, batch_size=batchsize)
    #dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=4)

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

    model_all = model_all(train_model=train_model).to(device)
    model_all.eval()

    # criterion_vae = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.MSELoss()

    step = 0

    loss_model = [[], [], []]
    accuracy = []

    for epoch in range(11):
        # print('epoch: ', epoch)
        loss_test = [[], [], []]
        sum_samp = 0
        true_samp = 0
        # model_all.load_state_dict(torch.load(os.path.join('./model_fea_rec_step1', str(0 + epoch*5) + 'model.pth')))
        model_all.load_state_dict(torch.load('./model_fea_rec_step1/170model.pth'))
        train_data = CARLA_Data(root_path=train_data_path, batch_size=batchsize, attack_level=epoch)
        dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=36)

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

            if train_model==0:
                features_rec, pred_wp = model_all(data, A)

                # # vae loss
                # loss_vae = criterion(model_all.dec_clean_imgs, model_all.imgs_clean_batch) + criterion(model_all.dec_clean_lidars, model_all.lidars_clean_batch) + model_all.ll_clean

                # # adj rec loss
                # class_labels = torch.stack([class_label for xxx in range(model_all.batch)], dim=0)
                # loss_c = criterion(model_all.node_class[:, node_attack].float(), class_labels.float())
                # A_labels = torch.stack([A_label for xxxx in range(model_all.batch)], dim=0)
                # e_clean = torch.sum(torch.mul(model_all.adj_rec, A_labels)) / 20
                # e_att = torch.sum(model_all.adj_rec[:, :, node_attack[0:4]]) / 4
                # loss_e = torch.exp((- e_clean + e_att) / 24)
                # loss_adj_rec = loss_c + loss_e

                # feature rec loss
                loss_feature_rec = criterion(features_rec, model_all.vae_feature_label).to(device, dtype=torch.float32)

                # model nav loss
                gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(device, dtype=torch.float32) for i in
                                range(1, len(data['waypoints']))]
                gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float32)
                loss_nav = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()

                # model all loss
                # loss = loss_vae + loss_adj_rec + loss_feature_rec

                loss = loss_nav + loss_feature_rec
                loss_test[0].append(loss.data)
                loss_test[1].append(loss_nav.data)
                loss_test[2].append(loss_feature_rec.data)


            if train_model==1:
                _, _ = model_all(data, A)

                # # vae loss
                # loss_vae = criterion(model_all.dec_clean_imgs, model_all.imgs_clean_batch) + criterion(model_all.dec_clean_lidars, model_all.lidars_clean_batch) + model_all.ll_clean

                # adj rec loss
                class_labels = torch.stack([class_label for xxx in range(model_all.batch)], dim=0)
                loss_c = criterion(model_all.node_class[:, node_attack].float(), class_labels.float())
                sum_samp += 8
                pre_class = model_all.node_class[:, node_attack].float()
                pre_class = pre_class[0]
                # print(pre_class)
                for p in range(4):
                    if pre_class[p][1]>pre_class[p][0]:
                        true_samp += 1
                for p in range(4,8):
                    if pre_class[p][0]>pre_class[p][1]:
                        true_samp += 1
                # print(true_samp)
                
                
                A_labels = torch.stack([A_label for xxxx in range(model_all.batch)], dim=0)
                e_clean = torch.sum(torch.mul(model_all.adj_rec, A_labels)) / 20
                e_att = torch.sum(model_all.adj_rec[:, :, node_attack[0:4]]) / 4
                loss_e = torch.exp((- e_clean + e_att) / 24)
                loss_adj_rec = loss_c + loss_e

                # # feature rec loss
                # loss_feature_rec = criterion(features_rec, model_all.vae_feature_label).to(device, dtype=torch.float32)

                # # model nav loss
                # gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(device, dtype=torch.float32) for i in
                #                 range(1, len(data['waypoints']))]
                # gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float32)
                # loss_nav = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()



                # model all loss
                # loss = loss_vae + loss_adj_rec + loss_feature_rec

                loss = loss_adj_rec

                loss_test[0].append(loss.data)
                loss_test[1].append(loss_c.data)
                loss_test[2].append(loss_e.data)


            if train_model==2:
                features_rec = model_all(data, A)

                # # vae loss
                # loss_vae = criterion(model_all.dec_clean_imgs, model_all.imgs_clean_batch) + criterion(model_all.dec_clean_lidars, model_all.lidars_clean_batch) + model_all.ll_clean

                # # adj rec loss
                # class_labels = torch.stack([class_label for xxx in range(model_all.batch)], dim=0)
                # loss_c = criterion(model_all.node_class[:, node_attack].float(), class_labels.float())
                # A_labels = torch.stack([A_label for xxxx in range(model_all.batch)], dim=0)
                # e_clean = torch.sum(torch.mul(model_all.adj_rec, A_labels)) / 20
                # e_att = torch.sum(model_all.adj_rec[:, :, node_attack[0:4]]) / 4
                # loss_e = torch.exp((- e_clean + e_att) / 24)
                # loss_adj_rec = loss_c + loss_e

                # feature rec loss
                loss_feature_rec = criterion(features_rec, model_all.vae_feature_label).to(device, dtype=torch.float32)

                # # model nav loss
                # gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(device, dtype=torch.float32) for i in
                #                 range(1, len(data['waypoints']))]
                # gt_waypoints = torch.stack(gt_waypoints, dim=1).to(device, dtype=torch.float32)
                # loss_nav = F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()



                # model all loss
                # loss = loss_vae + loss_adj_rec + loss_feature_rec

                loss = loss_feature_rec

                loss_test[0].append(loss.data)


        loss_model[0].append(torch.mean(torch.tensor(loss_test[0])))
        print('loss_model: ', loss_model[0])
        loss_model[1].append(torch.mean(torch.tensor(loss_test[1])))
        print('loss_c: ', loss_model[1])
        loss_model[2].append(torch.mean(torch.tensor(loss_test[2])))
        print('loss_e: ', loss_model[2])
        # accuracy.append(true_samp/sum_samp)
        print('accuracy: ', accuracy)


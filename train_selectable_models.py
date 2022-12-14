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
# from torch.utils.tensorboard import SummaryWriter

from data_load import CARLA_Data
from model_net.model_net import model_all


if __name__ == '__main__':

    # 0:all; 1:model_adj_rec; 2:model_feature_rec;
    train_model = 0

    batchsize = 24
    lr = 0.00001
    device = 'cuda:0'

    # loss_writer = './log_model_' + str(train_model) + '/0901'
    # writer = SummaryWriter(log_dir=loss_writer)

    train_data = ['/home/dataset_ssd/dataset2/town02/town02_long',
                  '/home/dataset_ssd/dataset2/town02/town02_short',
                  '/home/dataset_ssd/dataset2/town03/town03_long',
                  '/home/dataset_ssd/dataset2/town03/town03_short']

    train_data = CARLA_Data(root_path=train_data, batch_size=batchsize)
    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=18)

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

    model_all.model_vae.load_state_dict(torch.load('./pre_models/', map_location=device))
    model_all.load_state_dict(torch.load('./pre_models/'))

    for p in model_all.model_adj_rec.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_all.parameters()), lr=lr)

    # for p in model_all.model_vae_fixed_parameters.parameters():
    #     p.requires_grad = False

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_all.parameters()), lr=lr)
    # optimizer = optim.Adam(model_all.parameters(), lr=lr)
    model_all.train()

    # criterion_vae = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.MSELoss()

    step = 0

    total_loss=[]
    total_loss_1=[]
    total_loss_2=[]
    each_num=[]
    save_path='./model/'

    for epoch in range(100):
        writer_loss=[]
        writer_loss_1=[]
        writer_loss_2=[]
        num=0
        for i, data in enumerate(tqdm(dataloader_train), 0):
            step = step + 1
            num+=1
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
                # print('loss_vae: ', loss_vae)
                # print('loss_adj_rec: ', loss_adj_rec)
                # print('loss_feature_rec: ', loss_feature_rec)
                print('epoch: ', epoch, 'loss: ', loss, 'loss_feature_rec: ', loss_feature_rec)

                writer_loss.append(loss.item())
                writer_loss_1.append(loss_nav.item())
                writer_loss_2.append(loss_feature_rec.item())

                # writer.add_scalar('train_model_0/loss', loss, step)
                # writer.add_scalar('train_model_0/loss_vae', loss_vae, step)
                # writer.add_scalar('train_model_0/loss_adj_rec', loss_adj_rec, step)
                # writer.add_scalar('train_model_0/loss_feature_rec', loss_feature_rec, step)

                loss.backward()
                optimizer.step()

            if train_model==1:
                _, _ = model_all(data, A)

                # # vae loss
                # loss_vae = criterion(model_all.dec_clean_imgs, model_all.imgs_clean_batch) + criterion(model_all.dec_clean_lidars, model_all.lidars_clean_batch) + model_all.ll_clean

                # adj rec loss
                class_labels = torch.stack([class_label for xxx in range(model_all.batch)], dim=0)
                loss_c = criterion(model_all.node_class[:, node_attack].float(), class_labels.float())
                A_labels = torch.stack([A_label for xxxx in range(model_all.batch)], dim=0)
                e_clean = torch.sum(torch.mul(model_all.adj_rec, A_labels)) / 20
                e_att = torch.sum(model_all.adj_rec[:, :, node_attack[0:4]]) / 4
                loss_e = torch.exp((- e_clean + e_att) / 4)
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
                print('epoch: ', epoch, 'loss: ', loss, 'loss_c: ', loss_c, 'loss_e: ', loss_e)
                writer_loss.append(loss.item())
                writer_loss_1.append(loss_c.item())
                writer_loss_2.append(loss_e.item())

                # writer.add_scalar('train_model_1/loss', loss, step)
                # writer.add_scalar('train_model_1/loss_c', loss_c, step)
                # writer.add_scalar('train_model_1/loss_e', loss_e, step)

                loss.backward()
                optimizer.step()

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
                print('epoch: ', epoch, 'loss: ', loss)

                writer_loss.append(loss.item())

                # writer.add_scalar('train_model_2/loss', loss, step)

                loss.backward()
                optimizer.step()


        total_loss.append(writer_loss)
        total_loss_1.append(writer_loss_1)
        total_loss_2.append(writer_loss_2)
        each_num.append(num)
        
        filename_loss=save_path+'loss.txt'
        filename_loss_1=save_path+'loss_1.txt'
        filename_loss_2=save_path+'loss_2.txt'
        with open(filename_loss, "a", encoding='utf-8') as file:
            file.write(str(writer_loss)+'\n')
            file.close()
        with open(filename_loss_1, "a", encoding='utf-8') as file:
            file.write(str(writer_loss_1)+'\n')
            file.close()
        with open(filename_loss_2, "a", encoding='utf-8') as file:
            file.write(str(writer_loss_2)+'\n')
            file.close()

            
        torch.save(model_all.state_dict(), os.path.join('./model', 'current_model.pth'))
        if epoch % 5 == 0:
            torch.save(model_all.state_dict(), os.path.join('./model', str(0+epoch) + 'model.pth'))


    f_loss=save_path+'total_loss.txt'
    f_loss_1=save_path+'total_loss_1.txt'
    f_loss_2=save_path+'total_loss_2.txt'
    with open(f_loss, "w", encoding='utf-8') as file:
            file.write(str(total_loss))
            file.close()
    with open(f_loss_1, "w", encoding='utf-8') as file:
            file.write(str(total_loss_1))
            file.close()
    with open(f_loss_2, "w", encoding='utf-8') as file:
            file.write(str(total_loss_2))
            file.close()
    # writer.close()


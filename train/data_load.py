import os
import copy
import numpy as np
import json
import time
import math
import cv2
import torch
import random

import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform_img = transforms.Compose([transforms.Resize([128, 128]),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0., 0., 0.), (1., 1., 1.))])
transform_lidar = transforms.Compose([transforms.ToTensor()])


class ATTACK():

    def __init__(self):
        self.fore_image = np.array(Image.open('./others/car_att.png'))

    def att_img_semantic(self, base_image, x, l_r):
        fore_image = cv2.resize(self.fore_image, (200, 209))
        # base_image = base_image.copy()
        a = fore_image.shape
        resize = 2.628909478541378 / x + 2.3306802063092498 / (x * x) + 20.667384434972657 / (
                    x * x * x) - 0.03
        up_down = int(-209.7347024653435 / x + 4419.408427353277 / (x * x) - 10687.320528039269 / (
                    x * x * x) + 161.00304815257257)
        pingyi = - 1.5
        distance_car = x
        thea_ = np.pi / 2 + math.atan(pingyi / distance_car)
        self.x1 = int((thea_ + np.pi) * (180 / np.pi) * 1.25) - 1
        left_right_front = int(((self.x1 - 275) / 122) * 400)
        pingyi = -3 - 1.5
        thea = np.pi / 2 + math.atan(pingyi / distance_car)
        self.x1 = int((thea + np.pi) * (180 / np.pi) * 1.25) - 1
        left_right_l_r = int(((self.x1 - 275) / 122) * 400)

        if l_r == 3:
            left_right = int(left_right_front)  # 正前方
        if l_r == 2:
            left_right = int(left_right_front - (left_right_front - left_right_l_r)/3)
        if l_r == 1:
            left_right = int(left_right_front - 2*(left_right_front - left_right_l_r)/3)
        if l_r == 0:
            left_right = int(left_right_l_r)
        if l_r == 4:
            left_right = int(left_right_front + (left_right_front - left_right_l_r)/3)
        if l_r == 5:
            left_right = int(left_right_front + 2*(left_right_front - left_right_l_r)/3)
        if l_r == 6:
            left_right = int(left_right_front + 3*(left_right_front - left_right_l_r)/3)
        x = a[0] * resize
        y = a[1] * resize

        if up_down + int(x) >= 300 and left_right <= 0:
            base_part = base_image[up_down:, :left_right + int(y), :]
            fore_image = cv2.resize(fore_image, (int(y), int(x)))
        elif up_down + int(x) >= 300 and left_right + int(y) >= 400:
            base_part = base_image[up_down:, left_right:, :]
            fore_image = cv2.resize(fore_image, (int(y), int(x)))
        elif up_down + int(x) >= 300 and 0 < left_right and left_right + int(y) < 400:
            base_part = base_image[up_down:, left_right:left_right + int(y), :]
            fore_image = cv2.resize(fore_image, (int(y), int(x)))
        elif left_right <= 0 and up_down + int(x) < 300:
            base_part = base_image[up_down:, :left_right + int(y), :]
            fore_image = cv2.resize(fore_image, (int(y), int(x)))
        elif left_right + int(y) >= 400 and up_down + int(x) < 300:
            base_part = base_image[up_down:up_down + int(x), left_right:, :]
            fore_image = cv2.resize(fore_image, (int(y), int(x)))
        else:
            base_part = base_image[up_down:up_down + int(x), left_right:left_right + int(y), :]
            fore_image = cv2.resize(fore_image, (base_part.shape[1], base_part.shape[0]))

        base_part = base_image[up_down:up_down + int(x), left_right:left_right + int(y), :]
        fore_image = cv2.resize(fore_image, (base_part.shape[1], base_part.shape[0]))

        scope_map = fore_image[:, :, -1] / 255
        scope_map = scope_map[:, :, np.newaxis]
        scope_map = np.repeat(scope_map, repeats=3, axis=2)
        res_ = np.multiply(scope_map, np.array(fore_image)[:, :, :3])
        for i in range(res_.shape[0]):
            for p in range(res_.shape[1]):
                for z in range(res_.shape[2]):
                    if res_[i][p][z] != 0.:
                        res_[i][p][z] = fore_image[i][p][z] / 255
        if up_down + int(x) >= 300 and 0 < left_right and left_right + int(y) < 400:
            base_part = cv2.copyMakeBorder(base_part, 0, int(up_down + int(x) - 300), 0, 0,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif up_down + int(x) >= 300 and left_right <= 0:
            base_part = cv2.copyMakeBorder(base_part, 0, int(up_down + int(x) - 300), int(-left_right), 0,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif up_down + int(x) >= 300 and left_right + int(y) >= 400:
            base_part = cv2.copyMakeBorder(base_part, 0, int(up_down + int(x) - 300), 0, int(left_right + int(y) - 400),
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif left_right <= 0 and up_down + int(x) < 300:
            base_part = cv2.copyMakeBorder(base_part, 0, 0, int(-left_right), 0,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif left_right + int(y) >= 400 and up_down + int(x) < 300:
            base_part = cv2.copyMakeBorder(base_part, 0, 0, 0, int(left_right + int(y) - 400),
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])

        res_image = res_ + np.multiply((1 - scope_map), base_part)
        if up_down + int(x) >= 300 and 0 < left_right and left_right + int(y) < 400:
            base_image[up_down:, left_right:left_right + int(y)] = res_image[:300 - up_down, :]
        elif up_down + int(x) >= 300 and left_right <= 0:
            base_image[up_down:, :left_right + int(y)] = res_image[:300 - up_down, int(-left_right):]
        elif up_down + int(x) >= 300 and left_right + int(y) >= 400:
            base_image[up_down:, left_right:] = res_image[:300 - up_down, :400 - left_right]
        elif left_right <= 0 and up_down + int(x) < 300:
            base_image[up_down:up_down + int(x), :left_right + int(y)] = res_image[:, int(-left_right):]
        elif left_right + int(y) >= 400 and up_down + int(x) < 300:
            base_image[up_down:up_down + int(x), left_right:] = res_image[:, :400 - left_right]
        else:
            base_image[up_down:up_down + int(x), left_right:left_right + int(y)] = res_image

        base_image[up_down:up_down + int(x), left_right:left_right + int(y)] = res_image

        return base_image


    def att_lidar_semantic(self,lidar, dis_att = 5, pianyi_att = 0):
        pingyi = pianyi_att-3-1.5
        distance_car = dis_att

        kuan_car = 0.9
        gao_car = 1.8
        fai = -math.atan(3 / distance_car)
        thea = np.pi / 2 + math.atan(pingyi / distance_car)

        change_thea = np.pi / 2 + math.atan((2 * kuan_car + pingyi) / distance_car)
        change_fai = -math.atan((3 - gao_car) / distance_car)

        self.x1 = int((thea + np.pi) * (180 / np.pi) * 1.25) - 1
        self.x2 = int((change_thea + np.pi) * (180 / np.pi) * 1.25) - 1

        self.y1 = 64 - int((fai + (np.pi/6)) * (180 / np.pi) * 1.6) - 1
        self.y2 = 64 - int((change_fai + (np.pi/6)) * (180 / np.pi) * 1.6) - 1


        for ii in range(self.x1, self.x2):
            for jj in range(self.y2, self.y1):
                thea_temp = ((ii + 1)/1.25)*(np.pi/180)-np.pi
                fai_temp = ((64-(jj+1))/1.6)*(np.pi/180)-(np.pi/6)
                # print('ww', dis_att, lidar[jj, ii-106], (distance_car / (math.cos(abs(thea_temp - np.pi / 2)) * math.cos(abs(fai_temp))))/100)
                if lidar[0, jj, ii-275] < 1 - (distance_car / (math.cos(abs(thea_temp - np.pi / 2)) * math.cos(abs(fai_temp))))/60:
                    lidar[0, jj, ii-275] = 1 - (distance_car / (math.cos(abs(thea_temp - np.pi / 2)) * math.cos(abs(fai_temp))))/60
                    # print('w', lidar[jj, ii-275])
                    # lidar[jj, ii - 275] = 0

        return lidar


attack = ATTACK()

unloader=transforms.ToPILImage()
def tensor2PIL(tensor):
    image=tensor.cpu().clone()
    image=image.squeeze(0)
    image=unloader(image)
    return image

class CARLA_Data(Dataset):

    def __init__(self, root_path,batch_size):
        self.batch_size = batch_size

        # self.data_new_last = None
        # self.choice_nodes = random.sample(range(0, 24), 6)
        # self.rand()
        self.fan_mat=Sector(47.2,5)
        
        self.seq_len = 1
        self.pred_len = 4

        self.counter = 0


        self.input_resolution = 256
        self.scale = 1

        self.x = []
        self.y = []
        self.theta = []

        self.x_command = []
        self.y_command = []

        self.lidar = []
        self.lidar_l = []
        self.lidar_r = []
        self.lidar_h = []
        self.lidar_rf = []
        self.lidar_lf = []
        self.lidar_rh = []
        self.lidar_lh = []

        self.front = []
        self.front_l = []
        self.front_r = []
        self.front_h = []
        self.front_rf = []
        self.front_lf = []
        self.front_rh = []
        self.front_lh = []

        self.lidar_his = []
        self.lidar_l_his = []
        self.lidar_r_his = []
        self.lidar_h_his = []
        self.lidar_rf_his = []
        self.lidar_lf_his = []
        self.lidar_rh_his = []
        self.lidar_lh_his = []

        self.front_his = []
        self.front_l_his = []
        self.front_r_his = []
        self.front_h_his = []
        self.front_rf_his = []
        self.front_lf_his = []
        self.front_rh_his = []
        self.front_lh_his = []

        self.red_light = []

        for sub_root in root_path:
            preload_file = './others/dataset_0901.npy'
            if True:
                preload_x = []
                preload_y = []
                preload_theta = []
                preload_x_command = []
                preload_y_command = []

                preload_front = []
                preload_lidar = []
                preload_front_l = []
                preload_lidar_l = []
                preload_front_r = []
                preload_lidar_r = []
                preload_front_h = []
                preload_lidar_h = []
                preload_front_rf = []
                preload_lidar_rf = []
                preload_front_lf = []
                preload_lidar_lf = []
                preload_front_rh = []
                preload_lidar_rh = []
                preload_front_lh = []
                preload_lidar_lh = []

                preload_front_his = []
                preload_lidar_his = []
                preload_front_l_his = []
                preload_lidar_l_his = []
                preload_front_r_his = []
                preload_lidar_r_his = []
                preload_front_h_his = []
                preload_lidar_h_his = []
                preload_front_rf_his = []
                preload_lidar_rf_his = []
                preload_front_lf_his = []
                preload_lidar_lf_his = []
                preload_front_rh_his = []
                preload_lidar_rh_his = []
                preload_front_lh_his = []
                preload_lidar_lh_his = []

                preload_red_light = []

                root_files = os.listdir(sub_root)

                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root, folder))]

                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)

                    num_seq = (len(os.listdir(route_dir + "/rgb_front/")) - self.pred_len - 2) // self.seq_len

                    for seq in range(num_seq-20):

                        xs = []
                        ys = []
                        thetas = []

                        fronts = []
                        lidars = []
                        fronts_l = []
                        lidars_l = []
                        fronts_r = []
                        lidars_r = []
                        fronts_h = []
                        lidars_h = []
                        fronts_rf = []
                        lidars_rf = []
                        fronts_lf = []
                        lidars_lf = []
                        fronts_rh = []
                        lidars_rh = []
                        fronts_lh = []
                        lidars_lh = []

                        fronts_his = []
                        lidars_his = []
                        fronts_l_his = []
                        lidars_l_his = []
                        fronts_r_his = []
                        lidars_r_his = []
                        fronts_h_his = []
                        lidars_h_his = []
                        fronts_rf_his = []
                        lidars_rf_his = []
                        fronts_lf_his = []
                        lidars_lf_his = []
                        fronts_rh_his = []
                        lidars_rh_his = []
                        fronts_lh_his = []
                        lidars_lh_his = []

                        red_light = []

                        # read files sequentially (past and current frames)
                        for i in range(self.seq_len):
                            # images
                            for j in range(3): # 0, 1, 2
                                numb = seq * self.seq_len + i + j + 1
                                if numb>=10000:
                                    filename = f"{str(seq * self.seq_len + i + j + 1).zfill(4)}.png"
                                else:
                                    filename = f"{str(seq * self.seq_len + i + j + 1).zfill(4)}.png"
                                fronts.append(route_dir + "/rgb_front/" + filename)
                                fronts_l.append(route_dir + "/rgb_left/" + filename)
                                fronts_r.append(route_dir + "/rgb_right/" + filename)
                                fronts_h.append(route_dir + "/rgb_rear/" + filename)
                                fronts_rf.append(route_dir + "/rgb_front_right/" + filename)
                                fronts_lf.append(route_dir + "/rgb_front_left/" + filename)
                                fronts_rh.append(route_dir + "/rgb_rear_right/" + filename)
                                fronts_lh.append(route_dir + "/rgb_rear_left/" + filename)
                                # print('img address',filename)

                                if j ==0:
                                    file_id = seq * self.seq_len + j + i - 10
                                    if file_id<=0:
                                        file_id=0
                                    filename_his = f"{str(file_id).zfill(4)}.png"
                                    fronts_his.append(route_dir + "/rgb_front/" + filename_his)
                                    fronts_l_his.append(route_dir + "/rgb_left/" + filename_his)
                                    fronts_r_his.append(route_dir + "/rgb_right/" + filename_his)
                                    fronts_h_his.append(route_dir + "/rgb_rear/" + filename_his)
                                    fronts_rf_his.append(route_dir + "/rgb_front_right/" + filename_his)
                                    fronts_lf_his.append(route_dir + "/rgb_front_left/" + filename_his)
                                    fronts_rh_his.append(route_dir + "/rgb_rear_right/" + filename_his)
                                    fronts_lh_his.append(route_dir + "/rgb_rear_left/" + filename_his)

                            # lidars
                            for j in range(3):
                                numb = seq * self.seq_len + i + j + 1
                                if numb >= 10000:
                                    filename = f"{str(seq * self.seq_len + i + j + 1).zfill(4)}.npy"
                                else:
                                    filename = f"{str(seq * self.seq_len + i + j + 1).zfill(4)}.npy"

                                lidars.append(route_dir + "/lidar_qvzao_front/" + filename)
                                lidars_l.append(route_dir + "/lidar_qvzao_left/" + filename)
                                lidars_r.append(route_dir + "/lidar_qvzao_right/" + filename)
                                lidars_h.append(route_dir + "/lidar_qvzao_rear/" + filename)
                                lidars_rf.append(route_dir + "/lidar_qvzao_front_right/" + filename)
                                lidars_lf.append(route_dir + "/lidar_qvzao_front_left/" + filename)
                                lidars_rh.append(route_dir + "/lidar_qvzao_rear_right/" + filename)
                                lidars_lh.append(route_dir + "/lidar_qvzao_rear_left/" + filename)

                                if j ==0:
                                    file_id = seq * self.seq_len + j + i - 10
                                    if file_id<=0:
                                        file_id=0
                                    filename_his = f"{str(file_id).zfill(4)}.npy"
                                    lidars_his.append(route_dir + "/lidar_qvzao_front/" + filename_his)
                                    lidars_l_his.append(route_dir + "/lidar_qvzao_left/" + filename_his)
                                    lidars_r_his.append(route_dir + "/lidar_qvzao_right/" + filename_his)
                                    lidars_h_his.append(route_dir + "/lidar_qvzao_rear/" + filename_his)
                                    lidars_rf_his.append(route_dir + "/lidar_qvzao_front_right/" + filename_his)
                                    lidars_lf_his.append(route_dir + "/lidar_qvzao_front_left/" + filename_his)
                                    lidars_rh_his.append(route_dir + "/lidar_qvzao_rear_right/" + filename_his)
                                    lidars_lh_his.append(route_dir + "/lidar_qvzao_rear_left/" + filename_his)

                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i+2).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                                # print('current position address', read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])
                            red_light.append(data['is_red_light_present'])


                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])

                        for i in range(self.seq_len, self.seq_len + self.pred_len):
                            with open(route_dir + f"/measurements/{str(seq * self.seq_len + 1+2 + i*5).zfill(4)}.json","r") as read_file:
                                data = json.load(read_file)
                                # print('predicted position address', read_file)

                            xs.append(data['x'])
                            ys.append(data['y'])

                            # fix for theta=nan in some measurements
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])

                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)

                        preload_front.append(fronts)
                        preload_lidar.append(lidars)
                        preload_front_l.append(fronts_l)
                        preload_lidar_l.append(lidars_l)
                        preload_front_r.append(fronts_r)
                        preload_lidar_r.append(lidars_r)
                        preload_front_h.append(fronts_h)
                        preload_lidar_h.append(lidars_h)
                        preload_front_rf.append(fronts_rf)
                        preload_lidar_rf.append(lidars_rf)
                        preload_front_lf.append(fronts_lf)
                        preload_lidar_lf.append(lidars_lf)
                        preload_front_rh.append(fronts_rh)
                        preload_lidar_rh.append(lidars_rh)
                        preload_front_lh.append(fronts_lh)
                        preload_lidar_lh.append(lidars_lh)

                        preload_front_his.append(fronts_his)
                        preload_lidar_his.append(lidars_his)
                        preload_front_l_his.append(fronts_l_his)
                        preload_lidar_l_his.append(lidars_l_his)
                        preload_front_r_his.append(fronts_r_his)
                        preload_lidar_r_his.append(lidars_r_his)
                        preload_front_h_his.append(fronts_h_his)
                        preload_lidar_h_his.append(lidars_h_his)
                        preload_front_rf_his.append(fronts_rf_his)
                        preload_lidar_rf_his.append(lidars_rf_his)
                        preload_front_lf_his.append(fronts_lf_his)
                        preload_lidar_lf_his.append(lidars_lf_his)
                        preload_front_rh_his.append(fronts_rh_his)
                        preload_lidar_rh_his.append(lidars_rh_his)
                        preload_front_lh_his.append(fronts_lh_his)
                        preload_lidar_lh_his.append(lidars_lh_his)

                        preload_red_light.append(red_light)

                        # print('\n')

                # dump to npy
                preload_dict = {}

                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['theta'] = preload_theta
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command

                preload_dict['front'] = preload_front
                preload_dict['lidar'] = preload_lidar
                preload_dict['front_l'] = preload_front_l
                preload_dict['lidar_l'] = preload_lidar_l
                preload_dict['front_r'] = preload_front_r
                preload_dict['lidar_r'] = preload_lidar_r
                preload_dict['front_h'] = preload_front_h
                preload_dict['lidar_h'] = preload_lidar_h
                preload_dict['front_rf'] = preload_front_rf
                preload_dict['lidar_rf'] = preload_lidar_rf
                preload_dict['front_lf'] = preload_front_lf
                preload_dict['lidar_lf'] = preload_lidar_lf
                preload_dict['front_rh'] = preload_front_rh
                preload_dict['lidar_rh'] = preload_lidar_rh
                preload_dict['front_lh'] = preload_front_lh
                preload_dict['lidar_lh'] = preload_lidar_lh

                preload_dict['front_his'] = preload_front_his
                preload_dict['lidar_his'] = preload_lidar_his
                preload_dict['front_l_his'] = preload_front_l_his
                preload_dict['lidar_l_his'] = preload_lidar_l_his
                preload_dict['front_r_his'] = preload_front_r_his
                preload_dict['lidar_r_his'] = preload_lidar_r_his
                preload_dict['front_h_his'] = preload_front_h_his
                preload_dict['lidar_h_his'] = preload_lidar_h_his
                preload_dict['front_rf_his'] = preload_front_rf_his
                preload_dict['lidar_rf_his'] = preload_lidar_rf_his
                preload_dict['front_lf_his'] = preload_front_lf_his
                preload_dict['lidar_lf_his'] = preload_lidar_lf_his
                preload_dict['front_rh_his'] = preload_front_rh_his
                preload_dict['lidar_rh_his'] = preload_lidar_rh_his
                preload_dict['front_lh_his'] = preload_front_lh_his
                preload_dict['lidar_lh_his'] = preload_lidar_lh_his

                preload_dict['red_light'] = preload_red_light

                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)

            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.theta += preload_dict.item()['theta']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']

            self.front += preload_dict.item()['front']
            self.lidar += preload_dict.item()['lidar']
            self.front_l += preload_dict.item()['front_l']
            self.lidar_l += preload_dict.item()['lidar_l']
            self.front_r += preload_dict.item()['front_r']
            self.lidar_r += preload_dict.item()['lidar_r']
            self.front_h += preload_dict.item()['front_h']
            self.lidar_h += preload_dict.item()['lidar_h']
            self.front_rf += preload_dict.item()['front_rf']
            self.lidar_rf += preload_dict.item()['lidar_rf']
            self.front_lf += preload_dict.item()['front_lf']
            self.lidar_lf += preload_dict.item()['lidar_lf']
            self.front_rh += preload_dict.item()['front_rh']
            self.lidar_rh += preload_dict.item()['lidar_rh']
            self.front_lh += preload_dict.item()['front_lh']
            self.lidar_lh += preload_dict.item()['lidar_lh']

            self.front_his += preload_dict.item()['front_his']
            self.lidar_his += preload_dict.item()['lidar_his']
            self.front_l_his += preload_dict.item()['front_l_his']
            self.lidar_l_his += preload_dict.item()['lidar_l_his']
            self.front_r_his += preload_dict.item()['front_r_his']
            self.lidar_r_his += preload_dict.item()['lidar_r_his']
            self.front_h_his += preload_dict.item()['front_h_his']
            self.lidar_h_his += preload_dict.item()['lidar_h_his']
            self.front_rf_his += preload_dict.item()['front_rf_his']
            self.lidar_rf_his += preload_dict.item()['lidar_rf_his']
            self.front_lf_his += preload_dict.item()['front_lf_his']
            self.lidar_lf_his += preload_dict.item()['lidar_lf_his']
            self.front_rh_his += preload_dict.item()['front_rh_his']
            self.lidar_rh_his += preload_dict.item()['lidar_rh_his']
            self.front_lh_his += preload_dict.item()['front_lh_his']
            self.lidar_lh_his += preload_dict.item()['lidar_lh_his']

            self.red_light += preload_dict.item()['red_light']

            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """

        # t->0, 1, 2(now_t)

        # 0~7:t(0); 8~15:t(1), 16~23:t(2)   shijiao: f, lf, l, lh, h, rh, r, rf

        # att_nodes = np.array([0, 1, 2, 3, 4, 5])

        # choice_nodes = random.sample(range(0, 24), 6)

        if self.counter%self.batch_size==0:
            self.rand()
        att_nodes = self.choice_nodes[0:4]

        self.counter+=1

        # clean, att; shijiao:
        data_new = dict()
        data_new['imgs_clean'] = []
        data_new['imgs_clean_ori'] = []
        data_new['imgs_att'] = []
        data_new['lidars_clean'] = []
        data_new['lidars_att'] = []

        data_new['imgs_his'] = []
        data_new['lidars_his'] = []

        seq_fronts = self.front[index]
        seq_lidars = self.lidar[index]
        seq_fronts_l = self.front_l[index]
        seq_lidars_l = self.lidar_l[index]
        seq_fronts_r = self.front_r[index]
        seq_lidars_r = self.lidar_r[index]
        seq_fronts_h = self.front_h[index]
        seq_lidars_h = self.lidar_h[index]
        seq_fronts_rf = self.front_rf[index]
        seq_lidars_rf = self.lidar_rf[index]
        seq_fronts_lf = self.front_lf[index]
        seq_lidars_lf = self.lidar_lf[index]
        seq_fronts_rh = self.front_rh[index]
        seq_lidars_rh = self.lidar_rh[index]
        seq_fronts_lh = self.front_lh[index]
        seq_lidars_lh = self.lidar_lh[index]

        seq_fronts_his = self.front_his[index]
        seq_lidars_his = self.lidar_his[index]
        seq_fronts_l_his = self.front_l_his[index]
        seq_lidars_l_his = self.lidar_l_his[index]
        seq_fronts_r_his = self.front_r_his[index]
        seq_lidars_r_his = self.lidar_r_his[index]
        seq_fronts_h_his = self.front_h_his[index]
        seq_lidars_h_his = self.lidar_h_his[index]
        seq_fronts_rf_his = self.front_rf_his[index]
        seq_lidars_rf_his = self.lidar_rf_his[index]
        seq_fronts_lf_his = self.front_lf_his[index]
        seq_lidars_lf_his = self.lidar_lf_his[index]
        seq_fronts_rh_his = self.front_rh_his[index]
        seq_lidars_rh_his = self.lidar_rh_his[index]
        seq_fronts_lh_his = self.front_lh_his[index]
        seq_lidars_lh_his = self.lidar_lh_his[index]

        for i in range(len(seq_fronts)):
            data_new['imgs_clean'].append(transform_img(Image.open(seq_fronts[i])))
            data_new['imgs_clean'].append(transform_img(Image.open(seq_fronts_lf[i])))
            data_new['imgs_clean'].append(transform_img(Image.open(seq_fronts_l[i])))
            data_new['imgs_clean'].append(transform_img(Image.open(seq_fronts_lh[i])))
            data_new['imgs_clean'].append(transform_img(Image.open(seq_fronts_h[i])))
            data_new['imgs_clean'].append(transform_img(Image.open(seq_fronts_rh[i])))
            data_new['imgs_clean'].append(transform_img(Image.open(seq_fronts_r[i])))
            data_new['imgs_clean'].append(transform_img(Image.open(seq_fronts_rf[i])))

            data_new['imgs_clean_ori'].append(np.array(plt.imread(seq_fronts[i])))
            data_new['imgs_clean_ori'].append(np.array(plt.imread(seq_fronts_lf[i])))
            data_new['imgs_clean_ori'].append(np.array(plt.imread(seq_fronts_l[i])))
            data_new['imgs_clean_ori'].append(np.array(plt.imread(seq_fronts_lh[i])))
            data_new['imgs_clean_ori'].append(np.array(plt.imread(seq_fronts_h[i])))
            data_new['imgs_clean_ori'].append(np.array(plt.imread(seq_fronts_rh[i])))
            data_new['imgs_clean_ori'].append(np.array(plt.imread(seq_fronts_r[i])))
            data_new['imgs_clean_ori'].append(np.array(plt.imread(seq_fronts_rf[i])))

            if i==1:
                data_new['imgs_his'].append(transform_img(Image.open(seq_fronts_his[0])))
                data_new['imgs_his'].append(transform_img(Image.open(seq_fronts_lf_his[0])))
                data_new['imgs_his'].append(transform_img(Image.open(seq_fronts_l_his[0])))
                data_new['imgs_his'].append(transform_img(Image.open(seq_fronts_lh_his[0])))
                data_new['imgs_his'].append(transform_img(Image.open(seq_fronts_h_his[0])))
                data_new['imgs_his'].append(transform_img(Image.open(seq_fronts_rh_his[0])))
                data_new['imgs_his'].append(transform_img(Image.open(seq_fronts_r_his[0])))
                data_new['imgs_his'].append(transform_img(Image.open(seq_fronts_rf_his[0])))

                lidar_d = np.load(seq_lidars[0]).astype(np.float32)
                data_new['lidars_his'].append(transform_lidar(lidar_d))
                lidar_d_lf = np.load(seq_lidars_lf_his[0]).astype(np.float32)
                data_new['lidars_his'].append(transform_lidar(lidar_d_lf))
                lidar_d_l = np.load(seq_lidars_l_his[0]).astype(np.float32)
                data_new['lidars_his'].append(transform_lidar(lidar_d_l))
                lidar_d_lh = np.load(seq_lidars_lh_his[0]).astype(np.float32)
                data_new['lidars_his'].append(transform_lidar(lidar_d_lh))
                lidar_d_h = np.load(seq_lidars_h_his[0]).astype(np.float32)
                data_new['lidars_his'].append(transform_lidar(lidar_d_h))
                lidar_d_rh = np.load(seq_lidars_rh_his[0]).astype(np.float32)
                data_new['lidars_his'].append(transform_lidar(lidar_d_rh))
                lidar_d_r = np.load(seq_lidars_r_his[0]).astype(np.float32)
                data_new['lidars_his'].append(transform_lidar(lidar_d_r))
                lidar_d_rf = np.load(seq_lidars_rf_his[0]).astype(np.float32)
                data_new['lidars_his'].append(transform_lidar(lidar_d_rf))

            lidar_d = np.load(seq_lidars_his[i]).astype(np.float32)
            data_new['lidars_clean'].append(transform_lidar(lidar_d))
            lidar_d_lf = np.load(seq_lidars_lf[i]).astype(np.float32)
            data_new['lidars_clean'].append(transform_lidar(lidar_d_lf))
            lidar_d_l = np.load(seq_lidars_l[i]).astype(np.float32)
            data_new['lidars_clean'].append(transform_lidar(lidar_d_l))
            lidar_d_lh = np.load(seq_lidars_lh[i]).astype(np.float32)
            data_new['lidars_clean'].append(transform_lidar(lidar_d_lh))
            lidar_d_h = np.load(seq_lidars_h[i]).astype(np.float32)
            data_new['lidars_clean'].append(transform_lidar(lidar_d_h))
            lidar_d_rh = np.load(seq_lidars_rh[i]).astype(np.float32)
            data_new['lidars_clean'].append(transform_lidar(lidar_d_rh))
            lidar_d_r = np.load(seq_lidars_r[i]).astype(np.float32)
            data_new['lidars_clean'].append(transform_lidar(lidar_d_r))
            lidar_d_rf = np.load(seq_lidars_rf[i]).astype(np.float32)
            data_new['lidars_clean'].append(transform_lidar(lidar_d_rf))

            # att data
            data_new['imgs_att'].append(transform_img(Image.open(seq_fronts[i])))
            data_new['imgs_att'].append(transform_img(Image.open(seq_fronts_lf[i])))
            data_new['imgs_att'].append(transform_img(Image.open(seq_fronts_l[i])))
            data_new['imgs_att'].append(transform_img(Image.open(seq_fronts_lh[i])))
            data_new['imgs_att'].append(transform_img(Image.open(seq_fronts_h[i])))
            data_new['imgs_att'].append(transform_img(Image.open(seq_fronts_rh[i])))
            data_new['imgs_att'].append(transform_img(Image.open(seq_fronts_r[i])))
            data_new['imgs_att'].append(transform_img(Image.open(seq_fronts_rf[i])))

            lidar_d = np.load(seq_lidars[i]).astype(np.float32)
            data_new['lidars_att'].append(transform_lidar(lidar_d))
            lidar_d_lf = np.load(seq_lidars_lf[i]).astype(np.float32)
            data_new['lidars_att'].append(transform_lidar(lidar_d_lf))
            lidar_d_l = np.load(seq_lidars_l[i]).astype(np.float32)
            data_new['lidars_att'].append(transform_lidar(lidar_d_l))
            lidar_d_lh = np.load(seq_lidars_lh[i]).astype(np.float32)
            data_new['lidars_att'].append(transform_lidar(lidar_d_lh))
            lidar_d_h = np.load(seq_lidars_h[i]).astype(np.float32)
            data_new['lidars_att'].append(transform_lidar(lidar_d_h))
            lidar_d_rh = np.load(seq_lidars_rh[i]).astype(np.float32)
            data_new['lidars_att'].append(transform_lidar(lidar_d_rh))
            lidar_d_r = np.load(seq_lidars_r[i]).astype(np.float32)
            data_new['lidars_att'].append(transform_lidar(lidar_d_r))
            lidar_d_rf = np.load(seq_lidars_rf[i]).astype(np.float32)
            data_new['lidars_att'].append(transform_lidar(lidar_d_rf))



        for att_node in att_nodes:
            att_t = att_node // 8
            att_shijiao = att_node % 8
            # 1:Shelter; 2:noise; 3:Information all black(Loss of frames); 4:Brightness and contrast; 5:Frame overlap; 6:Semantic attacks
            # 7:Repeat frame; 8:Fan lost(5 degree); 9:Multi-sensor desynchronization (homogeneous desynchronization); 
            # 10:Multi-sensor desynchronization (heterogeneous desynchronization)
            att_type = np.random.randint(1, 11)
            # att_type = 1

            if att_type == 0:
                img_att = data_new['imgs_att'][att_t*8 + att_shijiao]
                lidar_att = data_new['lidars_att'][att_t * 8 + att_shijiao]
                img_att_ori = data_new['imgs_clean_ori'][att_t * 8 + att_shijiao]


                img_att1 = img_att.clone()
                img_att1[:, 32:, 32:96] = 0
                lidar_att1 = lidar_att.clone()
                lidar_att1[:, 16:, 30:90] = 0

                Attack_intensity = random.uniform(0.35, 0.45)
                img_att2 = sp_noise(img_att, Attack_intensity)
                lidar_att2 = sp_noise(lidar_att, Attack_intensity)

                img_att3 = img_att.clone()
                img_att3[:, :, :] = 0
                lidar_att3 = lidar_att.clone()
                lidar_att3[:, :, :] = 0

                img_att4 = bright_contrast(1.5, 0.5, img_att)
                lidar_att4 = bright_contrast(1.5, 0.5, lidar_att)

                img_att_t_1 = data_new['imgs_att'][(att_t - 1) * 8 + att_shijiao]
                img_att5 = (img_att + img_att_t_1) / 2
                lidar_att_t_1 = data_new['lidars_att'][(att_t - 1) * 8 + att_shijiao]
                lidar_att5 = (lidar_att + lidar_att_t_1) / 2

                img_att6 = attack.att_img_semantic(img_att_ori, 8, 3)
                img_att6 = transform_img(Image.fromarray(np.uint8(img_att_ori * 255)))
                lidar_att6 = attack.att_lidar_semantic(lidar_att, 8, 3)

                img_orl_show = img_att.cpu().numpy().transpose(1, 2, 0)
                lidar_orl_show = lidar_att.cpu().numpy().transpose(1, 2, 0)
                img_att1_show = img_att1.cpu().numpy().transpose(1, 2, 0)
                lidar_att1_show = lidar_att1.cpu().numpy().transpose(1, 2, 0)
                img_att2_show = img_att2.cpu().numpy().transpose(1, 2, 0)
                lidar_att2_show = lidar_att2.cpu().numpy().transpose(1, 2, 0)
                img_att3_show = img_att3.cpu().numpy().transpose(1, 2, 0)
                lidar_att3_show = lidar_att3.cpu().numpy().transpose(1, 2, 0)
                img_att4_show = img_att4.cpu().numpy().transpose(1, 2, 0)
                lidar_att4_show = lidar_att4.cpu().numpy().transpose(1, 2, 0)
                img_att5_show = img_att5.cpu().numpy().transpose(1, 2, 0)
                lidar_att5_show = lidar_att5.cpu().numpy().transpose(1, 2, 0)
                img_att6_show = img_att6.cpu().numpy().transpose(1, 2, 0)
                lidar_att6_show = lidar_att6.cpu().numpy().transpose(1, 2, 0)

                img_orl_show = cv2.cvtColor(img_orl_show, cv2.COLOR_BGR2RGB)
                img_att1_show = cv2.cvtColor(img_att1_show, cv2.COLOR_BGR2RGB)
                img_att2_show = cv2.cvtColor(img_att2_show, cv2.COLOR_BGR2RGB)
                img_att3_show = cv2.cvtColor(img_att3_show, cv2.COLOR_BGR2RGB)
                img_att4_show = cv2.cvtColor(img_att4_show, cv2.COLOR_BGR2RGB)
                img_att5_show = cv2.cvtColor(img_att5_show, cv2.COLOR_BGR2RGB)
                img_att6_show = cv2.cvtColor(img_att6_show, cv2.COLOR_BGR2RGB)

                # cv2.imshow('img_att_show', img_orl_show)
                # cv2.imshow('lidar_orl_show', lidar_orl_show)
                # cv2.imshow('img_att1_show', img_att1_show)
                # cv2.imshow('lidar_att1_show', lidar_att1_show)
                # cv2.imshow('img_att2_show', img_att2_show)
                # cv2.imshow('lidar_att2_show', lidar_att2_show)
                # cv2.imshow('img_att3_show', img_att3_show)
                # cv2.imshow('lidar_att3_show', lidar_att3_show)
                # cv2.imshow('img_att4_show', img_att4_show)
                # cv2.imshow('lidar_att4_show', lidar_att4_show)
                # cv2.imshow('img_att5_show', img_att5_show)
                # cv2.imshow('lidar_att5_show', lidar_att5_show)
                # cv2.imshow('img_att6_show', img_att6_show)
                # cv2.imshow('lidar_att6_show', lidar_att6_show)
                # cv2.waitKey(10)
                # time.sleep(600)

            if att_type == 1:
                img_att = data_new['imgs_att'][att_t*8 + att_shijiao]
                #######################
                # shelter with random size and pos
                ######################3
                self.s1_x=random.randint(5,11)
                self.s1_y=random.randint(5,11)
                x_min = np.random.randint(0, int(128 - 9.6 * self.s1_x))
                y_min = np.random.randint(0, 128 - int(5.4 * self.s1_y))
                x_max = x_min + int(9.6 * self.s1_x)
                y_max = y_min + int(5.4 * self.s1_y)
                self.x1 = x_min / 128
                self.y1 = y_min / 128
                #print(img_att.size())
                img_att[:, x_min:x_max, y_min:y_max] = 0
                data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att

                # att lidar
                lidar_att_zhedang = data_new['lidars_att'][att_t * 8 + att_shijiao]
                #print(lidar_att_zhedang.size())
                x_min=int(self.x1*64)
                y_min=int(self.y1*122)
                x_max=int(self.x1*64+4.8*self.s1_x)
                y_max=int(self.y1*122+5.4*self.s1_y)
                lidar_att_zhedang[:, x_min:x_max, y_min:y_max] = 0
                data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang

                # #test
                # img1=tensor2PIL(lidar_att_zhedang)
                # img2=tensor2PIL(img_att)
                # #print(type(img))
                # plt.figure()
                # plt.subplot(2,1,1)
                # plt.imshow(img1)
                # plt.subplot(2,1,2)
                # plt.imshow(img2)
                # plt.show()

            if att_type == 2:
                Attack_intensity = random.uniform(0.25, 0.45)
                # att img
                img_att = data_new['imgs_att'][att_t*8 + att_shijiao]
                img_att = sp_noise(img_att, Attack_intensity)
                data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att

                # att lidar
                lidar_att_zhedang = data_new['lidars_att'][att_t * 8 + att_shijiao]
                lidar_att_zhedang = sp_noise(lidar_att_zhedang, Attack_intensity)
                data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang

            if att_type == 3:
                img_att = data_new['imgs_att'][att_t*8 + att_shijiao]
                img_att[:, :, :] = 0
                data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att

                # att lidar
                lidar_att_zhedang = data_new['lidars_att'][att_t * 8 + att_shijiao]
                lidar_att_zhedang[:, :, :] = 0
                data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang

            if att_type == 4:
                # att img
                level=random.randint(5,11)
                a=0.2*level
                b=0.05*level
                img_att = data_new['imgs_att'][att_t*8 + att_shijiao]
                img_att = bright_contrast(a, b, img_att)
                data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att


                # # att lidar
                # lidar_att_zhedang = data_new['lidars_att'][att_t * 8 + att_shijiao]
                # lidar_att_zhedang = bright_contrast(1.5, 0.5, lidar_att_zhedang)
                # data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang

                # img1=tensor2PIL(lidar_att_zhedang)
                # img2=tensor2PIL(img_att)
                # #print(type(img))
                # plt.figure()
                # plt.subplot(2,1,1)
                # plt.imshow(img1)
                # plt.subplot(2,1,2)
                # plt.imshow(img2)
                # plt.show()

            # if att_type == 5 and self.data_new_last is not None:
            #     if att_t==0:
            #         # att img
            #         img_att_t0 = data_new['imgs_att'][att_t*8 + att_shijiao]
            #         img_att_t1 = self.data_new_last['imgs_att'][(att_t+1)*8 + att_shijiao]
            #         img_att_t0 = (img_att_t0 + img_att_t1)/2
            #         data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att_t0

            #         # att lidar
            #         lidar_att_zhedang_t0 = data_new['lidars_att'][att_t * 8 + att_shijiao]
            #         lidar_att_zhedang_t1 = self.data_new_last['lidars_att'][(att_t+1) * 8 + att_shijiao]
            #         lidar_att_zhedang_t0 = (lidar_att_zhedang_t0 + lidar_att_zhedang_t1)/2
            #         data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang_t0
            #     else:
            #         # att img
            #         img_att_t = data_new['imgs_att'][att_t*8 + att_shijiao]
            #         img_att_t_1 = self.data_new_last['imgs_att'][(att_t-1)*8 + att_shijiao]
            #         img_att_t = (img_att_t + img_att_t_1)/2
            #         data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att_t

            #         # att lidar
            #         lidar_att_zhedang_t = data_new['lidars_att'][att_t * 8 + att_shijiao]
            #         lidar_att_zhedang_t_1 = self.data_new_last['lidars_att'][(att_t-1) * 8 + att_shijiao]
            #         lidar_att_zhedang_t = (lidar_att_zhedang_t + lidar_att_zhedang_t_1)/2
            #         data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang_t


            if att_type == 5:
                # att img
                img_att_t = data_new['imgs_att'][att_t*8 + att_shijiao]
                img_att_t_his = data_new['imgs_his'][att_shijiao]
                img_att_t = (img_att_t + img_att_t_his)/2
                data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att_t

                # att lidar
                lidar_att_zhedang_t = data_new['lidars_att'][att_t * 8 + att_shijiao]
                lidar_att_zhedang_t_his = data_new['lidars_his'][att_shijiao]
                lidar_att_zhedang_t = (lidar_att_zhedang_t + lidar_att_zhedang_t_his)/2
                data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang_t


            if att_type == 6:
                # att img
                img_att_ori = data_new['imgs_clean_ori'][att_t*8 + att_shijiao]
                l_r=np.random.randint(0,7)
                x=np.random.randint(5,11)
                img_att_ori = attack.att_img_semantic(img_att_ori, x, l_r)
                data_new['imgs_att'][att_t * 8 + att_shijiao] = transform_img(Image.fromarray(np.uint8(img_att_ori*255)))

                # att lidar
                lidar_att = data_new['lidars_att'][att_t * 8 + att_shijiao]
                lidar_att = attack.att_lidar_semantic(lidar_att, x, l_r)
                data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att

                # # #print(img_att_ori)
                # img1 = Image.fromarray(np.uint8(img_att_ori*255))
                # #print(type(lidar_att))
                # # img2=Image.fromarray(np.uint8(lidar_att*255))
                # # img1=tensor2PIL(lidar_att)
                # img2=tensor2PIL(lidar_att)
                # #print(type(img))
                # plt.figure()
                # plt.subplot(2,1,1)
                # plt.imshow(img1)
                # plt.subplot(2,1,2)
                # plt.imshow(img2)
                # plt.show()

            if att_type == 7:
                if att_t==0:
                    # att img
                    img_att_t0 = data_new['imgs_att'][att_t*8 + att_shijiao]
                    data_new['imgs_att'][(att_t+1)*8 + att_shijiao] = img_att_t0

                    # att lidar
                    lidar_att_zhedang_t0 = data_new['lidars_att'][att_t * 8 + att_shijiao]
                    data_new['lidars_att'][(att_t+1)*8 + att_shijiao] = lidar_att_zhedang_t0
                else:
                    # att img
                    img_att_t_1 = data_new['imgs_att'][(att_t-1)*8 + att_shijiao]
                    data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att_t_1

                    # att lidar
                    lidar_att_zhedang_t_1 = data_new['lidars_att'][(att_t-1) * 8 + att_shijiao]
                    data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang_t_1

            # if att_type == 8:
            #     # att img
            #     img_att = data_new['imgs_att'][att_t*8 + att_shijiao]
            #     img_att = torch.tensor(self.fan_mat).unsqueeze(0).repeat(3, 1, 1)*img_att
            #     data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att

            #     # att lidar
            #     lidar_att_zhedang = data_new['lidars_att'][att_t * 8 + att_shijiao]
            #     # lidar_att_zhedang = torch.tensor(self.fan_mat).unsqueeze(0).repeat(3, 1, 1)*lidar_att_zhedang
            #     data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang

            if att_type == 8:
                # att img
                img_att = data_new['imgs_att'][att_t*8 + att_shijiao]
                level=np.random.randint(5,11)
                left=np.random.randint(6,128-3*level)
                img_att[:, :, left:left+3*level] = 0
                data_new['imgs_att'][att_t * 8 + att_shijiao] = img_att

                # att lidar
                lidar_att_zhedang = data_new['lidars_att'][att_t * 8 + att_shijiao]
                lidar_att_zhedang[:, :, left-6:left-6+3*level] = 0
                data_new['lidars_att'][att_t * 8 + att_shijiao] = lidar_att_zhedang

                # img1=tensor2PIL(lidar_att_zhedang)
                # img2=tensor2PIL(img_att)
                # #print(type(img))
                # plt.figure()
                # plt.subplot(2,1,1)
                # plt.imshow(img1)
                # plt.subplot(2,1,2)
                # plt.imshow(img2)
                # plt.show()

            if att_type == 9 and att_t>0 and 0<att_shijiao<7:
                att_sensor = np.random.randint(1, 3)
                if att_sensor == 1:
                    # att img
                    data_new['imgs_att'][att_t*8 + att_shijiao] = data_new['imgs_att'][(att_t-1)*8 + att_shijiao]
                    data_new['imgs_att'][att_t*8 + att_shijiao-1] = data_new['imgs_att'][(att_t-1)*8 + att_shijiao-1]
                    data_new['imgs_att'][att_t*8 + att_shijiao+1] = data_new['imgs_att'][(att_t-1)*8 + att_shijiao+1]
                else:
                    # att lidar
                    data_new['lidars_att'][att_t*8 + att_shijiao] = data_new['lidars_att'][(att_t-1)*8 + att_shijiao]
                    data_new['lidars_att'][att_t*8 + att_shijiao-1] = data_new['lidars_att'][(att_t-1)*8 + att_shijiao-1]
                    data_new['lidars_att'][att_t*8 + att_shijiao+1] = data_new['lidars_att'][(att_t-1)*8 + att_shijiao+1]
                break

            if att_type == 10:
                att_sensor = np.random.randint(1, 3)
                if att_sensor == 1:
                    # att img
                    data_new['imgs_att'][16:24] = data_new['imgs_att'][8:16]
                    data_new['imgs_att'][8:16] = data_new['imgs_att'][0:8]
                else:
                    # att lidar
                    data_new['lidars_att'][16:24] = data_new['lidars_att'][8:16]
                    data_new['lidars_att'][8:16] = data_new['lidars_att'][0:8]
                break


        # self.data_new_last = copy.deepcopy(data_new)

        data_new['att_nodes'] = self.choice_nodes

        data_new['red_light'] = []
        red_light = self.red_light[index]
        data_new['red_light'].append(red_light)

        # target_point and waypoints
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]
        if np.isnan(seq_theta[0]):
            seq_theta[0] = 0.

        ego_x = seq_x[0]
        ego_y = seq_y[0]
        ego_theta = seq_theta[0]

        waypoints = []
        for i in range(self.seq_len + self.pred_len):
            # waypoint is the transformed version of the origin in local coordinates
            # we use 90-theta instead of theta
            # LBC code uses 90+theta, but x is to the right and y is downwards here
            local_waypoint = transform_2d_points(np.zeros((1,3)), np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0,:2]))

        data_new['waypoints'] = waypoints

        # print('way shape',len(data_new['waypoints']))

        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]
        ])
        local_command_point = np.array([self.x_command[index] - ego_x, self.y_command[index] - ego_y])
        local_command_point = R.T.dot(local_command_point)
        data_new['target_point'] = tuple(local_command_point)

        # imgs_clean_batch = torch.stack(data_new['imgs_clean'], dim=0)  # 24 * 12 * 3 * 128 *128
        # imgs_clean_batch = imgs_clean_batch.view(-1, 3, 128, 128).numpy().transpose(0,2, 3, 1)
        # img_clean = cv2.cvtColor(imgs_clean_batch[0], cv2.COLOR_BGR2RGB)
        # cv2.imshow('img_clean', img_clean)
        #
        # imgs_att_batch = torch.stack(data_new['imgs_att'], dim=0)  # 24 * 12 * 3 * 128 *128
        # imgs_att_batch = imgs_att_batch.view(-1, 3, 128, 128).numpy().transpose(0, 2, 3, 1)
        # img_att = cv2.cvtColor(imgs_att_batch[0], cv2.COLOR_BGR2RGB)
        # cv2.imshow('img_att', img_att)
        # cv2.waitKey(1000)

        return data_new

    def rand(self):
        randt1 = random.randint(0,2)

        attnode1 = randt1*8
        r1 = random.randint(0,1)
        # print('r1',r1)
        if r1 == 0:
            cleannode1 = ((randt1+1)%3)*8
        else:
            cleannode1 = ((randt1 - 1) % 3) * 8

        other_views = random.sample(range(1, 8), 3)

        att_v2 = other_views[0]
        randt2 = random.randint(0, 2)
        attnode2 = randt2*8 + att_v2
        r2 = random.randint(0, 1)
        if r2 == 0:
            cleannode2 = (attnode2 + 8) % 24
        else:
            cleannode2 = (attnode2 - 8) % 24

        att_v3 = other_views[1]
        randt3 = random.randint(0, 2)
        attnode3 = randt3 * 8 + att_v3
        r3 = random.randint(0, 1)
        if r3 == 0:
            cleannode3 = (attnode3 + 8) % 24
        else:
            cleannode3 = (attnode3 - 8) % 24

        att_v4 = other_views[2]
        randt4 = random.randint(0, 2)
        attnode4 = randt4 * 8 + att_v4
        r4 = random.randint(0, 1)
        if r4 == 0:
            cleannode4 = (attnode4 + 8) % 24
        else:
            cleannode4 = (attnode4 - 8) % 24

        self.choice_nodes = [attnode1, attnode2, attnode3, attnode4, cleannode1, cleannode2, cleannode3, cleannode4]


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T

    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out


def sp_noise(img, prob):
    thres = 1 - prob
    img = img.numpy()
    mask = np.random.uniform(low=0.0, high=1.0, size=img.shape)
    # mask = torch.from_numpy(mask)
    output = np.where(mask>thres, 1, img)
    output = np.where(mask<prob, 0, output)
    output = torch.from_numpy(output)

    return output

def bright_contrast(a, b, img):
    img_out = torch.clip((a * img + b), 0, 1)
    return img_out

def sectoring(angle ,angle2, mat):
    # an intermediate function to calculate angle != 90 and 270
    ans=mat
    start_a=angle/180*math.pi
    tan_s=math.tan(start_a)
    end_a=(angle2)/180*math.pi
    tan_e=math.tan(end_a)
    flag=1  # flag=0 means I&IV ; 1 means II & III; 2 means across
    ######################################
    #Specicial Cases:
    #########################################
    if(angle==90):
        #print(tan_s)
        for j in range(64,128):
            start=0
            end_a=(angle2-90)/180*math.pi
            tan_e=math.tan(end_a)
            end=int((j-64)*tan_e) 
            for i in range(64,64+end):
                ans[i][j]=0
    if(angle2==90):
        for j in range(64,128):
            start=0
            end_a=(90-angle)/180*math.pi
            tan_e=math.tan(end_a)
            end=int((j-64)*tan_e) 
            for i in range(64-end,64):
                ans[i][j]=0
    if(angle==270):
        for j in range(0,64):
            start=0
            end_a=(angle2-270)/180*math.pi
            tan_e=math.tan(end_a)
            end=int((64-j)*tan_e) 
            for i in range(64-end,64):
                ans[i][j]=0
    if(angle2==270):
        for j in range(0,64):
            start=0
            end_a=(270-angle)/180*math.pi
            tan_e=math.tan(end_a)
            end=int((64-j)*tan_e) 
            for i in range(64,64+end):
                ans[i][j]=0
    ##############################################
    # Normal cases
    # ##########################################    
    if (angle2) <90 or angle>270 :
        flag=0
    if (angle2)< 270 and angle >90:
        flag=1
    if flag ==0:
        for i in range(0,64):
            
            start=int(64+(64-i)*tan_s) 
            end=int(64+(64-i)*tan_e) 
            if start>=end: 
                break
            if start>128 :
                start=128
            if end>128:
                end=128
            if start<0: 
                start=0
            if end<0 : 
                end=0
            for j in range(start -1,end):
                ans[i][j]=0
    if flag==1: 
        for i in range(64,128):
            
            start=int(64+(64-i)*tan_s) 
            end=int(64+(64-i)*tan_e) 
            if start>128:
               start=128
            if end>128:
                end=128
            if start<0: 
                start=0
            if end<0 : 
                end=0
            for j in range(end ,start):
                ans[i][j]=0
    
    return ans

def Sector(angle, theta=5):
    # function to get a mask that fans the figure
    if angle > 360 or angle <0:
        print("Choose a right angle")
        return
    mat=np.ones((128,128))
    #flag=1  # flag=0 means I&IV ; 1 means II & III; 2 means across
    if (angle+theta) <90 or angle>=270 :
        return sectoring(angle,angle+theta, mat)
       
    if (angle+ theta)< 270 and angle >=90:
        return sectoring(angle,angle+theta,mat)
    
    if angle<90 :
        ans=sectoring(angle, 90, mat)
        ans=sectoring(90,angle+theta,ans)
        return ans
    if angle >90 and angle<270 : 
        ans=sectoring(angle, 270, mat)
        ans=sectoring(270,angle+theta,ans)
        return ans


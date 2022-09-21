import os
import json
import datetime
import pathlib
import time
import cv2
from collections import deque
import random
import math
from math import *

import torch
import carla
import numpy as np

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner
from leaderboard.envs.sensor_interface import SensorInterface

from torchvision import transforms
from PIL import Image

from model_net.model_net import model_all

SAVE_PATH = os.environ.get('SAVE_PATH', None)

device = 'cuda'


class ATTACK():

    def __init__(self):
        self.fore_image = np.array(Image.open('./others/car_att.png'))

    def att_img_semantic(self, base_image, x, l_r):
        fore_image = cv2.resize(self.fore_image, (200, 209))
        base_image = base_image.copy()
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
        # if l_r == 2:
        #     left_right = int(left_right_front - (left_right_front - left_right_l_r)/3)
        # if l_r == 1:
        #     left_right = int(left_right_front - 2*(left_right_front - left_right_l_r)/3)
        # if l_r == 0:
        #     left_right = int(left_right_l_r)
        # if l_r == 4:
        #     left_right = int(left_right_front + (left_right_front - left_right_l_r)/3)
        # if l_r == 5:
        #     left_right = int(left_right_front + 2*(left_right_front - left_right_l_r)/3)
        # if l_r == 6:
        #     left_right = int(left_right_front + 3*(left_right_front - left_right_l_r)/3)
        x = a[0] * resize
        y = a[1] * resize

        # if up_down + int(x) >= 300 and left_right <= 0:
        #     base_part = base_image[up_down:, :left_right + int(y), :]
        #     fore_image = cv2.resize(fore_image, (int(y), int(x)))
        # elif up_down + int(x) >= 300 and left_right + int(y) >= 400:
        #     base_part = base_image[up_down:, left_right:, :]
        #     fore_image = cv2.resize(fore_image, (int(y), int(x)))
        # elif up_down + int(x) >= 300 and 0 < left_right and left_right + int(y) < 400:
        #     base_part = base_image[up_down:, left_right:left_right + int(y), :]
        #     fore_image = cv2.resize(fore_image, (int(y), int(x)))
        # elif left_right <= 0 and up_down + int(x) < 300:
        #     base_part = base_image[up_down:, :left_right + int(y), :]
        #     fore_image = cv2.resize(fore_image, (int(y), int(x)))
        # elif left_right + int(y) >= 400 and up_down + int(x) < 300:
        #     base_part = base_image[up_down:up_down + int(x), left_right:, :]
        #     fore_image = cv2.resize(fore_image, (int(y), int(x)))
        # else:
        #     base_part = base_image[up_down:up_down + int(x), left_right:left_right + int(y), :]
        #     fore_image = cv2.resize(fore_image, (base_part.shape[1], base_part.shape[0]))

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
        # if up_down + int(x) >= 300 and 0 < left_right and left_right + int(y) < 400:
        #     base_part = cv2.copyMakeBorder(base_part, 0, int(up_down + int(x) - 300), 0, 0,
        #                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # elif up_down + int(x) >= 300 and left_right <= 0:
        #     base_part = cv2.copyMakeBorder(base_part, 0, int(up_down + int(x) - 300), int(-left_right), 0,
        #                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # elif up_down + int(x) >= 300 and left_right + int(y) >= 400:
        #     base_part = cv2.copyMakeBorder(base_part, 0, int(up_down + int(x) - 300), 0, int(left_right + int(y) - 400),
        #                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # elif left_right <= 0 and up_down + int(x) < 300:
        #     base_part = cv2.copyMakeBorder(base_part, 0, 0, int(-left_right), 0,
        #                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # elif left_right + int(y) >= 400 and up_down + int(x) < 300:
        #     base_part = cv2.copyMakeBorder(base_part, 0, 0, 0, int(left_right + int(y) - 400),
        #                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])

        res_image = res_ + np.multiply((1 - scope_map), base_part)
        # if up_down + int(x) >= 300 and 0 < left_right and left_right + int(y) < 400:
        #     base_image[up_down:, left_right:left_right + int(y)] = res_image[:300 - up_down, :]
        # elif up_down + int(x) >= 300 and left_right <= 0:
        #     base_image[up_down:, :left_right + int(y)] = res_image[:300 - up_down, int(-left_right):]
        # elif up_down + int(x) >= 300 and left_right + int(y) >= 400:
        #     base_image[up_down:, left_right:] = res_image[:300 - up_down, :400 - left_right]
        # elif left_right <= 0 and up_down + int(x) < 300:
        #     base_image[up_down:up_down + int(x), :left_right + int(y)] = res_image[:, int(-left_right):]
        # elif left_right + int(y) >= 400 and up_down + int(x) < 300:
        #     base_image[up_down:up_down + int(x), left_right:] = res_image[:, :400 - left_right]
        # else:
        #     base_image[up_down:up_down + int(x), left_right:left_right + int(y)] = res_image

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
                start=127
            if end>128:
                end=127
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
               start=127
            if end>128:
                end=127
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

def get_entry_point():
    return 'EightAngleAgent'


class EightAngleAgent(autonomous_agent.AutonomousAgent):

    def to_Spherical(self,xyz):
        ptsnew = np.zeros(xyz.shape)
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
        # ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        ptsnew[:, 1] = np.arctan2(xyz[:, 2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
        ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
        return ptsnew

    def lidar_to_2D_why(self, lidar):
        lidar_p = self.to_Spherical(lidar)
        lidar_2D = np.zeros([64, 450])
        for point in lidar_p:
            x_thea = int((point[2] + np.pi) * (180 / np.pi) * 1.25) - 1
            y_fai = 64 - int((point[1] + (np.pi / 6)) * (180 / np.pi) * 1.6) - 1
            lidar_2D[y_fai, x_thea] = min(point[0] / 60, 1)

        return lidar_2D

    def lidar_no_noise(self,lidar, zhi=1):
        for i in range(64):
            for j in range(122):
                if lidar[i, j] == zhi:
                    if j < 112:
                        for t in range(5):
                            if lidar[i, j + t + 1] != zhi:
                                lidar[i, j] = lidar[i, j + t + 1]
                                break
                            lidar[i, j] = 0
                    else:
                        for t in range(5):
                            if lidar[i, j - t - 1] != zhi:
                                lidar[i, j] = lidar[i, j - t - 1]
                                break
                            lidar[i, j] = 0
        return lidar

    def __init__(self, path_to_conf_file):
        # super().__init__()
        self.data__ = {}
        self.canshu  = 0
        self.alist = ['this_moment', 'last_moment', 'last_last_moment']
        self.initialized = False
        self.sensor_interface = SensorInterface()
        self.track = autonomous_agent.Track.SENSORS
        self.wallclock_t0 = 0
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.input_buffer = {'rgb_front': deque(), 'rgb_left': deque(), 'rgb_right': deque(),
                             'rgb_rear': deque(), 'rgb_rear_left': deque(), 'rgb_rear_right': deque(),
                             'rgb_front_left': deque(), 'rgb_front_right': deque(), 'lidar_qvzao_front': deque(),
                             'lidar_qvzao_right': deque(), 'lidar_qvzao_left': deque(), 'lidar_qvzao_rear': deque(),
                             'lidar_qvzao_front_left': deque(), 'lidar_qvzao_front_right': deque(),
                             'lidar_qvzao_rear_left': deque(), 'lidar_qvzao_rear_right': deque()}

        self.A_same_t = np.array([[1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0, 1],
                         [0, 0, 0, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 1, 1]])

        self.A_8 = np.zeros([24, 24])

        self.A_8[0:8, 0:8] = self.A_same_t
        self.A_8[8:16, 8:16] = self.A_same_t
        self.A_8[16:24, 16:24] = self.A_same_t
        self.A_8[0:8, 8:16] = np.eye(8)
        self.A_8[8:16, 0:8] = np.eye(8)
        self.A_8[8:16, 16:24] = np.eye(8)
        self.A_8[16:24, 8:16] = np.eye(8)

        self.A = self.A_8
        self.A = torch.tensor(self.A).to('cuda')

        self.model_all = model_all(train_model=3).to(device)
        model_all.load_state_dict(torch.load('./trained_models_/'))

        self.transform_img = transforms.Compose([transforms.Resize([128, 128]),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0., 0., 0.), (1., 1., 1.))])

        self.transform_lidar = transforms.Compose([transforms.ToTensor()])

        self.save_path = None
        if SAVE_PATH is not None:
            print('SAVE_PATH:',SAVE_PATH)
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print(string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'meta').mkdir()

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_front'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_left'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_right'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_rear'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_front_left'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_front_right'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 135.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_rear_right'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 225.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_rear_left'
            },
            {
                'type': 'sensor.lidar.ray_cast',
                'x': 0.0, 'y': 0.0, 'z': 3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'id': 'lidar'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            }
        ]

    def tick(self, input_data):
        self.step += 1

        rgb_front = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_front_left = cv2.cvtColor(input_data['rgb_front_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear_left = cv2.cvtColor(input_data['rgb_rear_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_front_right = cv2.cvtColor(input_data['rgb_front_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear_right = cv2.cvtColor(input_data['rgb_rear_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        lidar = input_data['lidar'][1][:, :3]
        lidar_why = self.lidar_to_2D_why(lidar)

        lidar_img_forward_why = lidar_why[:, 275:397]
        lidar_img_left_why = lidar_why[:, 163:285]
        lidar_img_rear_why = lidar_why[:, 51:173]
        lidar_img_right_why = np.zeros((64, 122))
        lidar_img_right_why[:, 63:122] = lidar_why[:, :59]
        lidar_img_right_why[:, 0:63] = lidar_why[:, 387:450]

        lidar_img_forward_left_why = lidar_why[:, 219:341]
        lidar_img_rear_left_why = np.zeros((64, 122))
        lidar_img_rear_left_why[:, 5:122] = lidar_why[:, :117]
        lidar_img_rear_left_why[:, 0:5] = lidar_why[:, 445:450]
        lidar_img_rear_right_why = lidar_why[:, 107:229]
        lidar_img_forward_right_why = np.zeros((64, 122))
        lidar_img_forward_right_why[:, 119:122] = lidar_why[:, :3]
        lidar_img_forward_right_why[:, 0:119] = lidar_why[:, 331:450]

        lidar_qvzao_forward_left = lidar_img_forward_left_why.copy()
        lidar_qvzao_forward_left = 1 - lidar_qvzao_forward_left
        lidar_qvzao_forward_left = self.lidar_no_noise(lidar_qvzao_forward_left, 1)

        lidar_qvzao_forward_right = lidar_img_forward_right_why.copy()
        lidar_qvzao_forward_right = 1 - lidar_qvzao_forward_right
        lidar_qvzao_forward_right = self.lidar_no_noise(lidar_qvzao_forward_right, 1)

        lidar_qvzao_rear_left = lidar_img_rear_left_why.copy()
        lidar_qvzao_rear_left = 1 - lidar_qvzao_rear_left
        lidar_qvzao_rear_left = self.lidar_no_noise(lidar_qvzao_rear_left, 1)

        lidar_qvzao_rear_right = lidar_img_rear_right_why.copy()
        lidar_qvzao_rear_right = 1 - lidar_qvzao_rear_right
        lidar_qvzao_rear_right = self.lidar_no_noise(lidar_qvzao_rear_right, 1)

        lidar_qvzao_rear = lidar_img_rear_why.copy()
        lidar_qvzao_rear = 1 - lidar_qvzao_rear
        lidar_qvzao_rear = self.lidar_no_noise(lidar_qvzao_rear, 1)

        lidar_qvzao_front = lidar_img_forward_why.copy()
        lidar_qvzao_front = 1 - lidar_qvzao_front
        lidar_qvzao_front = self.lidar_no_noise(lidar_qvzao_front, 1)

        lidar_qvzao_left = lidar_img_left_why.copy()
        lidar_qvzao_left = 1 - lidar_qvzao_left
        lidar_qvzao_left = self.lidar_no_noise(lidar_qvzao_left, 1)

        lidar_qvzao_right = lidar_img_right_why.copy()
        lidar_qvzao_right = 1 - lidar_qvzao_right
        lidar_qvzao_right = self.lidar_no_noise(lidar_qvzao_right, 1)

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        result = {
            'rgb_front': rgb_front,
            'rgb_left': rgb_left,
            'rgb_right': rgb_right,
            'rgb_rear': rgb_rear,

            'rgb_front_right': rgb_front_right,
            'rgb_front_left': rgb_front_left,
            'rgb_rear_right': rgb_rear_right,
            'rgb_rear_left': rgb_rear_left,

            'lidar_qvzao_front':  lidar_qvzao_front,
            'lidar_qvzao_left':  lidar_qvzao_left,
            'lidar_qvzao_right': lidar_qvzao_right,
            'lidar_qvzao_rear': lidar_qvzao_rear,

            'lidar_qvzao_forward_right': lidar_qvzao_forward_right,
            'lidar_qvzao_forward_left': lidar_qvzao_forward_left,
            'lidar_qvzao_rear_right': lidar_qvzao_rear_right,
            'lidar_qvzao_rear_left':  lidar_qvzao_rear_left,

            'gps': gps,
            'speed': speed,
            'compass': compass,
        }
        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value

        theta = compass + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.canshu += 1
        if self.canshu >= 4:
            self.data__["last_last_moment"] = self.data__["last_moment"]
            self.data__["last_moment"] = self.data__["this_moment"]
            self.data__["this_moment"] = input_data
        if self.canshu == 1:
            self.data__["last_last_moment"] = input_data
        if self.canshu == 2:
            self.data__["last_moment"] = input_data
        if self.canshu == 3:
            self.data__["this_moment"] = input_data
        if self.canshu >= 3:
            tick_data_now = self.tick(self.data__["this_moment"])
            tick_data_last = self.tick(self.data__["last_moment"])
            tick_data_last_last = self.tick(self.data__["last_last_moment"])

        if self.canshu < 3:

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0

            return control

        data_new = dict()
        data_new['imgs_clean'] = []
        data_new['lidars_clean'] = []
        data_new['imgs_clean_ori'] = []

        tick_data_ti = [tick_data_last_last, tick_data_last, tick_data_now]

        for i in range(3):
            data_new['imgs_clean'].append(self.transform_img(Image.fromarray(tick_data_ti[i]['rgb_front'])))
            data_new['imgs_clean'].append(self.transform_img(Image.fromarray(tick_data_ti[i]['rgb_front_left'])))
            data_new['imgs_clean'].append(self.transform_img(Image.fromarray(tick_data_ti[i]['rgb_left'])))
            data_new['imgs_clean'].append(self.transform_img(Image.fromarray(tick_data_ti[i]['rgb_rear_left'])))
            data_new['imgs_clean'].append(self.transform_img(Image.fromarray(tick_data_ti[i]['rgb_rear'])))
            data_new['imgs_clean'].append(self.transform_img(Image.fromarray(tick_data_ti[i]['rgb_rear_right'])))
            data_new['imgs_clean'].append(self.transform_img(Image.fromarray(tick_data_ti[i]['rgb_right'])))
            data_new['imgs_clean'].append(self.transform_img(Image.fromarray(tick_data_ti[i]['rgb_front_right'])))

            data_new['imgs_clean_ori'].append(np.array(tick_data_ti[i]['rgb_front']))
            data_new['imgs_clean_ori'].append(np.array(tick_data_ti[i]['rgb_front_left']))
            data_new['imgs_clean_ori'].append(np.array(tick_data_ti[i]['rgb_left']))
            data_new['imgs_clean_ori'].append(np.array(tick_data_ti[i]['rgb_rear_left']))
            data_new['imgs_clean_ori'].append(np.array(tick_data_ti[i]['rgb_rear']))
            data_new['imgs_clean_ori'].append(np.array(tick_data_ti[i]['rgb_rear_right']))
            data_new['imgs_clean_ori'].append(np.array(tick_data_ti[i]['rgb_right']))
            data_new['imgs_clean_ori'].append(np.array(tick_data_ti[i]['rgb_front_right']))

            data_new['lidars_clean'].append(self.transform_lidar(tick_data_ti[i]['lidar_qvzao_front']))
            data_new['lidars_clean'].append(self.transform_lidar(tick_data_ti[i]['lidar_qvzao_forward_left']))
            data_new['lidars_clean'].append(self.transform_lidar(tick_data_ti[i]['lidar_qvzao_left']))
            data_new['lidars_clean'].append(self.transform_lidar(tick_data_ti[i]['lidar_qvzao_rear_left']))
            data_new['lidars_clean'].append(self.transform_lidar(tick_data_ti[i]['lidar_qvzao_rear']))
            data_new['lidars_clean'].append(self.transform_lidar(tick_data_ti[i]['lidar_qvzao_rear_right']))
            data_new['lidars_clean'].append(self.transform_lidar(tick_data_ti[i]['lidar_qvzao_right']))
            data_new['lidars_clean'].append(self.transform_lidar(tick_data_ti[i]['lidar_qvzao_forward_right']))

        att_nodes = [0, 8, 16]
        for att_node in att_nodes:
            att_t = att_node // 8
            att_shijiao = att_node % 8
            # 1:Shelter; 2:noise; 3:Information all black(Loss of frames); 4:Brightness and contrast; 5:Frame overlap; 6:Semantic attacks
            # 7:Repeat frame; 8:Fan lost(5 degree); 9:Multi-sensor desynchronization (homogeneous desynchronization); 
            # 10:Multi-sensor desynchronization (heterogeneous desynchronization)
            att_type = np.random.randint(1, 7)
            # att_type = 1
            if att_type == 1:
                img_att_zhedang = data_new['imgs_clean'][att_t*8 + att_shijiao]
                img_att_zhedang[:, 32:, 32:96] = 0

                # img_att_zhedang = img_att_zhedang.cpu().numpy().transpose(1, 2, 0)
                # cv2.imshow('img_orl', img_att_zhedang)

                data_new['imgs_clean'][att_t * 8 + att_shijiao] = img_att_zhedang

                # att lidar
                lidar_att_zhedang = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                lidar_att_zhedang[:, 16:, 30:90] = 0

                # img_att_zhedang = lidar_att_zhedang.cpu().numpy().transpose(1, 2, 0)
                # cv2.imshow('lidar_orl', img_att_zhedang)
                # cv2.waitKey(10)
                # time.sleep(6)

                data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att_zhedang


            if att_type == 2:
                Attack_intensity = random.uniform(0.35, 0.45)
                # att img
                img_att_zhedang = data_new['imgs_clean'][att_t*8 + att_shijiao]
                img_att_zhedang = sp_noise(img_att_zhedang, Attack_intensity)
                data_new['imgs_clean'][att_t * 8 + att_shijiao] = img_att_zhedang

                # att lidar
                lidar_att_zhedang = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                lidar_att_zhedang = sp_noise(lidar_att_zhedang, Attack_intensity)
                data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att_zhedang


            if att_type == 3:
                img_att_zhedang = data_new['imgs_clean'][att_t*8 + att_shijiao]
                img_att_zhedang[:, :, :] = 0
                data_new['imgs_clean'][att_t * 8 + att_shijiao] = img_att_zhedang

                # att lidar
                lidar_att_zhedang = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                lidar_att_zhedang[:, :, :] = 0
                data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att_zhedang


            if att_type == 4:
                # att img
                img_att_zhedang = data_new['imgs_clean'][att_t*8 + att_shijiao]
                img_att_zhedang = bright_contrast(1.5, 0.5, img_att_zhedang)
                data_new['imgs_clean'][att_t * 8 + att_shijiao] = img_att_zhedang

                # att lidar
                lidar_att_zhedang = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                lidar_att_zhedang = bright_contrast(1.5, 0.5, lidar_att_zhedang)
                data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att_zhedang


            if att_type == 5:
                if att_t==0:
                    # att img
                    img_att_zhedang_t0 = data_new['imgs_clean'][att_t*8 + att_shijiao]
                    img_att_zhedang_t1 = data_new['imgs_att'][(att_t+1)*8 + att_shijiao]
                    img_att_zhedang_t0 = (img_att_zhedang_t0 + img_att_zhedang_t1)/2
                    data_new['imgs_clean'][att_t * 8 + att_shijiao] = img_att_zhedang_t0

                    # att lidar
                    lidar_att_zhedang_t0 = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                    lidar_att_zhedang_t1 = data_new['lidars_clean'][(att_t+1) * 8 + att_shijiao]
                    lidar_att_zhedang_t0 = (lidar_att_zhedang_t0 + lidar_att_zhedang_t1)/2
                    data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att_zhedang_t0
                else:
                    # att img
                    img_att_zhedang_t = data_new['imgs_clean'][att_t*8 + att_shijiao]
                    img_att_zhedang_t_1 = data_new['imgs_clean'][(att_t-1)*8 + att_shijiao]
                    img_att_zhedang_t = (img_att_zhedang_t + img_att_zhedang_t_1)/2
                    data_new['imgs_clean'][att_t * 8 + att_shijiao] = img_att_zhedang_t

                    # att lidar
                    lidar_att_zhedang_t = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                    lidar_att_zhedang_t_1 = data_new['lidars_clean'][(att_t-1) * 8 + att_shijiao]
                    lidar_att_zhedang_t = (lidar_att_zhedang_t + lidar_att_zhedang_t_1)/2
                    data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att_zhedang_t


            if att_type == 6:
                # att img
                img_att_ori = data_new['imgs_clean_ori'][att_t*8 + att_shijiao]
                img_att_ori = attack.att_img_semantic(img_att_ori, 8, 3)
                data_new['imgs_clean'][att_t * 8 + att_shijiao] = self.transform_img(Image.fromarray(np.uint8(img_att_ori*255)))

                # att lidar
                lidar_att = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                lidar_att = attack.att_lidar_semantic(lidar_att, 8, 3)
                data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att


            if att_type == 7:
                if att_t==0:
                    # att img
                    img_att_zhedang_t0 = data_new['imgs_clean'][att_t*8 + att_shijiao]
                    data_new['imgs_clean'][(att_t+1)*8 + att_shijiao] = img_att_zhedang_t0

                    # att lidar
                    lidar_att_zhedang_t0 = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                    data_new['lidars_clean'][(att_t+1)*8 + att_shijiao] = lidar_att_zhedang_t0
                else:
                    # att img
                    img_att_zhedang_t_1 = data_new['imgs_clean'][(att_t-1)*8 + att_shijiao]
                    data_new['imgs_clean'][att_t * 8 + att_shijiao] = img_att_zhedang_t_1

                    # att lidar
                    lidar_att_zhedang_t_1 = data_new['lidars_clean'][(att_t-1) * 8 + att_shijiao]
                    data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att_zhedang_t_1

            if att_type == 8:
                # att img
                img_att_zhedang = data_new['imgs_clean'][att_t*8 + att_shijiao]
                img_att_zhedang = torch.tensor(self.fan_mat).unsqueeze(0).repeat(3, 1, 1)*img_att_zhedang
                data_new['imgs_clean'][att_t * 8 + att_shijiao] = img_att_zhedang

                # att lidar
                lidar_att_zhedang = data_new['lidars_clean'][att_t * 8 + att_shijiao]
                lidar_att_zhedang = torch.tensor(self.fan_mat).unsqueeze(0).repeat(3, 1, 1)*lidar_att_zhedang
                data_new['lidars_clean'][att_t * 8 + att_shijiao] = lidar_att_zhedang

        gt_velocity = torch.FloatTensor([tick_data_now['speed']]).to('cpu', dtype=torch.float32)

        tick_data_now['target_point'] = [torch.FloatTensor([tick_data_now['target_point'][0]]),
                                     torch.FloatTensor([tick_data_now['target_point'][1]])]
        data_new['target_point'] = tick_data_now['target_point']

        _, pred_wp = model_all(data_new, self.A)

        steer_, throttle_, brake_, metadata_ = self.model_all.model_nav.control_pid(pred_wp, gt_velocity)

        if brake_ < 0.05: brake_ = 0.0
        if throttle_ > brake_: brake_ = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer_)
        control.throttle = float(throttle_)
        control.brake = float(brake_)

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data_now)

        return control

    def save(self, tick_data_now):
        frame = self.step // 10

    def destroy(self):
        del self.model_all

        
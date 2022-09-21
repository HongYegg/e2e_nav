import torch
import cv2
import math

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from collections import deque
from torch import nn


device = 'cuda:0'

nav_shijiao = np.array([0, 1, 7, 8, 9, 15, 16, 17, 23])


def prox_nuclear(data, alpha):
    """Proximal operator for nuclear norm (trace norm).
    """
    data = data.cpu()
    # print(data)
    # data[np.isnan(data)] = 0
    U, S, V = np.linalg.svd(data)  # (36, 24, 24); (36, 24); (36, 24, 24)
    U, S, V = torch.FloatTensor(U), torch.FloatTensor(S), torch.FloatTensor(V)
    diag_S = torch.diag_embed(torch.clamp(S - alpha, min=0))
    return torch.bmm(torch.bmm(U, diag_S), V)


def feature_smoothing_batch(adj):
    
    rowsum = adj.sum(2)
    r_inv = torch.flatten(rowsum, 1,-1)
    D = torch.diag_embed(r_inv)
    L = D - adj
    r_inv = r_inv + 1e-3
    r_inv = r_inv.pow(-1 / 2)
    r_inv[torch.isinf(r_inv)] = 0.001
    r_mat_inv = torch.diag_embed(r_inv)
    L = torch.bmm(torch.bmm(r_mat_inv, L), r_mat_inv)

    return L


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        # self.linear2 = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Encoder1(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder1, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder1(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder1, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder, encoder1, decoder1,):
        super(VAE, self).__init__()
        self.encoder_img = encoder
        self.decoder_img = decoder
        self.encoder_lidar = encoder1
        self.decoder_lidar = decoder1
        self.fusion = torch.nn.Linear(512, 256)
        self._enc_mu = torch.nn.Linear(256, 256)
        self._enc_log_sigma = torch.nn.Linear(256, 256)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(device)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, img, lidar):
        x_img = self.encoder_img(img)
        x_lidar = self.encoder_lidar(lidar)
        # print(x_img.shape, x_lidar.shape)
        x_img_lidar = torch.cat((x_img, x_lidar), 1)
        # print(x_img_lidar.shape)
        h_enc = self.fusion(x_img_lidar)
        z = self._sample_latent(h_enc)
        return self.z_mean, self.decoder_img(self.z_mean), self.decoder_lidar(self.z_mean)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


class ADJ_rec(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, mid_features1, hidden, mid_feature2, out_features, dropout, alpha, concat=True):
        super(ADJ_rec, self).__init__()
        self.hidden = hidden  # 输出维度
        self.dropout = dropout
        self.alpha = alpha

        # gnn
        self.fcv = nn.Linear(in_features, mid_features1)
        nn.init.xavier_uniform_(self.fcv.weight)  # 原来注释掉了
        self.fck = nn.Linear(in_features, mid_features1)
        nn.init.xavier_uniform_(self.fck.weight)  # 原来注释掉了
        self.fcq = nn.Linear(in_features, mid_features1)
        nn.init.xavier_normal_(self.fcq.weight)  # 原来注释掉了
        self.fcout1 = nn.Linear(mid_features1, hidden)

        self.fcv2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_uniform_(self.fcv2.weight)  # 原来注释掉了
        self.fck2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_uniform_(self.fck2.weight)  # 原来注释掉了
        self.fcq2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_normal_(self.fcq2.weight)  # 原来注释掉了
        self.fcout2 = nn.Linear(mid_feature2, out_features)

        self.finalMLP = nn.Linear(out_features, out_features)

    def forward(self, h, adj):
        Wh1 = F.relu(self.fcv(h))
        q1 = F.relu(self.fcq(h))
        k1 = F.relu(self.fck(h)).permute(0, 2, 1)
        # att1 = F.softmax(torch.mul(torch.bmm(q1, k1), adj) - 9e15 * (1 - adj), dim=2).to(torch.float32)
        att1 = torch.sigmoid(torch.mul(torch.bmm(q1, k1), adj) - 9e15 * (1 - adj)).to(torch.float32)
        f1 = torch.bmm(att1, Wh1)
        f1 = self.fcout1(f1)

        Wh2 = F.relu(self.fcv2(f1))
        q2 = F.relu(self.fcq2(f1))
        k2 = F.relu(self.fck2(f1)).permute(0, 2, 1)
        # att2 = F.softmax(torch.mul(torch.bmm(q2, k2), adj) - 9e15 * (1 - adj), dim=2).to(torch.float32)
        att2 = torch.sigmoid(torch.mul(torch.bmm(q2, k2), adj) - 9e15 * (1 - adj)).to(torch.float32)
        f2 = torch.bmm(att2, Wh2)
        f2 = self.fcout2(f2)
        out = F.softmax(self.finalMLP(f2), dim=2)
        return out, att2


class FEATURE_rec(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, mid_features1, hidden, mid_feature2,out_features, dropout, alpha, concat=True):
        super(FEATURE_rec, self).__init__()
        self.hidden = hidden  # 输出维度
        self.dropout = dropout
        self.alpha = alpha

        # gnn
        self.fcv = nn.Linear(in_features, mid_features1)
        nn.init.xavier_uniform_(self.fcv.weight)  # 原来注释掉了
        self.fck = nn.Linear(in_features, mid_features1)
        nn.init.xavier_uniform_(self.fck.weight)  # 原来注释掉了
        self.fcq = nn.Linear(in_features, mid_features1)
        nn.init.xavier_normal_(self.fcq.weight)  # 原来注释掉了
        self.fcout1 = nn.Linear(mid_features1, hidden)

        self.fcv2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_uniform_(self.fcv2.weight)  # 原来注释掉了
        self.fck2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_uniform_(self.fck2.weight)  # 原来注释掉了
        self.fcq2 = nn.Linear(hidden, mid_feature2)
        nn.init.xavier_normal_(self.fcq2.weight)  # 原来注释掉了
        self.fcout2 = nn.Linear(mid_feature2, out_features)

        self.finalMLP = nn.Linear(out_features*2, out_features)

    def forward(self, h, adj, vae2_fetures):
        Wh1 = F.relu(self.fcv(h))
        q1 = F.relu(self.fcq(h))
        k1 = F.relu(self.fck(h)).permute(0, 2, 1)
        att1 = F.softmax(torch.mul(torch.bmm(q1, k1), adj) - 9e15 * (1 - adj), dim=2).to(torch.float32)
        f1 = torch.bmm(att1, Wh1)
        f1 = self.fcout1(f1)

        Wh2 = F.relu(self.fcv2(f1))
        q2 = F.relu(self.fcq2(f1))
        k2 = F.relu(self.fck2(f1)).permute(0, 2, 1)
        att2 = F.softmax(torch.mul(torch.bmm(q2, k2), adj) - 9e15 * (1 - adj), dim=2).to(torch.float32)
        f2 = torch.bmm(att2, Wh2)
        f2 = self.fcout2(f2)
        f3 = torch.cat([f2, vae2_fetures], 2)
        out = self.finalMLP(f3)
        return out


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class LSTMModule(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size, dropout=0, device=torch.device(device)):
        super(LSTMModule, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True).to(device)

    def initHidden(self, batch_size):
        self.hidden_cell = (
            torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device),
            torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
        )

    def forward(self, input_seq):
        self.initHidden(input_seq.shape[0])
        self.out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        return self.out, self.hidden_cell


class VaeNav(nn.Module):
    '''
    Image + LiDAR network with waypoint output and pid controller
    '''

    def __init__(self, config, device, fusion_net=None):
        super().__init__()
        self.config = config
        self.device = device

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.model_rec = fusion_net

        self.join = nn.Sequential(
                            nn.Linear(768, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.sequential_decoder = LSTMModule(num_layers=1, hidden_size=512, input_size=256, device=torch.device(device))

        self.join2 = nn.Sequential(nn.Linear(512, 256),
                               nn.ReLU(inplace=True),
                               nn.Linear(256, 64),
                               nn.ReLU(inplace=True),
                               ).to(self.device)

        self.decoder = nn.GRUCell(input_size=5, hidden_size=64).to(self.device)
        self.output = nn.Linear(64, 2).to(self.device)

    def forward(self, feature_emb, target_point, red_light):
        '''
        Predicts future waypoints from image features and target point (goal location)
        Args:
            feature_emb (list): list of feature tensors
            target_point (tensor): goal location registered to ego-frame
        '''
        f1 = [feature_emb[:, i, :] for i in range(3)]
        f1 = torch.cat(f1, 1)
        z1 = self.join(f1)

        f2 = [feature_emb[:, i, :] for i in range(3, 6)]
        f2=torch.cat(f2, 1)
        z2 = self.join(f2)

        f3 = [feature_emb[:, i, :] for i in range(6, 9)]
        f3 = torch.cat(f3, 1)
        z3 = self.join(f3)

        zs = torch.stack([z3, z2, z1], 1)

        _, hidden_state = self.sequential_decoder(zs)
        hidden_state = hidden_state[0][0]

        z = self.join2(hidden_state)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        red_light = red_light.unsqueeze(1)
        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            x_in = torch.cat([x_in, red_light], dim=1)
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp

    def control_pid(self, waypoints, velocity):
        '''
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        aim = (waypoints[1] + waypoints[0] + waypoints[2]*0.5+ waypoints[3]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)*6
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata


class GlobalConfig():
    """ base architecture configurations """

    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.4  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25  # maximum change in speed input to logitudinal controller


encoder = Encoder(128*128*3, 512, 256)
decoder = Decoder(256, 512, 128*128*3)
encoder1 = Encoder1(64*122, 512, 256)
decoder1 = Decoder1(256, 512, 64*122)
vae = VAE(encoder, decoder, encoder1, decoder1)
vae_fixed_parameters = VAE(encoder, decoder, encoder1, decoder1)
vae_fixed_parameters.load_state_dict(torch.load('./Pre-trained_models/', map_location=device))

adj_rec = ADJ_rec(in_features=256, mid_features1=512, hidden=256, mid_feature2=32,
                  out_features=2, dropout=0, alpha=0.2, concat=False)

feature_rec = FEATURE_rec(in_features=256, mid_features1=1024, hidden=2048, mid_feature2=1024,
                          out_features=256, dropout=0, alpha=0.2, concat=False)

config_nav = GlobalConfig()
model_nav = VaeNav(config_nav, device)


class model_all(nn.Module):

    def __init__(self, train_model = 0): # 0:all; 1:model_adj_rec; 2:model_feature_rec;
        super(model_all, self).__init__()
        self.train_model = train_model
        # self.model_vae_fixed_parameters = vae_fixed_parameters
        self.model_vae = vae
        self.model_adj_rec = adj_rec
        self.model_feature_rec = feature_rec
        self.model_nav = model_nav

    def forward(self, data, A):

        if self.train_model==0:
            imgs_clean_batch = torch.stack(data['imgs_clean'], dim=0)  # 24 * 12 * 3 * 128 *128
            self.imgs_clean_batch = imgs_clean_batch.view(-1, 3 * 128 * 128).to(device)
            lidars_clean_batch = torch.stack(data['lidars_clean'], dim=0)  # 24 * 12 * 3 * 128 *128
            self.lidars_clean_batch = lidars_clean_batch.view(-1, 64 * 122).to(device)
            vae_feature_label, self.dec_clean_imgs, self.dec_clean_lidars = vae_fixed_parameters(self.imgs_clean_batch, self.lidars_clean_batch)
            self.ll_clean = latent_loss(vae_fixed_parameters.z_mean, vae_fixed_parameters.z_sigma)
            self.vae_feature_label = vae_feature_label.view(24, -1, 256).permute(1, 0, 2)

            imgs_att_batch = torch.stack(data['imgs_att'], dim=0)  # 24 * 12 * 3 * 128 *128
            imgs_att_batch = imgs_att_batch.view(-1, 3 * 128 * 128).to(device)
            lidars_att_batch = torch.stack(data['lidars_att'], dim=0)  # 24 * 12 * 3 * 128 *128
            lidars_att_batch = lidars_att_batch.view(-1, 64 * 122).to(device)

            vae_feature_att, _, _ = self.model_vae(imgs_att_batch, lidars_att_batch)
            vae_feature_att = vae_feature_att.view(24, -1, 256).permute(1, 0, 2)

            self.batch = vae_feature_att.shape[0]

            self.node_class, self.adj_rec = self.model_adj_rec(vae_feature_att, A)

            vae_feature_att0 = vae_feature_att.clone()

            vae_feature_att0[:, 0:8, :] /= math.sqrt(3)
            vae_feature_att0[:, 8:16, :] /= 2
            vae_feature_att0[:, 16:24, :] /= math.sqrt(3)

            vae_feature_att1 = vae_feature_att0.unsqueeze(2)
            vae_feature_att2 = vae_feature_att0.unsqueeze(1)
            vae_feature_att3 = vae_feature_att2 - vae_feature_att1
            vae_feature_att4 = torch.norm(vae_feature_att3, p=2, dim=3)

            S_adj_pro = self.adj_rec - vae_feature_att4 * 0.01 / 2
            S_adj = prox_nuclear(S_adj_pro.detach(), 1).to(device)
            b_s = S_adj.shape[0]
            S_adj += torch.eye(24).unsqueeze(0).repeat(b_s, 1, 1).to(device)

            S_adj[S_adj < 0] = 0
            S_adj[S_adj > 1] = 1

            features_rec = self.model_feature_rec(vae_feature_att, S_adj, vae_feature_att)

            L = feature_smoothing_batch(S_adj)
            coeffient = torch.eye(24).unsqueeze(0).repeat(self.batch, 1, 1).to(device) + 2.5 * L
            coeffient = torch.linalg.inv(coeffient)
            features_rec_step2 = torch.bmm(coeffient, features_rec)

            target_point = torch.stack(data['target_point'], dim=1).to(device, dtype=torch.float32)
            red_light = data['red_light'][0][0].to(device, dtype=torch.float32)
            pred_wp = self.model_nav(features_rec_step2[:, nav_shijiao], target_point, red_light)

            return features_rec, pred_wp

        if self.train_model==1:
            # imgs_clean_batch = torch.stack(data['imgs_clean'], dim=0)  # 24 * 12 * 3 * 128 *128
            # self.imgs_clean_batch = imgs_clean_batch.view(-1, 3 * 128 * 128).to(device)
            # lidars_clean_batch = torch.stack(data['lidars_clean'], dim=0)  # 24 * 12 * 3 * 128 *128
            # self.lidars_clean_batch = lidars_clean_batch.view(-1, 64 * 122).to(device)
            # vae_feature_label, self.dec_clean_imgs, self.dec_clean_lidars = self.model_vae_fixed_parameters(self.imgs_clean_batch, self.lidars_clean_batch)
            # self.ll_clean = latent_loss(self.model_vae_fixed_parameters.z_mean, self.model_vae_fixed_parameters.z_sigma)
            # self.vae_feature_label = vae_feature_label.view(24, -1, 256).permute(1, 0, 2)

            imgs_att_batch = torch.stack(data['imgs_att'], dim=0)  # 24 * 12 * 3 * 128 *128
            imgs_att_batch = imgs_att_batch.view(-1, 3 * 128 * 128).to(device)
            lidars_att_batch = torch.stack(data['lidars_att'], dim=0)  # 24 * 12 * 3 * 128 *128
            lidars_att_batch = lidars_att_batch.view(-1, 64 * 122).to(device)

            vae_feature_att, _, _ = self.model_vae(imgs_att_batch, lidars_att_batch)
            vae_feature_att = vae_feature_att.view(24, -1, 256).permute(1, 0, 2)

            self.batch = vae_feature_att.shape[0]

            self.node_class, self.adj_rec = self.model_adj_rec(vae_feature_att, A)

            return self.node_class, self.adj_rec

        if self.train_model==2:
            imgs_clean_batch = torch.stack(data['imgs_clean'], dim=0)  # 24 * 12 * 3 * 128 *128
            self.imgs_clean_batch = imgs_clean_batch.view(-1, 3 * 128 * 128).to(device)
            lidars_clean_batch = torch.stack(data['lidars_clean'], dim=0)  # 24 * 12 * 3 * 128 *128
            self.lidars_clean_batch = lidars_clean_batch.view(-1, 64 * 122).to(device)
            vae_feature_label, self.dec_clean_imgs, self.dec_clean_lidars = vae_fixed_parameters(self.imgs_clean_batch, self.lidars_clean_batch)
            self.ll_clean = latent_loss(vae_fixed_parameters.z_mean, vae_fixed_parameters.z_sigma)
            self.vae_feature_label = vae_feature_label.view(24, -1, 256).permute(1, 0, 2)

            imgs_att_batch = torch.stack(data['imgs_att'], dim=0)  # 24 * 12 * 3 * 128 *128
            imgs_att_batch = imgs_att_batch.view(-1, 3 * 128 * 128).to(device)
            lidars_att_batch = torch.stack(data['lidars_att'], dim=0)  # 24 * 12 * 3 * 128 *128
            lidars_att_batch = lidars_att_batch.view(-1, 64 * 122).to(device)

            vae_feature_att, _, _ = self.model_vae(imgs_att_batch, lidars_att_batch)
            vae_feature_att = vae_feature_att.view(24, -1, 256).permute(1, 0, 2)

            self.batch = vae_feature_att.shape[0]

            self.node_class, self.adj_rec = self.model_adj_rec(vae_feature_att, A)

            vae_feature_att0 = vae_feature_att.clone()

            vae_feature_att0[:, 0:8, :] /= math.sqrt(3)
            vae_feature_att0[:, 8:16, :] /= 2
            vae_feature_att0[:, 16:24, :] /= math.sqrt(3)

            vae_feature_att1 = vae_feature_att0.unsqueeze(2)
            vae_feature_att2 = vae_feature_att0.unsqueeze(1)
            vae_feature_att3 = vae_feature_att2 - vae_feature_att1
            vae_feature_att4 = torch.norm(vae_feature_att3, p=2, dim=3)

            S_adj_pro = self.adj_rec - vae_feature_att4 * 0.01 / 2
            S_adj = prox_nuclear(S_adj_pro.detach(), 1).to(device)
            b_s = S_adj.shape[0]
            S_adj += torch.eye(24).unsqueeze(0).repeat(b_s, 1, 1).to(device)

            S_adj[S_adj < 0] = 0
            S_adj[S_adj > 1] = 1

            features_rec = self.model_feature_rec(vae_feature_att, S_adj, vae_feature_att)

            return features_rec

        if self.train_model==3:
            imgs_att_batch = torch.stack(data['imgs_clean'], dim=0)  # 24 * 12 * 3 * 128 *128
            imgs_att_batch = imgs_att_batch.view(-1, 3 * 128 * 128).to(device)
            lidars_att_batch = torch.stack(data['lidars_clean'], dim=0)  # 24 * 12 * 3 * 64 *122
            lidars_att_batch = lidars_att_batch.view(-1, 64 * 122).to(device)

            vae_feature_att, _, _ = self.model_vae(imgs_att_batch, lidars_att_batch)
            vae_feature_att = vae_feature_att.view(24, -1, 256).permute(1, 0, 2)

            self.batch = vae_feature_att.shape[0]

            self.node_class, self.adj_rec = self.model_adj_rec(vae_feature_att, A)

            vae_feature_att0 = vae_feature_att.clone()

            vae_feature_att0[:, 0:8, :] /= math.sqrt(3)
            vae_feature_att0[:, 8:16, :] /= 2
            vae_feature_att0[:, 16:24, :] /= math.sqrt(3)

            vae_feature_att1 = vae_feature_att0.unsqueeze(2)
            vae_feature_att2 = vae_feature_att0.unsqueeze(1)
            vae_feature_att3 = vae_feature_att2 - vae_feature_att1
            vae_feature_att4 = torch.norm(vae_feature_att3, p=2, dim=3)

            S_adj_pro = self.adj_rec - vae_feature_att4 * 0.01 / 2
            S_adj = prox_nuclear(S_adj_pro.detach(), 1).to(device)
            b_s = S_adj.shape[0]
            S_adj += torch.eye(24).unsqueeze(0).repeat(b_s, 1, 1).to(device)

            S_adj[S_adj < 0] = 0
            S_adj[S_adj > 1] = 1

            features_rec = self.model_feature_rec(vae_feature_att, S_adj, vae_feature_att)

            L = feature_smoothing_batch(S_adj)
            coeffient = torch.eye(24).unsqueeze(0).repeat(self.batch, 1, 1).to(device) + 2.5 * L
            coeffient = torch.linalg.inv(coeffient)
            features_rec_step2 = torch.bmm(coeffient, features_rec)

            target_point = torch.stack(data['target_point'], dim=1).to(device, dtype=torch.float32)
            # red_light = data['red_light'][0][0].to(device, dtype=torch.float32)
            red_light = torch.randn(24, 1).to(device, dtype=torch.float32)
            pred_wp = self.model_nav(features_rec_step2[:, nav_shijiao], target_point, red_light)

            return pred_wp


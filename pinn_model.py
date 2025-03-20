import scipy.io
import numpy as np
import torch
import torch.nn as nn
from pyDOE import lhs
import pandas as pd
import os
import math

# 定义PINN网络模块，包括数据读取函数，参数初始化
# 正问题和反问题的偏差和求导函数
# 全局参数
filename_load_model = './3DNS_model_train.pt'
filename_save_model = './3DNS_model_train.pt'
file_path = "./data_sample/"
filename_loss = './loss.csv'
# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("the processor is cpu!")
else:
    print("the processor is gpu!")

'''自定义sf-PINN激活函数'''


class Sinusoidal_act(nn.Module):
    def forward(self, input):
        return torch.sin(input)


fre_wight = np.linspace(1, 100, 50)
weights = torch.diag(torch.FloatTensor(fre_wight))


class MatMul(nn.Module):
    def __init__(self, weights):
        super(MatMul, self).__init__()
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.weights)


# 定义网络结构,由layer列表指定网络层数和神经元数
class PINN_Net(nn.Module):
    def __init__(self, layer_mat):
        super(PINN_Net, self).__init__()
        self.layer_num = len(layer_mat) - 1
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            # if i == 0:
            #     self.base.add_module(str(i) + "matmul", MatMul(weights))
            self.base.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            if i == 0:
                self.base.add_module(str(i) + "matmul", MatMul(weights))
                self.base.add_module(str(i) + "Act", Sinusoidal_act())
            else:
                self.base.add_module(str(i) + "Act", nn.Tanh())
        self.base.add_module(str(self.layer_num - 1) + "linear",
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        self.Re_nn = nn.Parameter(torch.tensor(8.0), requires_grad=True)
        self.Initial_parm(1)

    # 对权重和偏置初始化
    def Initial_parm(self, theta):
        for name, param in self.base.named_parameters():
            if name.endswith("width"):
                if name.startswith("0"):
                    param.data.normal_(0, theta)
                else:
                    nn.init.xavier_normal_(param)
            elif name.endswith("bias"):
                nn.init.zeros_(param)

    # def Initial_parm(self):
    #     for name, param in self.base.named_parameters():
    #         if name.endswith("width"):
    #             nn.init.xavier_normal_(param)
    #         elif name.endswith("bias"):
    #             nn.init.zeros_(param)

    def forward(self, x, y, z, t):
        X = torch.cat([x, y, z, t], 1).requires_grad_(True)
        predict = self.base(X)
        return predict

    # 定义类内方法，求数据点的损失 data_loss,不计算压力的误差
    def data_mse_without_p(self, x, y, t, u, v):
        predict_out = self.forward(x, y, t)
        psi = predict_out[:, 0].reshape(-1, 1)
        # p_predict = predict_out[:, 1].reshape(-1, 1)
        u_predict = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict
        # 定义类内方法，求数据点的损失 data_loss,不计算压力和垂直速度v的的误差

    def data_mse_windp(self, x, y, z, t, Vp):
        predict_out = self.forward(x, y, z, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        w_predict = predict_out[:, 2].reshape(-1, 1)
        p_predict = predict_out[:, 3].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        mse_predict = mse(u_predict, Vp) + mse(v_predict, batch_t_zeros)
        return mse_predict

    def data_mse_3D(self, x, y, z, t, theta, phi, v_r):
        predict_out = self.forward(x, y, z, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        w_predict = predict_out[:, 2].reshape(-1, 1)
        p_predict = predict_out[:, 3].reshape(-1, 1)

        v_r_predict = u_predict * torch.cos(phi) * torch.cos(theta) + v_predict * torch.cos(phi) * torch.sin(
            theta) + w_predict * torch.sin(phi)
        # f_exp = torch.sqrt(v_predict ** 2 + u_predict ** 2) + 40 * (z+1)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        # modify_vr = v_r / torch.cos(phi)
        mse_predict = mse(v_r_predict, v_r)
        return mse_predict

    def data_mse_points(self, x, y, z, t, u, v, w):
        predict_out = self.forward(x, y, z, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        w_predict = predict_out[:, 2].reshape(-1, 1)
        p_predict = predict_out[:, 3].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(w_predict, w)
        return mse_predict

    # 定义类内方法，求方程点的损失 equation_loss
    def equation_mse_3D(self, x, y, z, t):
        predict_out = self.forward(x, y, z, t)
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        w = predict_out[:, 2].reshape(-1, 1)
        p = predict_out[:, 3].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_z = torch.autograd.grad(v.sum(), z, create_graph=True)[0]
        w_t = torch.autograd.grad(w.sum(), t, create_graph=True)[0]
        w_x = torch.autograd.grad(w.sum(), x, create_graph=True)[0]
        w_y = torch.autograd.grad(w.sum(), y, create_graph=True)[0]
        w_z = torch.autograd.grad(w.sum(), z, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        p_z = torch.autograd.grad(p.sum(), z, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        v_zz = torch.autograd.grad(v_z.sum(), z, create_graph=True)[0]
        w_xx = torch.autograd.grad(w_x.sum(), x, create_graph=True)[0]
        w_yy = torch.autograd.grad(w_y.sum(), y, create_graph=True)[0]
        w_zz = torch.autograd.grad(w_z.sum(), z, create_graph=True)[0]
        # 计算偏微分方程的残差
        f_equation_x = (400 / (100 * 10)) * u_t + (u * u_x + v * u_y + w * u_z) + (1 / (10 ** 2)) * p_x - (
                u_xx + u_yy + u_zz) / (10 ** self.Re_nn)
        f_equation_y = (400 / (100 * 10)) * v_t + (u * v_x + v * v_y + w * v_z) + (1 / (10 ** 2)) * p_y - (
                v_xx + v_yy + v_zz) / (10 ** self.Re_nn)
        f_equation_z = (400 / (100 * 10)) * w_t + (u * w_x + v * w_y + w * w_z) + (1 / (10 ** 2)) * p_z - (
                w_xx + w_yy + w_zz) / (10 ** self.Re_nn)
        f_div = u_x + v_y + w_z

        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_z, batch_t_zeros) + mse(f_div, batch_t_zeros)
        return mse(f_equation_x, batch_t_zeros), mse(f_equation_y, batch_t_zeros), mse(f_equation_z,
                                                                                       batch_t_zeros), mse(f_div,
                                                                                                           batch_t_zeros)

    # 定义类内方法，求边界点的损失 equation_loss
    def bound_mse_3D(self, x, y, z, t):
        predict_out = self.forward(x, y, z, t)
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        w = predict_out[:, 2].reshape(-1, 1)
        p = predict_out[:, 3].reshape(-1, 1)
        # boundary loss
        mse = torch.nn.MSELoss()
        bound_value = torch.full((x.shape[0], 1), 0.5).float().requires_grad_(True).to(device)
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        mse_bound = mse(torch.sqrt(u ** 2 + v ** 2), bound_value) + mse(w, batch_t_zeros)
        return mse_bound

    # 定义类内方法，求方程点的损失 equation_loss
    def experience_mse_3D(self, x, y, z, t, z_mean, feature_mat):
        predict_out = self.forward(x, y, z, t)
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        w = predict_out[:, 2].reshape(-1, 1)
        p = predict_out[:, 3].reshape(-1, 1)
        # experience loss
        # z_mean_tensor = torch.tensor(z_mean, dtype=torch.float32).to(device)
        # z_anti = z * feature_mat[0, 6] + z_mean_tensor
        # f_wind_profile = torch.sqrt(u ** 2 + v ** 2) - 0.8 * (
        #         (z_anti / 100) ** 0.138)  # 风廓线 (z - z_mean) / feature_mat[0, 6]
        # f_wind_derection = torch.arctan2(v, u)
        # print(self.lam1, self.lam2)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        mse_equation = mse(w, batch_t_zeros)
        return mse_equation

    # 定义类内方法，求验证集损失
    def validation_error_plane(self, select_time, x_mean, y_mean, z_mean, feature_mat_numpy):
        path = './LES_data/WD_3D_0_0_turth_plane_100m/plane_' + str(int(select_time * 2)) + '.csv'
        # 使用pandas读入
        data = pd.read_csv(path)  # 读取文件中所有数据
        # 按列分离数据
        LES_data = np.array(data[['Points:0', 'Points:1', 'Points:2', 'U:0', 'U:1', 'U:2']])  # 读取速度u,v,w
        print('LES_u:', np.max(LES_data[:, 3]), np.min(LES_data[:, 3]), np.mean(LES_data[:, 3]))
        print('LES_v:', np.max(LES_data[:, 4]), np.min(LES_data[:, 4]), np.mean(LES_data[:, 4]))
        print('LES_w:', np.max(LES_data[:, 5]), np.min(LES_data[:, 5]), np.mean(LES_data[:, 5]))
        x_nom = (((LES_data[:, 0]) - x_mean.numpy()) / feature_mat_numpy[0, 6]).reshape(-1, 1)
        y_nom = (((LES_data[:, 1]) - y_mean.numpy()) / feature_mat_numpy[0, 6]).reshape(-1, 1)
        z_nom = (((LES_data[:, 2]) - z_mean.numpy()) / feature_mat_numpy[0, 6]).reshape(-1, 1)
        t_nom = (np.repeat(
            (2 * (select_time - feature_mat_numpy[1, 3]) / (feature_mat_numpy[0, 3] - feature_mat_numpy[1, 3]) - 1),
            x_nom.shape[0])).reshape(-1, 1)

        # _x = np.concatenate([x_nom, y_nom, z_nom, t_nom], axis=1)
        # x_selected = torch.FloatTensor(_x).requires_grad_(True).to(device)
        x_selected = torch.tensor(x_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        y_selected = torch.tensor(y_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        z_selected = torch.tensor(z_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        t_selected = torch.tensor(t_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)

        predict_out = self.forward(x_selected, y_selected, z_selected, t_selected)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        w_predict = predict_out[:, 2].reshape(-1, 1)
        p_predict = predict_out[:, 3].reshape(-1, 1)
        NN_u = u_predict.to('cpu').data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_v = v_predict.to('cpu').data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_w = w_predict.to('cpu').data.numpy().reshape(x_nom.shape[0], ) * 10
        print('NN_u:', np.max(NN_u), np.min(NN_u), np.mean(NN_u))
        print('NN_v:', np.max(NN_v), np.min(NN_v), np.mean(NN_v))
        print('NN_w:', np.max(NN_w), np.min(NN_w), np.mean(NN_w))
        error_u = NN_u - LES_data[:, 3]
        error_v = NN_v - LES_data[:, 4]
        error_w = NN_w - LES_data[:, 5]
        error_v_r = np.sqrt(NN_u ** 2 + NN_v ** 2 + NN_w ** 2) - np.sqrt(
            LES_data[:, 3] ** 2 + LES_data[:, 4] ** 2 + LES_data[:, 5] ** 2)
        print('v_mag:', np.max(error_v_r), np.min(error_v_r), np.mean(error_v_r))
        # plot_at_select_time_xyz(LES_data[:, 0] - offset_x, LES_data[:, 1] - offset_y, 100, error_v_r,
        #                         select_time, feature_mat_numpy)
        error_mse_u = (error_u ** 2).mean()
        error_mse_v = (error_v ** 2).mean()
        error_mse_w = (error_w ** 2).mean()
        error_mse_v_r = (error_v_r ** 2).mean()
        Eu = np.sqrt((NN_u ** 2).mean()) * np.sqrt((LES_data[:, 3] ** 2).mean())
        Ev = np.sqrt((NN_v ** 2).mean()) * np.sqrt((LES_data[:, 4] ** 2).mean())
        Ew = np.sqrt((NN_w ** 2).mean()) * np.sqrt((LES_data[:, 5] ** 2).mean())
        Evr = np.sqrt((np.sqrt(NN_u ** 2 + NN_v ** 2 + NN_w ** 2) ** 2).mean()) * np.sqrt(
            (np.sqrt(LES_data[:, 3] ** 2 + LES_data[:, 4] ** 2 + LES_data[:, 5] ** 2)).mean())
        return error_mse_u, error_mse_v, error_mse_w, error_mse_v_r, Eu, Ev, Ew, Evr


class SchedulerCosineDecayWarmup:
    def __init__(self, optimizer, lr, warmup_len, total_iters):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_len = warmup_len
        self.total_iters = total_iters
        self.current_iter = 0

    def get_lr(self):
        if self.current_iter < self.warmup_len:
            lr = self.lr * (self.current_iter + 1) / self.warmup_len
        else:
            cur = self.current_iter - self.warmup_len
            total = self.total_iters - self.warmup_len
            lr = 0.5 * (1 + np.cos(np.pi * cur / total)) * self.lr
        return lr

    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1


def read_lidar_data(file_path):
    csvs = [pd.read_csv(file_path + fname, header=1) for fname in os.listdir('./data_sample') if 'csv' in fname]
    df = pd.concat(csvs)
    SNR_num = np.linspace(18 + 1, 258 + 1, 61)
    column = np.linspace(18, 258, 61)  # 显示105m~1005m距离范围的风速数据，分辨率=15m
    add_clo = np.array([0, 1, 2, 3, 4, 5])
    total_col = np.append(column, add_clo)
    data = np.array(df.iloc[:, total_col])  # 61+6=67列
    SNR = np.array(df.iloc[:, SNR_num])  # 61列
    bad_points = np.where(SNR < 10) or np.where(SNR > 500)
    print("badpoints:")
    print(bad_points)
    # 删除坏点所在行
    # data = np.delete(data, 8, axis=0)
    for i in range(len(data[:, 1])):
        t = data[i, 61][9:21].replace(':', '.')
        t_hours = float(t[0:2])
        t_minutes = float(t[3:5])
        t_seconds = float(t[6:12])
        t_real = t_hours * 3600 + t_minutes * 60 + t_seconds
        data[i, 61] = t_real
    # print(type(data[:, 61][0]))
    data[:, 61] = np.around((data[:, 61] - data[:, 61][0]).astype('float'), decimals=3)
    data = data[np.argsort(data[:, 61])]
    time_seires = data[:, 61]
    azimuth = data[:, 65] * np.pi / 180
    pitch = data[:, 66] * np.pi / 180
    r = np.resize(np.linspace(105, 1005, 61), (61, 1))  # 生成距离门
    train_data = np.zeros(((len(time_seires) * (len(r))), 8))  # x ,y ,z ,t ,theta ,phi ,r,v_r
    print(train_data.shape)
    print(len(time_seires))
    print(len(r))
    for i in range(len(time_seires)):
        for j in range(len(r)):
            train_data[i * len(r) + j, 0] = r[j] * np.cos(pitch[i]) * np.cos(azimuth[i])
            train_data[i * len(r) + j, 1] = r[j] * np.cos(pitch[i]) * np.sin(azimuth[i])
            train_data[i * len(r) + j, 2] = r[j] * np.sin(pitch[i])
            train_data[i * len(r) + j, 3] = time_seires[i]
            train_data[i * len(r) + j, 4] = azimuth[i]
            train_data[i * len(r) + j, 5] = pitch[i]
            train_data[i * len(r) + j, 6] = r[j]
            train_data[i * len(r) + j, 7] = data[i, j]

    feature_mat = np.empty((2, 8))
    feature_mat[0, :] = np.max(train_data, 0)
    feature_mat[1, :] = np.min(train_data, 0)
    x = torch.tensor(train_data[:, 0].reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(train_data[:, 1].reshape(-1, 1), dtype=torch.float32)
    z = torch.tensor(train_data[:, 2].reshape(-1, 1), dtype=torch.float32)
    t = torch.tensor(train_data[:, 3].reshape(-1, 1), dtype=torch.float32)
    theta = torch.tensor(train_data[:, 4].reshape(-1, 1), dtype=torch.float32)
    phi = torch.tensor(train_data[:, 5].reshape(-1, 1), dtype=torch.float32)
    r = torch.tensor(train_data[:, 6].reshape(-1, 1), dtype=torch.float32)
    v_r = torch.tensor(train_data[:, 7].reshape(-1, 1), dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, z, t, theta, phi, r, v_r, feature_mat


# 定义离散标准高斯核函数
def GaussianFilter():
    # k_size = len(data)
    k_size = 15
    sigma = np.sqrt(2)
    gauss = np.zeros(k_size)
    for x in range(k_size):
        gauss[x] = np.exp(-((x - k_size / 2) ** 2) / (2 * (sigma ** 2)))
    gauss /= (sigma * np.sqrt(2 * np.pi))
    # gauss[7] = 1
    # plt.figure('Draw')
    # plt.plot(gauss)
    # plt.show()
    return gauss


def LES_data_extraction1(file_path):
    num_range_points = 50
    num_theta = 25
    num_phi = 16
    num_time = 400
    theta = np.zeros([num_time])
    phi = np.zeros([num_time])

    for i in range(num_phi):
        if i % 2 == 0:
            theta[i * num_theta: (i + 1) * num_theta] = np.linspace(168, 192, num_theta)
        else:
            theta[i * num_theta: (i + 1) * num_theta] = np.linspace(192, 168, num_theta)
    print(theta)
    print(phi)
    time_series = np.linspace(0, 199.5, num_time)
    # Lidar position
    offset_x = 1400
    offset_y = 0
    offset_z = 100
    point_star = [offset_x, offset_y, offset_z]
    # 数据存储矩阵：input:{x,y,z,t,theta,phi,r,v_r,u,v,w,p} 12
    data_input = np.zeros((num_time * num_range_points, 8))

    for t in range(num_time):
        # temp_phi = phi[t // num_theta]
        # temp_theta = theta[t % num_theta]
        temp_phi = phi[t]
        temp_theta = theta[t]
        # 读取射线的速度数据，分辨率为1m
        path = file_path + 'WD_multiplane_1800_0_plane_180_100m/' + str(t) + '.csv'
        # 使用pandas读入
        data = pd.read_csv(path)  # 读取文件中所有数据
        # 按列分离数据
        U = np.array(data[['U:0', 'U:1', 'U:2']])  # 读取速度u,v,w
        # lidar_u = np.zeros(len(U[47:798, 0]))
        lidar_v_r = np.zeros(len(U[1047:1798, 0]))
        # for i in range(len(U[1047:1798, 0])):
        #     lidar_v_r[i] = U[i, 0] * np.cos((temp_phi / 180) * np.pi) * np.cos((temp_theta / 180) * np.pi) + U[
        #         i, 1] * np.cos((temp_phi / 180) * np.pi) * np.sin((temp_theta / 180) * np.pi) + U[i, 2] * np.sin(
        #         (temp_phi / 180) * np.pi)
        for i in range(len(U[1047:1798, 0])):
            k = i + 1047
            lidar_v_r[i] = U[k, 0] * np.cos((temp_phi / 180) * np.pi) * np.cos((temp_theta / 180) * np.pi) + U[
                k, 1] * np.cos((temp_phi / 180) * np.pi) * np.sin((temp_theta / 180) * np.pi) + U[k, 2] * np.sin(
                (temp_phi / 180) * np.pi)
        # 将原始数值数据转化为lidar探测数据
        # lidar_u = (lidar_u[1:len(U[97:1013, 0])]).reshape(-1, 15)
        lidar_v_r = (lidar_v_r[1:len(U[1047:1798, 0])]).reshape(-1, 15)
        gauss_factor = GaussianFilter()
        # lidar_u = np.around(np.dot(lidar_u, gauss_factor), decimals=4)
        lidar_v_r = np.around(np.dot(lidar_v_r, gauss_factor), decimals=4)
        # 当前激光射线数据对应的径向坐标
        lidar_r = np.linspace(1055, 1790, num_range_points)
        lidar_rx = lidar_r * np.cos((temp_phi / 180) * np.pi) * np.cos((temp_theta / 180) * np.pi) + offset_x  # x
        lidar_ry = lidar_r * np.cos((temp_phi / 180) * np.pi) * np.sin((temp_theta / 180) * np.pi) + offset_y  # y
        lidar_rz = lidar_r * np.sin((temp_phi / 180) * np.pi) + offset_z  # z  不含坐标offset
        # 存储转换后的雷达探测数据
        data_input[t * num_range_points:(t + 1) * num_range_points, 0] = lidar_rx
        data_input[t * num_range_points:(t + 1) * num_range_points, 1] = lidar_ry
        data_input[t * num_range_points:(t + 1) * num_range_points, 2] = lidar_rz
        data_input[t * num_range_points:(t + 1) * num_range_points, 3] = time_series[t].repeat(num_range_points)
        data_input[t * num_range_points:(t + 1) * num_range_points, 4] = (temp_theta * np.pi / 180).repeat(
            num_range_points)
        data_input[t * num_range_points:(t + 1) * num_range_points, 5] = (temp_phi * np.pi / 180).repeat(
            num_range_points)
        data_input[t * num_range_points:(t + 1) * num_range_points, 6] = lidar_r
        data_input[t * num_range_points:(t + 1) * num_range_points, 7] = lidar_v_r
        # data_input[t * num_range_points:(t + 1) * num_range_points, 8] = U[lidar_r.astype(int), 0]
        # data_input[t * num_range_points:(t + 1) * num_range_points, 9] = U[lidar_r.astype(int), 1]
        # data_input[t * num_range_points:(t + 1) * num_range_points, 10] = U[lidar_r.astype(int), 2]
        # data_input[t * num_range_points:(t + 1) * num_range_points, 11] = U[lidar_r.astype(int), 3]

    feature_mat = np.empty((2, 8))
    feature_mat[0, :] = np.max(data_input[:, 0:8], 0)
    feature_mat[1, :] = np.min(data_input[:, 0:8], 0)
    feature_mat[0, 6] = 400
    # feature_mat[1, 2] = 0
    x_o = torch.tensor(data_input[:, 0].reshape(-1, 1), dtype=torch.float32)
    y_o = torch.tensor(data_input[:, 1].reshape(-1, 1), dtype=torch.float32)
    z_o = torch.tensor(data_input[:, 2].reshape(-1, 1), dtype=torch.float32)
    t_o = torch.tensor(data_input[:, 3].reshape(-1, 1), dtype=torch.float32)
    theta_o = torch.tensor(data_input[:, 4].reshape(-1, 1), dtype=torch.float32)
    phi_o = torch.tensor(data_input[:, 5].reshape(-1, 1), dtype=torch.float32)
    r_o = torch.tensor(data_input[:, 6].reshape(-1, 1), dtype=torch.float32)
    v_r_o = torch.tensor(data_input[:, 7].reshape(-1, 1), dtype=torch.float32)
    # u_o = torch.tensor(data_input[:, 8].reshape(-1, 1), dtype=torch.float32)
    # v_o = torch.tensor(data_input[:, 9].reshape(-1, 1), dtype=torch.float32)
    # w_o = torch.tensor(data_input[:, 10].reshape(-1, 1), dtype=torch.float32)
    # p_o = torch.tensor(data_input[:, 11].reshape(-1, 1), dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)

    return x_o, y_o, z_o, t_o, theta_o, phi_o, r_o, v_r_o, feature_mat


def LES_data_extraction2(file_path, feature_mat, x_mean, y_mean, z_mean):
    num_points = 5
    num_time = 400
    time_series = np.linspace(0, 199.5, num_time)
    # 数据存储矩阵：input:{x,y,z,t,theta,phi,r,v_r,u,v,w,p} 12
    data_o = np.zeros((num_time * num_points, 7))
    csvs = [pd.read_csv(file_path + 'WD_points/' + fname, header=0) for fname in os.listdir('./data_sample/WD_points')
            if
            'csv' in fname]
    df = pd.concat(csvs)
    df_final = df.dropna(axis=0, how="all")
    data_input = np.array(df_final)
    t_serirs = np.tile(time_series, 5)
    data_input[:, 3] = t_serirs
    np.random.shuffle(data_input)
    data_shuffle = torch.tensor(data_input, dtype=torch.float32)
    '''归一化'''
    x_nom = ((data_shuffle[:, 4] - x_mean) / feature_mat[0, 6]).reshape(-1, 1)
    y_nom = ((data_shuffle[:, 5] - y_mean) / feature_mat[0, 6]).reshape(-1, 1)
    z_nom = ((data_shuffle[:, 6] - z_mean) / feature_mat[0, 6]).reshape(-1, 1)
    t_nom = (2 * (data_shuffle[:, 3] - feature_mat[1, 3]) / (feature_mat[0, 3] - feature_mat[1, 3]) - 1).reshape(-1, 1)
    u_nom = (data_shuffle[:, 0] / 10).reshape(-1, 1)
    v_nom = (data_shuffle[:, 1] / 10).reshape(-1, 1)
    w_nom = (data_shuffle[:, 2] / 10).reshape(-1, 1)
    data_out = torch.cat([x_nom, y_nom, z_nom, t_nom, u_nom, v_nom, w_nom], 1)

    return data_out


def LES_initial_state(file_path):
    offset_x = 4000
    offset_y = 2000
    offset_z = 0
    path = file_path + 'initial_state/3D_10to500_0.csv'
    data = pd.read_csv(path)  # 读取文件中所有数据
    # 按列分离数据
    initial_data = np.array(data[['Points:0', 'Points:1', 'Points:2', 'U:0', 'U:1', 'U:2', 'p_rgh']])  # 读取速度x,y,y,u,v,w
    x = torch.tensor((initial_data[:, 0] - offset_x).reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor((initial_data[:, 1] - offset_y).reshape(-1, 1), dtype=torch.float32)
    z = torch.tensor((initial_data[:, 2] - offset_z).reshape(-1, 1), dtype=torch.float32)
    t = torch.zeros_like(z, dtype=torch.float32)
    u = torch.tensor((initial_data[:, 3]).reshape(-1, 1), dtype=torch.float32)
    v = torch.tensor((initial_data[:, 4]).reshape(-1, 1), dtype=torch.float32)
    w = torch.tensor((initial_data[:, 5]).reshape(-1, 1), dtype=torch.float32)
    return x, y, z, t, u, v, w


def LES_data_extraction_wind_profile(file_path):
    num_time = 196
    height_points = 602
    height_max = 600
    height_min = 0
    time_series = np.linspace(0.2, 39.2, num_time)
    data_output = np.zeros((num_time * height_points, 7))  # x, y, z, t, u, v, w
    for t in range(num_time):
        # 读取射线的速度数据，分辨率为1m
        path = file_path + 'wd_prodile/wd_' + str(t) + '.csv'
        # 使用pandas读入
        data = pd.read_csv(path)  # 读取文件中所有数据
        # 按列分离数据
        U = np.array(data[['U:0', 'U:1', 'U:2']])  # 读取速度u,v,w
        hight_value = np.linspace(height_min, height_max, height_points)
        data_output[t * height_points:(t + 1) * height_points, 0] = np.zeros((height_points,))
        data_output[t * height_points:(t + 1) * height_points, 1] = np.zeros((height_points,))
        data_output[t * height_points:(t + 1) * height_points, 2] = hight_value
        data_output[t * height_points:(t + 1) * height_points, 3] = time_series[t].repeat(height_points)
        data_output[t * height_points:(t + 1) * height_points, 4] = U[:, 0]
        data_output[t * height_points:(t + 1) * height_points, 5] = U[:, 1]
        data_output[t * height_points:(t + 1) * height_points, 6] = U[:, 2]

    x = torch.tensor(data_output[:, 0].reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(data_output[:, 1].reshape(-1, 1), dtype=torch.float32)
    z = torch.tensor(data_output[:, 2].reshape(-1, 1), dtype=torch.float32)
    t = torch.tensor(data_output[:, 3].reshape(-1, 1), dtype=torch.float32)
    u = torch.tensor(data_output[:, 4].reshape(-1, 1), dtype=torch.float32)
    v = torch.tensor(data_output[:, 5].reshape(-1, 1), dtype=torch.float32)
    w = torch.tensor(data_output[:, 6].reshape(-1, 1), dtype=torch.float32)
    return x, y, z, t, u, v, w


def data_nom(x, y, z, t, theta, phi, r, v_r, feature_mat, x_mean, y_mean, z_mean):
    t_nom = 2 * (t - feature_mat[1, 3]) / (feature_mat[0, 3] - feature_mat[1, 3]) - 1
    r_nom = r / feature_mat[0, 6]
    # v_r_nom = 2 * (v_r - feature_mat[1, 7]) / (feature_mat[0, 7] - feature_mat[1, 7]) - 1
    # v_r不做归一化，但变量名仍保留
    v_r_nom = v_r / 10
    x_nom = (x - x_mean) / feature_mat[0, 6]
    y_nom = (y - y_mean) / feature_mat[0, 6]
    z_nom = (z - z_mean) / feature_mat[0, 6]
    feature_mat_nom = feature_mat.clone()
    feature_mat_nom[0, :] = 1
    feature_mat_nom[1, :] = -1
    feature_mat_nom[:, 4:6] = feature_mat[:, 4:6]
    feature_mat_nom[:, 7] = feature_mat[:, 7]
    return x_nom, y_nom, z_nom, t_nom, r_nom, v_r_nom, feature_mat_nom


def read_data_portion(filename, portion_x, portion_y):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T

    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    index_arr_x = np.linspace(0, len(x_unique) - 1, int(len(x_unique) * portion_x)).astype(int).reshape(-1, 1)
    index_arr_y = np.linspace(0, len(y_unique) - 1, int(len(y_unique) * portion_y)).astype(int).reshape(-1, 1)
    x_select = x_unique[index_arr_x].reshape(-1, 1)
    y_select = y_unique[index_arr_y].reshape(-1, 1)
    del x_unique, y_unique, index_arr_x, index_arr_y
    index_x = np.empty((0, 1), dtype=int)
    index_y = np.empty((0, 1), dtype=int)
    for select_1 in x_select:
        index_x = np.append(index_x, np.where(x == select_1)[0].reshape(-1, 1), 0)
    for select_2 in y_select:
        index_y = np.append(index_y, np.where(y == select_2)[0].reshape(-1, 1), 0)
    index_all = np.intersect1d(index_x, index_y, assume_unique=False, return_indices=False).reshape(-1, 1)
    x = x[index_all].reshape(-1, 1)
    y = y[index_all].reshape(-1, 1)
    t = t[index_all].reshape(-1, 1)
    u = u[index_all].reshape(-1, 1)
    v = v[index_all].reshape(-1, 1)
    p = p[index_all].reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat


def shuffle_data(x, y, z, t, theta, phi, r, v_r):
    X_total = torch.cat([x, y, z, t, theta, phi, r, v_r], 1)
    X_total_arr = X_total.data.numpy()
    np.random.shuffle(X_total_arr)
    X_total_random = torch.tensor(X_total_arr)
    return X_total_random


def nom_shuffle_data(x, y, z, t, u, v, w, x_mean, y_mean, z_mean, feature_mat):
    t_nom = 2 * (t - feature_mat[1, 3]) / (feature_mat[0, 3] - feature_mat[1, 3]) - 1
    x_nom = (x - x_mean) / feature_mat[0, 6]
    y_nom = (y - y_mean) / feature_mat[0, 6]
    z_nom = (z - z_mean) / feature_mat[0, 6]
    u_nom = u / 10
    v_nom = v / 10
    w_nom = w / 10
    X_total = torch.cat([x_nom, y_nom, z_nom, t_nom, u_nom, v_nom, w_nom], 1)
    X_total_arr = X_total.data.numpy()
    np.random.shuffle(X_total_arr)
    X_total_random = torch.tensor(X_total_arr)
    return X_total_random


def shuffle_data_uvw(x, y, z, t, theta, phi, r, v_r, u, v, w):
    X_total = torch.cat([x, y, z, t, theta, phi, r, v_r, u, v, w], 1)
    X_total_arr = X_total.data.numpy()
    np.random.shuffle(X_total_arr)
    X_total_random = torch.tensor(X_total_arr)
    return X_total_random


# 生成矩形域方程点
def generate_eqp_rect(low_bound, up_bound, dimension, points, feature_mat, x_mean, y_mean, z_mean):
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)  # xyzt
    eqa_xyzt_nom = eqa_xyzt.copy()
    eqa_xyzt_nom[:, 0] = (eqa_xyzt[:, 0] - x_mean.numpy()) / feature_mat[0, 6]
    eqa_xyzt_nom[:, 1] = (eqa_xyzt[:, 1] - y_mean.numpy()) / feature_mat[0, 6]
    eqa_xyzt_nom[:, 2] = (eqa_xyzt[:, 2] - z_mean.numpy()) / feature_mat[0, 6]
    eqa_xyzt_nom[:, 3] = 2 * (eqa_xyzt[:, 3] - feature_mat[1, 3]) / (feature_mat[0, 3] - feature_mat[1, 3]) - 1
    per = np.random.permutation(eqa_xyzt_nom.shape[0])
    new_xyzt = eqa_xyzt_nom[per, :]
    Eqa_points = torch.from_numpy(new_xyzt).float()
    return Eqa_points


# 生成球体域方程点
def generate_eqp_sphere(low_bound, up_bound, dimension, points, feature_mat, x_mean, y_mean, z_mean):
    eqa_rt = low_bound + (up_bound - low_bound) * lhs(dimension, points)  # t,theta,phi,r
    eqa_xyzt = eqa_rt.copy()
    for i in range(eqa_rt.shape[0]):
        eqa_xyzt[i, 0] = eqa_rt[i, 3] * np.cos(eqa_rt[i, 2]) * np.cos(eqa_rt[i, 1])
        eqa_xyzt[i, 1] = eqa_rt[i, 3] * np.cos(eqa_rt[i, 2]) * np.sin(eqa_rt[i, 1])
        eqa_xyzt[i, 2] = eqa_rt[i, 3] * (np.sin(eqa_rt[i, 2]))
        eqa_xyzt[i, 3] = eqa_rt[i, 0]
    x_eqa = (eqa_xyzt[:, 0] - x_mean.numpy()) / feature_mat[0, 6]
    y_eqa = (eqa_xyzt[:, 1] - y_mean.numpy()) / feature_mat[0, 6]
    z_eqa = (eqa_xyzt[:, 2] - z_mean.numpy()) / feature_mat[0, 6]
    t_eqa = 2 * (eqa_xyzt[:, 3] - feature_mat[1, 3]) / (feature_mat[0, 3] - feature_mat[1, 3]) - 1

    per = np.random.permutation(eqa_rt.shape[0])
    x_eqa_new = x_eqa[per].reshape(-1, 1)
    y_eqa_new = y_eqa[per].reshape(-1, 1)
    z_eqa_new = z_eqa[per].reshape(-1, 1)
    t_eqa_new = t_eqa[per].reshape(-1, 1)
    new_xyzt = np.concatenate((x_eqa_new, y_eqa_new, z_eqa_new, t_eqa_new), 1)
    Eqa_points = torch.from_numpy(new_xyzt).float()
    return Eqa_points


# 生成地表边界配点
def generate_boundary_points(low_bound, up_bound, dimension, points, feature_mat, x_mean, y_mean, z_mean):
    eqa_rt = low_bound + (up_bound - low_bound) * lhs((dimension - 1), points)  # x,y,t
    eqa_xyzt = eqa_rt.copy()
    z_b = np.zeros((len(eqa_xyzt[:, 0]), 1))
    x_eqa = (eqa_xyzt[:, 0] - x_mean.numpy()) / feature_mat[0, 6]
    y_eqa = (eqa_xyzt[:, 1] - y_mean.numpy()) / feature_mat[0, 6]
    z_eqa = (z_b - z_mean.numpy()) / feature_mat[0, 6]
    t_eqa = 2 * (eqa_xyzt[:, 2] - feature_mat[1, 3]) / (feature_mat[0, 3] - feature_mat[1, 3]) - 1

    per = np.random.permutation(eqa_rt.shape[0])
    x_eqa_new = x_eqa[per].reshape(-1, 1)
    y_eqa_new = y_eqa[per].reshape(-1, 1)
    z_eqa_new = z_eqa[per].reshape(-1, 1)
    t_eqa_new = t_eqa[per].reshape(-1, 1)
    new_xyzt = np.concatenate((x_eqa_new, y_eqa_new, z_eqa_new, t_eqa_new), 1)
    Bound_points = torch.from_numpy(new_xyzt).float()
    return Bound_points


def generate_profile_points(low_bound, up_bound, dimension, points, feature_mat, x_mean, y_mean, z_mean):
    z_series = np.array([10, 20, 40, 50, 80, 100, 150, 160, 200, 250, 300, 320, 350])
    v_plane = np.array([6.18, 7.06, 7.70, 7.92, 8.34, 8.54, 8.83, 8.86, 9.02, 9.25, 9.55, 9.61, 9.65])
    data_pro = np.zeros((points * len(z_series), 5))
    for i in range(len(z_series)):
        data_pro[points * i: points * (i + 1), 0:3] = low_bound + (up_bound - low_bound) * lhs((dimension - 1),
                                                                                               points)  # x,y,t
        data_pro[points * i: points * (i + 1), 3] = z_series[i]
        data_pro[points * i: points * (i + 1), 4] = v_plane[i]
    data_pro_x = data_pro.copy()
    x_p = (data_pro_x[:, 0] - x_mean.numpy()) / feature_mat[0, 6]
    y_p = (data_pro_x[:, 1] - y_mean.numpy()) / feature_mat[0, 6]
    z_p = (data_pro_x[:, 3] - z_mean.numpy()) / feature_mat[0, 6]
    t_eqa = 2 * (data_pro_x[:, 2] - feature_mat[1, 3]) / (feature_mat[0, 3] - feature_mat[1, 3]) - 1
    v_p = data_pro_x[:, 4] / 10
    per = np.random.permutation(data_pro.shape[0])
    x_eqa_new = x_p[per].reshape(-1, 1)
    y_eqa_new = y_p[per].reshape(-1, 1)
    z_eqa_new = z_p[per].reshape(-1, 1)
    t_eqa_new = t_eqa[per].reshape(-1, 1)
    v_euq_new = v_p[per].reshape(-1, 1)
    new_xyzt = np.concatenate((x_eqa_new, y_eqa_new, z_eqa_new, t_eqa_new, v_euq_new), 1)
    Bound_points = torch.from_numpy(new_xyzt).float()
    return Bound_points


# 生成预训练的归一化数据
def pre_train_data_generation(feature_mat_numpy):
    mesh_t_points = 80
    mesh_theta_points = 50
    mesh_phi_points = 15
    mesh_r_points = 60
    mesh_t = np.linspace(feature_mat_numpy[1, 3], feature_mat_numpy[0, 3], mesh_t_points)
    mesh_theta = np.linspace(feature_mat_numpy[1, 4], feature_mat_numpy[0, 4], mesh_theta_points)
    mesh_phi = np.linspace(feature_mat_numpy[1, 5], feature_mat_numpy[0, 5], mesh_phi_points)
    mesh_r = np.linspace(feature_mat_numpy[1, 6], feature_mat_numpy[0, 6], mesh_r_points)
    # mesh_grid: x ,y ,z , t, theta ,phi ,r,u,v,w
    mesh_grid = np.zeros(((mesh_t_points * mesh_r_points * mesh_theta_points * mesh_phi_points), 7))
    for tn in range(mesh_t_points):
        for i in range(mesh_phi_points):
            for j in range(mesh_theta_points):
                for k in range(mesh_r_points):
                    row_num = tn * mesh_phi_points * mesh_theta_points * mesh_r_points + i * mesh_theta_points * mesh_r_points + j * mesh_r_points + k
                    mesh_grid[row_num, 0] = mesh_r[k] * np.cos(mesh_phi[i]) * np.cos(mesh_theta[j])
                    mesh_grid[row_num, 1] = mesh_r[k] * np.cos(mesh_phi[i]) * np.sin(mesh_theta[j])
                    mesh_grid[row_num, 2] = mesh_r[k] * (np.sin(mesh_phi[i]))
                    mesh_grid[row_num, 3] = mesh_t[tn]
                    mesh_grid[row_num, 4] = -3 * np.log10(5 * mesh_grid[row_num, 2]) + np.random.normal(0, 0.5)
                    mesh_grid[row_num, 5] = np.random.normal(0, 0.5)
                    mesh_grid[row_num, 6] = np.random.normal(0, 0.25)

    np.random.shuffle(mesh_grid)
    pre_train_data = mesh_grid.copy()
    pre_train_data[:, 0] = 2 * (mesh_grid[:, 0] - feature_mat_numpy[1, 0]) / (
            feature_mat_numpy[0, 0] - feature_mat_numpy[1, 0]) - 1
    pre_train_data[:, 1] = 2 * (mesh_grid[:, 1] - feature_mat_numpy[1, 1]) / (
            feature_mat_numpy[0, 1] - feature_mat_numpy[1, 1]) - 1
    pre_train_data[:, 2] = 2 * (mesh_grid[:, 2] - feature_mat_numpy[1, 2]) / (
            feature_mat_numpy[0, 2] - feature_mat_numpy[1, 2]) - 1
    pre_train_data[:, 3] = 2 * (mesh_grid[:, 3] - feature_mat_numpy[1, 3]) / (
            feature_mat_numpy[0, 3] - feature_mat_numpy[1, 3]) - 1
    pre_train_data[:, 4] = mesh_grid[:, 4]
    pre_train_data[:, 5] = mesh_grid[:, 5]
    pre_train_data[:, 6] = mesh_grid[:, 6]
    X_total_pre = torch.tensor(pre_train_data, dtype=torch.float32)
    return X_total_pre


def f_equation_identification_3D(x, y, z, t, pinn_example):
    # 正问题,需要用户自行提供系统的参数值
    predict_out = pinn_example.forward(x, y, z, t)
    u = predict_out[:, 0].reshape(-1, 1)
    v = predict_out[:, 1].reshape(-1, 1)
    w = predict_out[:, 2].reshape(-1, 1)
    p = predict_out[:, 3].reshape(-1, 1)
    return u, v, w, p

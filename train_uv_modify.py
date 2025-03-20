import numpy as np
import torch.optim.lr_scheduler

from pinn_model import *
import pandas as pd
import os
# import sys
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(4)
path = './train_history/'
writer = SummaryWriter(path)

# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("the processor is cpu!")

# 稀疏训练-无数据归一化处理
# 训练代码主体
# 稀疏占比
air_density = 1.18
air_viscosity = 14.8e-6  # 运动粘度
# Re = 1 * 1050 * 10 / air_viscosity  # UL/V
# Re = 100000  # UL/V
debug_key = 1
# 数据点和方程点加载
N_eqa = 500000
N_bound = 50000
dimension = 3 + 1
# layer_mat = [4, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 4]
layer_mat = [4, 50, 50, 50, 50, 4]
# layer_mat = [4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]
'''定义中心位置'''
x_mean = torch.tensor(0)
y_mean = torch.tensor(0)
z_mean = torch.tensor(100)
'''导入雷达1数据'''
x, y, z, t, theta, phi, r, v_r, feature_mat = LES_data_extraction1(file_path)  # x ,y ,z ,t ,theta ,phi ,r,v_r
x_nom, y_nom, z_nom, t_nom, r_nom, v_r_nom, feature_mat_nom = data_nom(x, y, z, t, theta, phi, r, v_r, feature_mat,
                                                                       x_mean, y_mean, z_mean)
X_random = shuffle_data(x_nom, y_nom, z_nom, t_nom, theta, phi, r_nom, v_r_nom).to(device)
show1 = X_random.cpu().numpy()
feature_mat_numpy = feature_mat.data.numpy()
print(feature_mat_numpy)
del x, y, z, t, theta, phi, r, v_r
del x_nom, y_nom, z_nom, t_nom, r_nom, v_r_nom
'''导入离散点数据'''
# data_points = LES_data_extraction2(file_path, feature_mat, x_mean, y_mean, z_mean)  # x ,y ,z ,t ,theta ,phi ,r,v_r

'''生成N-S方程计算配点'''
lb = np.array([-400.0, -400.0, 100.0, 0.0])  # x,y,z,t
ub = np.array([400.0, 400.0, 100.0, 199.5])
Eqa_points = generate_eqp_rect(lb, ub, dimension, N_eqa, feature_mat_numpy, x_mean, y_mean, z_mean)
show_e = Eqa_points.data.numpy()
'''生成边界条件配点'''
# xy_lb = np.array([feature_mat_numpy[1, 0], feature_mat_numpy[1, 1], feature_mat_numpy[1, 3]])  # x,y,t
# xy_ub = np.array([feature_mat_numpy[0, 0], feature_mat_numpy[0, 1], feature_mat_numpy[0, 3]])  # x,y,t
# Boundary_points = generate_boundary_points(xy_lb, xy_ub, dimension, N_bound, feature_mat_numpy, x_mean, y_mean, z_mean)
# show_b = Boundary_points.data.numpy()

# 创建PINN模型实例，并将实例分配至对应设备
pinn_net = PINN_Net(layer_mat)
pinn_net = pinn_net.to(device)
# 用以记录各部分损失的列表
losses = np.empty((0, 5), dtype=float)
#
if os.path.exists(filename_save_model):
    pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
if os.path.exists(filename_loss):
    loss_read = pd.read_csv('loss.csv', header=None)
    losses = loss_read.values
# 优化器和学习率衰减设置
lr = 0.002
optimizer = torch.optim.Adam(pinn_net.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
epochs = 400
scheduler = SchedulerCosineDecayWarmup(optimizer, lr, epochs / 10, epochs)

# 选取batch size 此处也可使用data_loader
ratio = 0.02
batch_size_data = int(ratio * X_random.shape[0])
batch_size_eqa = int(ratio * Eqa_points.shape[0])

inner_iter = int(X_random.size(0) / batch_size_data)
eqa_iter = int(Eqa_points.size(0) / batch_size_eqa)

print(batch_size_data, batch_size_eqa, inner_iter, eqa_iter)

for epoch in range(epochs):
    for batch_iter in range(inner_iter + 1):
        optimizer.zero_grad()
        # 在全集中随机取batch
        if batch_iter < inner_iter:
            x_train = X_random[:, 0].reshape(-1, 1).clone().requires_grad_(True)
            y_train = X_random[:, 1].reshape(-1, 1).clone().requires_grad_(True)
            z_train = X_random[:, 2].reshape(-1, 1).clone().requires_grad_(True)
            t_train = X_random[:, 3].reshape(-1, 1).clone().requires_grad_(True)
            theta_train = X_random[:, 4].reshape(-1, 1).clone().requires_grad_(True)
            phi_train = X_random[:, 5].reshape(-1, 1).clone().requires_grad_(True)
            r_v_train = X_random[:, 7].reshape(-1, 1).clone().requires_grad_(True)

            x_eqa = Eqa_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 0].reshape(
                batch_size_eqa, 1).clone().requires_grad_(True).to(device)
            y_eqa = Eqa_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 1].reshape(
                batch_size_eqa, 1).clone().requires_grad_(True).to(device)
            z_eqa = Eqa_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 2].reshape(
                batch_size_eqa, 1).clone().requires_grad_(True).to(device)
            t_eqa = Eqa_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 3].reshape(
                batch_size_eqa, 1).clone().requires_grad_(True).to(device)

        mse_predict1 = pinn_net.data_mse_3D(x_train, y_train, z_train, t_train, theta_train, phi_train, r_v_train)
        # mse_predict2 = pinn_net.data_mse_points(x_point, y_point, z_point, t_point, u_point, v_point, w_point)
        mse_equation1, mse_equation2, mse_equation3, mse_equation4 = pinn_net.equation_mse_3D(x_eqa, y_eqa, z_eqa,
                                                                                              t_eqa)
        mse_experince_equation = pinn_net.experience_mse_3D(x_eqa, y_eqa, z_eqa, t_eqa, z_mean, feature_mat)
        loss = 10 * mse_predict1 + 2 * (mse_equation1 + mse_equation2 + mse_equation3 + 0.2 * mse_equation4)
        # debug 查看loss变化曲线
        loss.backward()
        optimizer.step()
        with torch.autograd.no_grad():
            # 输出状态
            if (batch_iter + 1) % 5 == 0 and debug_key == 1:
                print("Epoch:", (epoch + 1), "  Bacth_iter:", batch_iter + 1, " Training Loss:",
                      round(float(loss.data), 8))
            # 每1个epoch保存状态（模型状态,loss,迭代次数）
            if (batch_iter + 1) % inner_iter == 0:
                torch.save(pinn_net.state_dict(), filename_save_model)
                writer.add_scalars('Train_loss', {'total_loss': loss,
                                                  'loss_predict': mse_predict1,
                                                  # 'loss_points': mse_predict2,
                                                  'mse_experince_equation': mse_experince_equation,
                                                  'loss_equation1': mse_equation1,
                                                  'loss_equation2': mse_equation2,
                                                  'loss_equation3': mse_equation3,
                                                  'loss_equation4': mse_equation4
                                                  # 'loss_wp': mse_windp,
                                                  # 'mse_boundary': mse_boundary
                                                  # 'loss_windp': mse_windp
                                                  # 'mse_experince_predict': mse_experince_predict
                                                  }, epoch)
                # loss_all = loss.cpu().data.numpy().reshape(1, 1)
                # loss_predict = mse_predict.cpu().data.numpy().reshape(1, 1)
                # loss_data_exp = mse_predict.cpu().data.numpy().reshape(1, 1)
                # loss_equation = mse_equation.cpu().data.numpy().reshape(1, 1)
                # loss_experince = mse_equation.cpu().data.numpy().reshape(1, 1)
                # loss_set = np.concatenate((loss_all, loss_predict, loss_data_exp, loss_equation, loss_experince), 1)
                # losses = np.append(losses, loss_set, 0)
                # loss_save = pd.DataFrame(losses)
                # loss_save.to_csv(filename_loss, index=False, header=False)
                # del loss_save
    scheduler.step()
    if (epoch + 1) == epochs / 4 or (epoch + 1) == epochs / 2:
        torch.save(pinn_net.state_dict(), './3DNS_model_train' + "epochs_{:.0f}".format((epoch + 1)) + '.pt')
        print(epoch + 1)
print("one oK")
torch.save(pinn_net.state_dict(), filename_save_model)

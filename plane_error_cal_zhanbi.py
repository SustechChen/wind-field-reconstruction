# 将预测结果和真实结果进行可视化对比
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pinn_model import *
import pandas as pd
import seaborn as sns
# import imageio
import imageio.v2 as imageio
import scipy.stats as st
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

device = torch.device("cpu")
# mpl.use("Agg")

filename_load_model = './3DNS_model_train.pt'
layer_mat = [4, 50, 50, 50, 50,  4]
# layer_mat = [4, 100, 100, 100, 100,  4]
'''定义中心位置'''
x_mean = torch.tensor(0)
y_mean = torch.tensor(0)
z_mean = torch.tensor(100)
'''导入雷达1数据'''
x, y, z, t, theta, phi, r, v_r, feature_mat = LES_data_extraction1(file_path)  # x ,y ,z ,t ,theta ,phi ,r,v_r
x_nom, y_nom, z_nom, t_nom, r_nom, v_r_nom, feature_mat_nom = data_nom(x, y, z, t, theta, phi, r, v_r, feature_mat,
                                                                       x_mean, y_mean, z_mean)
X_random = shuffle_data(x_nom, y_nom, z_nom, t_nom, theta, phi, r_nom, v_r_nom)
show1 = X_random.cpu().numpy()
feature_mat_numpy = feature_mat.data.numpy()
print(feature_mat_numpy)
del x, y, z, t, theta, phi, r, v_r
del x_nom, y_nom, z_nom, t_nom, r_nom, v_r_nom

pinn_net = PINN_Net(layer_mat)
pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))

select_time_step = 100  # 实际值
select_time = select_time_step * 2
num_time = 100
time_series = np.linspace(0, 199.5, num_time)


def fluctuation_u(time_steps, height_value):
    time_lists = np.linspace(0, 199.5, time_steps)
    error_umag_t = np.zeros((time_steps, 6561))
    error_umag_avg = np.zeros(time_steps)
    error_umag_avg_rmse = np.zeros(time_steps)
    for t in range(len(time_lists)):
        path = './LES_data/WD_3D_0_0_turth_plane_100m/plane_' + str(int(time_lists[t] * 2)) + '.csv'
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
            (2 * (time_lists[t] - feature_mat_numpy[1, 3]) / (feature_mat_numpy[0, 3] - feature_mat_numpy[1, 3]) - 1),
            x_nom.shape[0])).reshape(-1, 1)
        x_selected = torch.tensor(x_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        y_selected = torch.tensor(y_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        z_selected = torch.tensor(z_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        t_selected = torch.tensor(t_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        u_predict, v_predict, w_predict, p_predict = f_equation_identification_3D(x_selected, y_selected, z_selected,
                                                                                  t_selected, pinn_net)
        NN_u = u_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_v = v_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_w = w_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_vr = np.sqrt(NN_u ** 2 + NN_v ** 2 + NN_w ** 2)
        LES_vr = np.sqrt(LES_data[:, 3] ** 2 + LES_data[:, 4] ** 2 + LES_data[:, 5] ** 2)
        error = NN_vr - LES_vr
        # NN_vr_2D = NN_vr[:6400].reshape(80, 80) - NN_enavg
        # LES_vr_2D = LES_vr[:6400].reshape(80, 80) - LES_enavg
        # NN_u_2D = NN_u[:6400].reshape(80, 80) - NN_avg_u_t
        # LES_u_2D = LES_data[:, 3][:6400].reshape(80, 80) - LES_avg_u_t
        # NN_v_2D = NN_v[:6400].reshape(80, 80) - NN_avg_v_t
        # LES_v_2D = LES_data[:, 4][:6400].reshape(80, 80) - LES_avg_v_t
        # NN_w_2D = NN_w[:6400].reshape(80, 80) - NN_avg_w_t
        # LES_w_2D = LES_data[:, 5][:6400].reshape(80, 80) - LES_avg_w_t
        NN_u_2D = NN_u[:6400].reshape(80, 80)
        LES_u_2D = LES_data[:, 3][:6400].reshape(80, 80)
        NN_v_2D = NN_v[:6400].reshape(80, 80)
        LES_v_2D = LES_data[:, 4][:6400].reshape(80, 80)
        NN_w_2D = NN_w[:6400].reshape(80, 80)
        LES_w_2D = LES_data[:, 5][:6400].reshape(80, 80)
        # 计算归一化速度
        U_inf = 8
        co_avg_mag = U_inf
        NN_umag_2D = np.sqrt(NN_u_2D ** 2 + NN_v_2D ** 2 + NN_w_2D ** 2) / co_avg_mag
        LES_umag_2D = np.sqrt(LES_u_2D ** 2 + LES_v_2D ** 2 + LES_w_2D ** 2) / co_avg_mag
        error_umag_2D = NN_umag_2D - LES_umag_2D
        error_umag_avg[t] = np.mean(error_umag_2D)
        error_umag_avg_rmse[t] = np.sqrt(np.mean(error_umag_2D ** 2))
        error_umag_t[t, :] = np.mean(((NN_vr - LES_vr) / co_avg_mag) ** 2)
        '''画图du_dx,du_dy'''
        # 创建一个新的图形和三个子图
        # 设置全局字体
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.75)
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        min_value = 0.8
        max_value = 1.2
        min_value2 = 0
        max_value2 = 0.1
        # xx = np.linspace(-400+600, 400+600, 80)
        # yy = np.linspace(-400+600, 400+600, 80)
        xx = np.linspace(-400, 400, 80)
        yy = np.linspace(-400, 400, 80)
        x_p, y_p = np.meshgrid(xx, yy)
        v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        v_norm2 = mpl.colors.Normalize(vmin=min_value2, vmax=max_value2)
        # 在每个子图上创建一个散点图
        titles = ['Prediction ' + 't = ' + "{:.1f}".format(time_lists[t]) + ' s',
                  'True ''t = ' + "{:.1f}".format(time_lists[t]) + ' s',
                  'Error ''t = ' + "{:.1f}".format(time_lists[t]) + ' s']  # 这里添加你的标题
        for i, (ax, data, title) in enumerate(zip(axs, [(NN_umag_2D).reshape(-1),
                                                        (LES_umag_2D).reshape(-1),
                                                        (np.abs(error_umag_2D)).reshape(-1)], titles)):
            if i == 2:
                c2 = ax.scatter(x_p, y_p, c=data.reshape(-1), s=10, cmap='coolwarm', alpha=0.8, norm=v_norm2)
                ax.set_ylabel('Y (m)', fontsize=14, labelpad=10)
            else:
                c1 = ax.scatter(x_p, y_p, c=data.reshape(-1), s=10, cmap='coolwarm', alpha=0.8, norm=v_norm)
            ax.set_axisbelow(True)  # Make the scatter plot overlay the grid
            ax.set_aspect('equal')  # Set the aspect ratio of the plot to be equal
            # 设置x轴和y轴的字体大小
            ax.tick_params(axis='both', labelsize=14)
            # 设置x轴和y轴的标签和形状
            ax.set_xlabel('X (m)', fontsize=14, labelpad=10)
            # ax.set_ylabel('Y-axis', fontsize=14, labelpad=10)
            ax.set_xticks([-400, -200, 0, 200, 400])
            ax.set_yticks([-400, -200, 0, 200, 400])
            ax.set_ylim(-400, 400)  # 标签范围为[0, 5000)
            ax.set_xlim(-400, 400)  # 标签范围为[0, 5000)
            ax.set_title(title, fontsize=14)  # 添加标题
            ax.text(-500, 500, f'({i + 1})', fontsize=16)  # 在左上角添加序号

        # 添加一个垂直的y轴标签
        fig.text(0.075, 0.5, 'Y (m)', va='center', rotation='vertical', fontsize=14)
        # 在图形上添加一个颜色条
        cbar = fig.colorbar(c1, ax=[axs[0], axs[1]], shrink=0.7, aspect=25)
        cbar.ax.tick_params(labelsize=12)
        # cbar.set_label('u\u2032u\u2032', size=16)
        cbar.set_label('U / U_avg', size=14)
        cbar1 = fig.colorbar(c2, ax=[axs[2]], shrink=0.7, aspect=25)
        cbar1.ax.tick_params(labelsize=12)
        # cbar.set_label('u\u2032u\u2032', size=16)
        cbar1.set_label('$\Delta$U / U_avg', size=14)
        # plt.show()
        plt.savefig(
            './flow_gif_make/' + "Height_" + "{:.1f}".format(height_value) + '_u_error_' + 'time' + "{:.1f}".format(
                time_lists[t]) + '.png')
        plt.close('all')
    gif_images = []
    for select_time in time_lists:
        gif_images.append(imageio.imread(
            './flow_gif_make/' + "Height_" + "{:.1f}".format(height_value) + '_u_error_' + 'time' + "{:.1f}".format(
                select_time) + '.png'))
    imageio.mimsave(('U_error_on_selected_height_' + "{:.1f}".format(height_value) + '.gif'), gif_images,
                    fps=10)
    # '''画误差随时间变化图'''
    # fig = plt.figure(figsize=(6, 5.5))
    # plt.rc('font', family='Times New Roman')
    # ax = plt.subplot(111)
    # plt.title("Error in Flow Field Reconstruction at Different Time Interval",
    #           fontdict={'weight': 'normal', 'size': 14}, family="Times New Roman")  # 改变图标题字体
    # plt.xlabel('t (s)', fontdict={'weight': 'normal', 'size': 14}, family="Times New Roman")  # 改变坐标轴标题字体
    # plt.ylabel('<$\Delta$U / U_avg> ', fontdict={'weight': 'normal', 'size': 14}, family="Times New Roman")  # 改变坐标轴标题字体
    # plt.yticks(fontproperties='Times New Roman', size=12)
    # plt.xticks(fontproperties='Times New Roman', size=12)
    # plt.plot(time_lists, error_umag_avg)
    # plt.plot(time_lists, error_umag_avg_0)
    # plt.plot(time_lists, error_umag_avg_200)
    # plt.plot(time_lists, error_umag_avg_200n)
    # ax.set_ylim(0, 0.08)  # 标签范围为[0, 5000)
    # ax.set_xlim(0, 200)  # 标签范围为[0, 5000)
    # plt.show()

    return time_lists.reshape(-1, 1), error_umag_avg.reshape(-1, 1), error_umag_avg_rmse.reshape(-1, 1), \
           error_umag_t.mean().reshape(-1, 1)


def fluctuation_uu(LES_avg_u_t, NN_avg_u_t, LES_avg_v_t, NN_avg_v_t, LES_avg_w_t, NN_avg_w_t, time_steps, height_value):
    time_lists = np.linspace(0, 199.5, time_steps)
    NN_uu_avg = np.zeros(time_steps)
    LES_uu_avg = np.zeros(time_steps)
    Error_uu_avg = np.zeros(time_steps)
    for t in range(len(time_lists)):
        path = './LES_data/WD_turth_plane_100m/plane_' + str(int(time_lists[t] * 2)) + '.csv'
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
            (2 * (time_lists[t] - feature_mat_numpy[1, 3]) / (feature_mat_numpy[0, 3] - feature_mat_numpy[1, 3]) - 1),
            x_nom.shape[0])).reshape(-1, 1)
        x_selected = torch.tensor(x_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        y_selected = torch.tensor(y_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        z_selected = torch.tensor(z_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        t_selected = torch.tensor(t_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        u_predict, v_predict, w_predict, p_predict = f_equation_identification_3D(x_selected, y_selected, z_selected,
                                                                                  t_selected, pinn_net)
        NN_u = u_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_v = v_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_w = w_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_vr = np.sqrt(NN_u ** 2 + NN_v ** 2 + NN_w ** 2)
        LES_vr = np.sqrt(LES_data[:, 3] ** 2 + LES_data[:, 4] ** 2 + LES_data[:, 5] ** 2)
        error = NN_vr - LES_vr
        # NN_vr_2D = NN_vr[:6400].reshape(80, 80) - NN_enavg
        # LES_vr_2D = LES_vr[:6400].reshape(80, 80) - LES_enavg
        NN_u_2D = NN_u[:6400].reshape(80, 80) - NN_avg_u_t
        LES_u_2D = LES_data[:, 3][:6400].reshape(80, 80) - LES_avg_u_t
        NN_v_2D = NN_v[:6400].reshape(80, 80) - NN_avg_v_t
        LES_v_2D = LES_data[:, 4][:6400].reshape(80, 80) - LES_avg_v_t
        NN_w_2D = NN_w[:6400].reshape(80, 80) - NN_avg_w_t
        LES_w_2D = LES_data[:, 5][:6400].reshape(80, 80) - LES_avg_w_t
        NN_avg_umag = np.sqrt(NN_avg_u_t ** 2 + NN_avg_v_t ** 2 + NN_avg_w_t ** 2)
        LES_avg_umag = np.sqrt(LES_avg_u_t ** 2 + LES_avg_v_t ** 2 + LES_avg_w_t ** 2)
        U_inf = 8
        co_avg_energy = U_inf ** 2
        TKE_NN = 0.5 * (NN_u_2D ** 2 + NN_v_2D ** 2 + NN_w_2D ** 2)
        TKE_LES = 0.5 * (LES_u_2D ** 2 + LES_v_2D ** 2 + LES_w_2D ** 2)
        uu_nn = NN_u_2D ** 2 / co_avg_energy
        uu_les = LES_u_2D ** 2 / co_avg_energy
        error_umag_2D = np.abs(uu_nn - uu_les)
        NN_uu_avg[t] = uu_nn.mean()
        LES_uu_avg[t] = uu_les.mean()
        Error_uu_avg[t] = np.abs(NN_uu_avg[t] - LES_uu_avg[t])

        '''画图du_dx,du_dy'''
        # 计算du/dx
        # 创建一个新的图形和三个子图
        # 设置全局字体
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.75)
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        min_value = 0.001
        max_value = 0.07
        min_value2 = 0.001
        max_value2 = 0.07
        xx = np.linspace(-400, 400, 80)
        yy = np.linspace(-400, 400, 80)
        x_p, y_p = np.meshgrid(xx, yy)
        v_norm = mpl.colors.LogNorm(vmin=min_value, vmax=max_value)
        v_norm2 = mpl.colors.LogNorm(vmin=min_value2, vmax=max_value2)
        # 在每个子图上创建一个散点图
        titles = ['Prediction ' + 't = ' + "{:.1f}".format(time_lists[t]) + ' s',
                  'True ' + 't = ' + "{:.1f}".format(time_lists[t]) + ' s',
                  'Error ' + 't = ' + "{:.1f}".format(time_lists[t]) + ' s']  # 这里添加你的标题
        for i, (ax, data, title) in enumerate(zip(axs, [(uu_nn).reshape(-1),
                                                        (uu_les).reshape(-1),
                                                        (error_umag_2D).reshape(-1)], titles)):
            if i == 2:
                c2 = ax.scatter(x_p, y_p, c=data.reshape(-1), s=10, cmap='coolwarm', alpha=0.8, norm=v_norm2)
                ax.set_ylabel('Y (m)', fontsize=14, labelpad=10)
            else:
                c1 = ax.scatter(x_p, y_p, c=data.reshape(-1), s=10, cmap='coolwarm', alpha=0.8, norm=v_norm)
            ax.set_axisbelow(True)  # Make the scatter plot overlay the grid
            ax.set_aspect('equal')  # Set the aspect ratio of the plot to be equal
            # 设置x轴和y轴的字体大小
            ax.tick_params(axis='both', labelsize=14)
            # 设置x轴和y轴的标签和形状
            ax.set_xlabel('X (m)', fontsize=14, labelpad=10)
            # ax.set_ylabel('Y-axis', fontsize=14, labelpad=10)
            ax.set_xticks([-400, -200, 0, 200, 400])
            ax.set_yticks([-400, -200, 0, 200, 400])
            ax.set_ylim(-400, 400)  # 标签范围为[0, 5000)
            ax.set_xlim(-400, 400)  # 标签范围为[0, 5000)
            ax.set_title(title, fontsize=14)  # 添加标题
            ax.text(-500, 500, f'({i + 1})', fontsize=16)  # 在左上角添加序号

        # 添加一个垂直的y轴标签
        fig.text(0.075, 0.5, 'Y (m)', va='center', rotation='vertical', fontsize=14)
        # 在图形上添加一个颜色条
        cbar = fig.colorbar(c1, ax=[axs[0], axs[1]], shrink=0.7, aspect=25)
        cbar.ax.tick_params(labelsize=12)
        # cbar.set_label('u\u2032u\u2032', size=16)
        cbar.set_label('$u\u2032u\u2032 /U^2$', size=14)
        cbar1 = fig.colorbar(c2, ax=[axs[2]], shrink=0.7, aspect=25)
        cbar1.ax.tick_params(labelsize=12)
        # cbar.set_label('u\u2032u\u2032', size=16)
        cbar1.set_label('$\Delta(u\u2032u\u2032) / U^2$', size=14)
        # plt.show()
        plt.savefig(
            './flow_gif_make/' + "Height_" + "{:.1f}".format(height_value) + '_uu_error_' + 'time' + "{:.1f}".format(
                time_lists[t]) + '.png')
        plt.close('all')
    gif_images = []
    for select_time in time_lists:
        gif_images.append(imageio.imread(
            './flow_gif_make/' + "Height_" + "{:.1f}".format(height_value) + '_uu_error_' + 'time' + "{:.1f}".format(
                select_time) + '.png'))
    imageio.mimsave(('uu_error_on_selected_height_' + "{:.1f}".format(height_value) + '.gif'), gif_images,
                    fps=10)
    '''画误差随时间变化图'''
    fig = plt.figure(figsize=(6, 5.5))
    plt.rc('font', family='Times New Roman')
    ax = plt.subplot(111)
    plt.title("Error in Flow Field Reconstruction at Different Time Interval",
              fontdict={'weight': 'normal', 'size': 14}, family="Times New Roman")  # 改变图标题字体
    plt.xlabel('t (s)', fontdict={'weight': 'normal', 'size': 14}, family="Times New Roman")  # 改变坐标轴标题字体
    plt.ylabel('<$\Delta$U / U_avg> ', fontdict={'weight': 'normal', 'size': 14}, family="Times New Roman")  # 改变坐标轴标题字体
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.plot(time_lists, NN_uu_avg)
    plt.plot(time_lists, LES_uu_avg)
    plt.plot(time_lists, Error_uu_avg)
    ax.set_ylim(0, 0.015)  # 标签范围为[0, 5000)
    ax.set_xlim(0, 200)  # 标签范围为[0, 5000)
    plt.show()

    return time_lists.reshape(-1, 1), NN_uu_avg.reshape(-1, 1), LES_uu_avg.reshape(-1, 1), Error_uu_avg.reshape(-1, 1)


def fluctuation_TKE(LES_avg_u_t, NN_avg_u_t, LES_avg_v_t, NN_avg_v_t, LES_avg_w_t, NN_avg_w_t, time_steps,
                    height_value):
    time_lists = np.linspace(0, 199.5, time_steps)
    NN_TKE_avg = np.zeros(time_steps)
    LES_TKE_avg = np.zeros(time_steps)
    Error_TKE_avg = np.zeros(time_steps)
    for t in range(len(time_lists)):
        path = './LES_data/WD_turth_plane_100m/plane_' + str(int(time_lists[t] * 2)) + '.csv'
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
            (2 * (time_lists[t] - feature_mat_numpy[1, 3]) / (feature_mat_numpy[0, 3] - feature_mat_numpy[1, 3]) - 1),
            x_nom.shape[0])).reshape(-1, 1)
        x_selected = torch.tensor(x_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        y_selected = torch.tensor(y_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        z_selected = torch.tensor(z_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        t_selected = torch.tensor(t_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        u_predict, v_predict, w_predict, p_predict = f_equation_identification_3D(x_selected, y_selected, z_selected,
                                                                                  t_selected, pinn_net)
        NN_u = u_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_v = v_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_w = w_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_vr = np.sqrt(NN_u ** 2 + NN_v ** 2 + NN_w ** 2)
        LES_vr = np.sqrt(LES_data[:, 3] ** 2 + LES_data[:, 4] ** 2 + LES_data[:, 5] ** 2)
        error = NN_vr - LES_vr
        # NN_vr_2D = NN_vr[:6400].reshape(80, 80) - NN_enavg
        # LES_vr_2D = LES_vr[:6400].reshape(80, 80) - LES_enavg
        NN_u_2D = NN_u[:6400].reshape(80, 80) - NN_avg_u_t
        LES_u_2D = LES_data[:, 3][:6400].reshape(80, 80) - LES_avg_u_t
        NN_v_2D = NN_v[:6400].reshape(80, 80) - NN_avg_v_t
        LES_v_2D = LES_data[:, 4][:6400].reshape(80, 80) - LES_avg_v_t
        NN_w_2D = NN_w[:6400].reshape(80, 80) - NN_avg_w_t
        LES_w_2D = LES_data[:, 5][:6400].reshape(80, 80) - LES_avg_w_t
        NN_avg_umag = np.sqrt(NN_avg_u_t ** 2 + NN_avg_v_t ** 2 + NN_avg_w_t ** 2)
        LES_avg_umag = np.sqrt(LES_avg_u_t ** 2 + LES_avg_v_t ** 2 + LES_avg_w_t ** 2)
        U_inf = 8
        co_avg_energy = U_inf ** 2
        TKE_NN = 0.5 * (NN_u_2D ** 2 + NN_v_2D ** 2 + NN_w_2D ** 2) / co_avg_energy
        TKE_LES = 0.5 * (LES_u_2D ** 2 + LES_v_2D ** 2 + LES_w_2D ** 2) / co_avg_energy
        error_umag_2D = np.abs(TKE_NN - TKE_LES)
        NN_TKE_avg[t] = TKE_NN.mean()
        LES_TKE_avg[t] = TKE_LES.mean()
        Error_TKE_avg[t] = np.abs(NN_TKE_avg[t] - LES_TKE_avg[t])
        '''画图du_dx,du_dy'''
        # 计算du/dx
        # 创建一个新的图形和三个子图
        # 设置全局字体
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.75)
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        min_value = 0.001
        max_value = 0.03
        min_value2 = 0.001
        max_value2 = 0.03
        xx = np.linspace(-400, 400, 80)
        yy = np.linspace(-400, 400, 80)
        x_p, y_p = np.meshgrid(xx, yy)
        v_norm = mpl.colors.LogNorm(vmin=min_value, vmax=max_value)
        v_norm2 = mpl.colors.LogNorm(vmin=min_value2, vmax=max_value2)
        # 在每个子图上创建一个散点图
        titles = ['Prediction ' + 't = ' + "{:.1f}".format(time_lists[t]) + ' s',
                  'True ' + 't = ' + "{:.1f}".format(time_lists[t]) + ' s',
                  'Error ' + 't = ' + "{:.1f}".format(time_lists[t]) + ' s']  # 这里添加你的标题
        for i, (ax, data, title) in enumerate(zip(axs, [(TKE_NN).reshape(-1),
                                                        (TKE_LES).reshape(-1),
                                                        (error_umag_2D).reshape(-1)], titles)):
            if i == 2:
                c2 = ax.scatter(x_p, y_p, c=data.reshape(-1), s=10, cmap='coolwarm', alpha=0.8, norm=v_norm2)
                ax.set_ylabel('Y (m)', fontsize=14, labelpad=10)
            else:
                c1 = ax.scatter(x_p, y_p, c=data.reshape(-1), s=10, cmap='coolwarm', alpha=0.8, norm=v_norm)
            ax.set_axisbelow(True)  # Make the scatter plot overlay the grid
            ax.set_aspect('equal')  # Set the aspect ratio of the plot to be equal
            # 设置x轴和y轴的字体大小
            ax.tick_params(axis='both', labelsize=14)
            # 设置x轴和y轴的标签和形状
            ax.set_xlabel('X (m)', fontsize=14, labelpad=10)
            # ax.set_ylabel('Y-axis', fontsize=14, labelpad=10)
            ax.set_xticks([-400, -200, 0, 200, 400])
            ax.set_yticks([-400, -200, 0, 200, 400])
            ax.set_ylim(-400, 400)  # 标签范围为[0, 5000)
            ax.set_xlim(-400, 400)  # 标签范围为[0, 5000)
            ax.set_title(title, fontsize=14)  # 添加标题
            ax.text(-500, 500, f'({i + 1})', fontsize=16)  # 在左上角添加序号

        # 添加一个垂直的y轴标签
        fig.text(0.075, 0.5, 'Y (m)', va='center', rotation='vertical', fontsize=14)
        # 在图形上添加一个颜色条
        cbar = fig.colorbar(c1, ax=[axs[0], axs[1]], shrink=0.7, aspect=25)
        cbar.ax.tick_params(labelsize=12)
        # cbar.set_label('u\u2032u\u2032', size=16)
        cbar.set_label('$TKE/U^2$', size=14)
        cbar1 = fig.colorbar(c2, ax=[axs[2]], shrink=0.7, aspect=25)
        cbar1.ax.tick_params(labelsize=12)
        # cbar.set_label('u\u2032u\u2032', size=16)
        cbar1.set_label('$\Delta$$ TKE $$/U^2$', size=14)
        # plt.show()
        plt.savefig(
            './flow_gif_make/' + "Height_" + "{:.1f}".format(height_value) + '_TKE_error_' + 'time' + "{:.1f}".format(
                time_lists[t]) + '.png')
        plt.close('all')
    gif_images = []
    for select_time in time_lists:
        gif_images.append(imageio.imread(
            './flow_gif_make/' + "Height_" + "{:.1f}".format(height_value) + '_TKE_error_' + 'time' + "{:.1f}".format(
                select_time) + '.png'))
    imageio.mimsave(('TKE_error_on_selected_height_' + "{:.1f}".format(height_value) + '.gif'), gif_images,
                    fps=10)
    return time_lists.reshape(-1, 1), NN_TKE_avg.reshape(-1, 1), LES_TKE_avg.reshape(-1, 1), Error_TKE_avg.reshape(-1,
                                                                                                                   1)


def energy_spectrum(LES_avg_u_t, NN_avg_u_t, LES_avg_v_t, NN_avg_v_t, LES_avg_w_t, NN_avg_w_t, time_steps,
                    height_value):
    time_lists = np.linspace(0, 199.5, time_steps)
    energy_spectrum_avg_les_u = np.zeros((time_steps, 40))
    energy_spectrum_avg_nn_u = np.zeros((time_steps, 40))
    energy_spectrum_avg_les_v = np.zeros((time_steps, 40))
    energy_spectrum_avg_nn_v = np.zeros((time_steps, 40))
    for t in range(len(time_lists)):
        path = './LES_data/WD_turth_plane_100m/plane_' + str(int(time_lists[t] * 2)) + '.csv'
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
            (2 * (time_lists[t] - feature_mat_numpy[1, 3]) / (feature_mat_numpy[0, 3] - feature_mat_numpy[1, 3]) - 1),
            x_nom.shape[0])).reshape(-1, 1)
        x_selected = torch.tensor(x_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        y_selected = torch.tensor(y_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        z_selected = torch.tensor(z_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        t_selected = torch.tensor(t_nom.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
        u_predict, v_predict, w_predict, p_predict = f_equation_identification_3D(x_selected, y_selected, z_selected,
                                                                                  t_selected, pinn_net)
        NN_u = u_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_v = v_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_w = w_predict.data.numpy().reshape(x_nom.shape[0], ) * 10
        NN_vr = np.sqrt(NN_u ** 2 + NN_v ** 2 + NN_w ** 2)
        LES_vr = np.sqrt(LES_data[:, 3] ** 2 + LES_data[:, 4] ** 2 + LES_data[:, 5] ** 2)
        error = NN_vr - LES_vr
        NN_vr_2D = NN_vr[:6400].reshape(80, 80)
        LES_vr_2D = LES_vr[:6400].reshape(80, 80)
        NN_u_2D = NN_u[:6400].reshape(80, 80) - NN_avg_u_t
        LES_u_2D = LES_data[:, 3][:6400].reshape(80, 80) - LES_avg_u_t
        NN_v_2D = NN_v[:6400].reshape(80, 80) - NN_avg_v_t
        LES_v_2D = LES_data[:, 4][:6400].reshape(80, 80) - LES_avg_v_t
        NN_w_2D = NN_w[:6400].reshape(80, 80) - NN_avg_w_t
        LES_w_2D = LES_data[:, 5][:6400].reshape(80, 80) - LES_avg_w_t
        fft_NN_vr_2D = np.fft.fft2(NN_vr_2D)
        fft_LES_vr_2D = np.fft.fft2(LES_vr_2D)
        fft_NN_u_2D = np.fft.fft2(NN_u_2D)
        fft_LES_u_2D = np.fft.fft2(LES_u_2D)
        fft_NN_v_2D = np.fft.fft2(NN_v_2D)
        fft_LES_v_2D = np.fft.fft2(LES_v_2D)
        fft_NN_w_2D = np.fft.fft2(NN_w_2D)
        fft_LES_w_2D = np.fft.fft2(LES_w_2D)
        # energy_spectrum_nn = np.abs(fft_NN_u_2D) ** 2 + np.abs(fft_NN_v_2D) ** 2 + np.abs(fft_NN_w_2D) ** 2
        # energy_spectrum_les = np.abs(fft_LES_u_2D) ** 2 + np.abs(fft_LES_v_2D) ** 2 + np.abs(fft_LES_w_2D) ** 2
        energy_spectrum_nn_u = np.abs(fft_NN_u_2D) ** 2
        energy_spectrum_les_u = np.abs(fft_LES_u_2D) ** 2
        energy_spectrum_nn_v = np.abs(fft_NN_v_2D) ** 2
        energy_spectrum_les_v = np.abs(fft_LES_v_2D) ** 2
        energy_spectrum_avg_les_u[t, :] = (energy_spectrum_les_u.mean(axis=1))[:40]
        energy_spectrum_avg_nn_u[t, :] = (energy_spectrum_nn_u.mean(axis=1))[:40]
        energy_spectrum_avg_les_v[t, :] = (energy_spectrum_les_v.mean(axis=1))[:40]
        energy_spectrum_avg_nn_v[t, :] = (energy_spectrum_nn_v.mean(axis=1))[:40]
        # xx = np.linspace(200, 1000, 80)
        # yy = np.linspace(-400, 400, 80)
        # x_p, y_p = np.meshgrid(xx, yy)
        # min_value = 0.1
        # max_value = 1000
        # v_norm = mpl.colors.LogNorm(vmin=min_value, vmax=max_value)
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        # c1 = ax.scatter(x_p, y_p, c=energy_spectrum_les.reshape(-1), s=10, cmap='coolwarm', alpha=0.8, norm=v_norm)

    fig = plt.figure(figsize=(6, 5.5))
    energy_x = np.linspace(1, 40, 40)
    energy_35 = 40000 * energy_x ** (-5 / 3)
    print(energy_35)
    plt.rc("font", family="Times New Roman")
    plt.plot(energy_x, energy_spectrum_avg_nn_u.mean(axis=0), label='Energy_spectrum_NN')
    plt.plot(energy_x, energy_spectrum_avg_les_u.mean(axis=0), label='Energy_spectrum_LES')
    plt.plot(energy_x[3:-10], energy_35[3:-10], label='-5/3')
    plt.title("Energy spectrum u", fontdict={'weight': 'normal', 'size': 16}, family="Times New Roman")  # 改变图标题字体
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.xlabel('k', fontdict={'weight': 'normal', 'size': 14}, family="Times New Roman")
    # plt.plot(energy_x, energy_spectrum_nn.mean(axis=0) / energy_spectrum_les.mean(axis=0))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, 40)
    plt.ylim(10e-1, 10e5)
    plt.legend(prop={'size': 12})
    plt.show()

    fig = plt.figure(figsize=(6, 5.5))
    energy_x = np.linspace(1, 40, 40)
    energy_35 = 40000 * energy_x ** (-5 / 3)
    print(energy_35)
    plt.rc("font", family="Times New Roman")
    plt.plot(energy_x, energy_spectrum_avg_nn_v.mean(axis=0), label='Energy_spectrum_NN')
    plt.plot(energy_x, energy_spectrum_avg_les_v.mean(axis=0), label='Energy_spectrum_LES')
    plt.plot(energy_x[3:-10], energy_35[3:-10], label='-5/3')
    plt.title("Energy spectrum v", fontdict={'weight': 'normal', 'size': 16}, family="Times New Roman")  # 改变图标题字体
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.xlabel('k', fontdict={'weight': 'normal', 'size': 14}, family="Times New Roman")
    # plt.plot(energy_x, energy_spectrum_nn.mean(axis=0) / energy_spectrum_les.mean(axis=0))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, 40)
    plt.ylim(10e-1, 10e5)
    plt.legend(prop={'size': 12})
    plt.show()

    return energy_spectrum_avg_les_u.mean(0).reshape(-1, 1), energy_spectrum_avg_nn_u.mean(0).reshape(-1, 1), \
           energy_spectrum_avg_les_v.mean(0).reshape(-1, 1), energy_spectrum_avg_nn_v.mean(0).reshape(-1, 1)


# def JS_div(p, q):
#     M = (p + q) / 2
#     return 0.5 * st.entropy(p, M, base=2) + 0.5 * st.entropy(q, M, base=2)
#


selected_height = 100
time_lists, error_umag_avg, error_umag_avg_rmse, mse_total = fluctuation_u(num_time,height_value=100)
# time_lists, NN_uu_avg, LES_uu_avg, Error_uu_avg = fluctuation_uu(LES_avg_u_t, NN_avg_u_t, LES_avg_v_t, NN_avg_v_t,
#                                                                  LES_avg_w_t, NN_avg_w_t, num_time, selected_height)
# time_lists, NN_TKE_avg, LES_TKE_avg, Error_TKE_avg = fluctuation_TKE(LES_avg_u_t, NN_avg_u_t, LES_avg_v_t, NN_avg_v_t,
#                                                                   LES_avg_w_t, NN_avg_w_t, num_time, selected_height)
# energy_spectrum_avg_les_u, energy_spectrum_avg_nn_u, energy_spectrum_avg_les_v, energy_spectrum_avg_nn_v \
#     = energy_spectrum(LES_avg_u_t, NN_avg_u_t, LES_avg_v_t, NN_avg_v_t, LES_avg_w_t, NN_avg_w_t, num_time,
#                       selected_height)
# print(LES_avg_u_t, NN_avg_u_t, LES_avg_v_t, NN_avg_v_t, LES_avg_w_t, NN_avg_w_t, error_umean_t)
print(error_umag_avg_rmse)
# plane_error_at_whole_time(num_time, feature_mat_numpy)
# print(pinn_net.Re_nn)
time = np.linspace(0, 199.5, 100)

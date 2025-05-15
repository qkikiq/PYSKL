import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from learner import Learner
from copy import deepcopy


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr  # 内部学习率
        self.meta_lr = args.meta_lr  # 外部学习率
        self.n_way = args.n_way  # 类别数
        self.k_spt = args.k_spt  # 样本数
        self.k_qry = args.k_qry  # 查询数
        self.task_num = args.task_num  # 任务数
        self.update_step = args.update_step  # 内循环更新步数
        self.update_step_test = args.update_step_test  # 外循环更新步数

        self.net = Learner(config, args.imgc, args.imgsz)  # learner网络
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)  # 外部优化器

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        原地（in-place）梯度裁剪。
        :param grad: 梯度列表
        :param max_norm: 允许的最大梯度范数
        :return: 平均梯度范数
        """

        total_norm = 0  # 初始化总梯度范数为0
        counter = 0  # 初始化计数器，用于记录梯度的数量

        # 遍历每个梯度
        for g in grad:
            param_norm = g.data.norm(2)  # 计算梯度的L2范数（欧几里得范数）

            total_norm += param_norm.item() ** 2  # 累加梯度范数的平方
            counter += 1  # 计数器加1

        total_norm = total_norm ** (1. / 2)  # 计算所有梯度平方和的平方根（即总L2范数）

        clip_coef = max_norm / (total_norm + 1e-6)  # 计算裁剪系数，避免除以0
        if clip_coef < 1:  # 如果裁剪系数小于1，说明需要裁剪
            for g in grad:
                g.data.mul_(clip_coef)  # 原地缩放梯度，乘以裁剪系数

        return total_norm / counter  # 返回平均梯度范数

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w] - 支持集输入数据
    :param y_spt:   [b, setsz] - 支持集标签
    :param x_qry:   [b, querysz, c_, h, w] - 查询集输入数据
    :param y_qry:   [b, querysz] - 查询集标签
    :return: 每次更新后的准确率
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # 初始化 损失和正确率列表
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        # 遍历每个任务
        for i in range(task_num):
            # 第一步：计算第i个任务在k=0时的损失和梯度
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)  # 支持集前向传播
            loss = F.cross_entropy(logits, y_spt[i])  # 计算交叉熵损失
            grad = torch.autograd.grad(loss, self.net.parameters())  # 计算梯度
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))  # 使用梯度和学习率更新权重
            # 在第一次更新前评估模型
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)  # 查询集前向传播
                loss_q = F.cross_entropy(logits_q, y_qry[i])  # 查询集损失
                losses_q[0] += loss_q  # 累加损失

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # 获取预测结果
                correct = torch.eq(pred_q, y_qry[i]).sum().item()  # 计算正确预测数
                corrects[0] = corrects[0] + correct  # 累加正确预测数
            # 在第一次更新后评估模型
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
            # 执行多次更新
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)  # 支持集前向传播
                loss = F.cross_entropy(logits, y_spt[i])  # 计算交叉熵损失
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)  # 计算梯度
                # 3. theta_pi = theta_pi - train_lr * grad # 使用梯度和学习率更新权重
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])  # 查询集损失
                losses_q[k + 1] += loss_q  # 累加损失

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # 获取预测结果
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy   计算正确预测数
                    corrects[k + 1] = corrects[k + 1] + correct  # 累加正确预测数

        # 计算所有任务的查询集平均损失
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # 优化元参数
        # optimize theta parameters
        self.meta_optim.zero_grad()  # 清空梯度
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()  # 更新元参数

        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
    微调函数，用于在支持集上进行模型更新，并在查询集上评估性能。

    :param x_spt:   [setsz, c_, h, w] - 支持集输入数据
    :param y_spt:   [setsz] - 支持集标签
    :param x_qry:   [querysz, c_, h, w] - 查询集输入数据
    :param y_qry:   [querysz] - 查询集标签
    :return: 每次更新后的准确率
        """
        # 确保支持集输入数据的维度正确
        assert len(x_spt.shape) == 4
        # 获取查询集的大小
        querysz = x_qry.size(0)
        # 初始化一个列表，用于存储每次更新后的正确预测数
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # 为了不影响模型的运行均值/方差和批归一化参数，复制模型进行微调
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 第一步：在支持集上计算初始损失和梯度
        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)  # 支持集前向传播
        loss = F.cross_entropy(logits, y_spt)  # 计算交叉熵损失
        grad = torch.autograd.grad(loss, net.parameters())  # 计算梯度
        # 使用梯度和学习率更新权重
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # 在第一次更新前评估模型
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)  # 查询集前向传播
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # 获取预测结果
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()  # 计算正确预测数
            corrects[0] = corrects[0] + correct  # 记录第0步的正确预测数

        # 在第一次更新后评估模型
        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)  # 使用更新后的权重进行前向传播
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # 获取预测结果
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()  # 计算正确预测数
            corrects[1] = corrects[1] + correct  # 记录第1步的正确预测数

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)  # 支持集前向传播
            loss = F.cross_entropy(logits, y_spt)  # 计算交叉熵损失
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            # 使用梯度和学习率更新权重
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)  # 查询集前向传播
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)  # 查询集损失

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # 获取预测结果
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy# 计算正确预测数
                corrects[k + 1] = corrects[k + 1] + correct  # 记录第k+1步的正确预测数

        # 删除临时模型以释放内存
        del net
        # 计算每次更新后的准确率
        accs = np.array(corrects) / querysz

        return accs


def main():
    pass


if __name__ == '__main__':
    main()

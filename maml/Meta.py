from copy import deepcopy  # 保持这个
# from torch.utils.data import TensorDataset, DataLoader # 可能不需要，取决于数据加载方式
import numpy as np
import torch
from torch import nn
from torch import optim

from maml.learner import Learner


class Meta(nn.Module):
    """
    Meta Learner (适用于骨架数据和LDA损失)
    """

    def __init__(self, args, learner_args):  # config 参数可能不再直接使用，改为传递learner_args
        """
        :param args: MAML相关的通用参数 (meta_lr, update_lr, task_num, n_way, k_spt, k_qry, update_step等)
        :param learner_args: 传递给 NewLearner 的特定参数字典
                             (例如: input_features, mlp_hidden, gcn_hidden, output_embedding_dim, num_joints 等)
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way  # N-way 分类
        self.k_spt = args.k_spt  # K-shot 支持样本
        self.k_qry = args.k_qry  # K-shot 查询样本 (每个类)
        self.task_num = args.task_num  # Meta-batch size
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        # 实例化新的 Learner
        # self.net = NewLearner(**learner_args) # 使用 ** 解包字典参数
        # 示例，你需要用你实际的 NewLearner 替换
        # 为了让代码能跑通，这里用一个简单的占位符，实际中你需要替换
        # self.net = nn.Linear(learner_args.get('input_dim', 784), self.n_way) # 这是一个非常简化的占位符
        self.net = Learner(**learner_args)

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):  # 这个函数可以保持不变
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            if g is None:  # 梯度可能为None，如果某些参数没有参与计算
                continue
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        if counter == 0:
            return 0
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                if g is None:
                    continue
                g.data.mul_(clip_coef)
        return total_norm / counter

    def forward(self, x_spt, y_spt, adj_spt, x_qry, y_qry, adj_qry):  # 增加了adj_spt, adj_qry参数
        """
        元训练步骤
        :param x_spt:   [b, setsz_spt, feature_dim] 或 [b, setsz_spt, num_nodes, node_feature_dim] (取决于Learner输入)
        :param y_spt:   [b, setsz_spt] (标签)
        :param adj_spt: [b, setsz_spt, num_nodes, num_nodes] (支持集邻接矩阵)
        :param x_qry:   [b, setsz_qry, feature_dim] 或 [b, setsz_qry, num_nodes, node_feature_dim]
        :param y_qry:   [b, setsz_qry] (标签)
        :param adj_qry: [b, setsz_qry, num_nodes, num_nodes] (查询集邻接矩阵)
        :return: query set上的平均准确率 (在每个内部更新步骤之后)
        """
        # 获取任务数量(meta-batch size)，即有多少个独立的任务需要元学习
        task_num = x_spt.size(0)

        # 获取每个任务的查询集大小(样本数)
        querysz = x_qry.size(1)

        # 初始化列表存储每个更新步骤后的查询集损失
        losses_q = [0 for _ in range(self.update_step + 1)]
        # 初始化列表存储每个更新步骤后的查询集正确分类样本数
        corrects = [0 for _ in range(self.update_step + 1)]

        # 遍历每个任务(每个episodic task)
        for i in range(task_num):
            # 1. 初始评估  - 在任何参数更新前计算模型性能基准
            #    注意: 这里的self.net.parameters()是元参数theta
            #    BN层的training状态在MAML中通常保持True，因为它也参与内循环的适应，或者，如果BN层的统计数据不希望在内循环中改变，可以考虑更复杂的BN处理
            # 用当前参数获取支持集的嵌入表示
            spt_embeddings_for_eval = self.net(x_spt[i], adj_spt[i], vars=None, bn_training=True)  # 使用元参数
            with torch.no_grad():  # 不需要梯度计算，节省内存
                # 获取查询集的初始嵌入表示(用原始元参数)
                qry_embeddings_initial = self.net(x_qry[i], adj_qry[i], vars=self.net.parameters(), bn_training=True)


               #  #计算LDA损失 (可选，如果主要关心准确率)
               #  loss_q_initial = lda_loss(qry_embeddings_initial, y_qry[i])
               #  losses_q[0] += loss_q_initial
               # # 计算准确率
               #  spt_embeddings_initial = self.net(x_spt[i], adj_spt[i], vars=self.net.parameters(), bn_training=True)
               #  acc_initial = calculate_embedding_accuracy(qry_embeddings_initial, y_qry[i], spt_embeddings_initial, y_spt[i])
               #  corrects[0] += acc_initial * querysz # acc_initial 应该是0-1的值，乘以querysz得到正确数

            # 2. 内循环适应阶段 - 在当前任务上快速学习
            # a. 计算支持集上的损失，得到梯度
            spt_embeddings = self.net(x_spt[i], adj_spt[i], vars=None, bn_training=True)  # vars=None表示使用self.net的当前参数

            # 计算支持集的LDA损失
            loss = lda_loss(spt_embeddings, y_spt[i])

            # 计算损失对当前网络参数的梯度
            grad = torch.autograd.grad(loss, self.net.parameters())

            # 根据梯度更新得到fast_weights(临时参数)：θ' = θ - α∇θL
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1],
                                    zip(grad, self.net.parameters())))
            # fast_weights 是适应当前任务 i 的临时参数 theta_prime

            # 2b. 使用更新后的参数(fast_weights)在查询集上评估第一次更新后的效果
            with torch.no_grad():
                # 使用fast_weights获取查询集的嵌入表示
                qry_embeddings_step1 = self.net(x_qry[i], adj_qry[i], vars=fast_weights, bn_training=True)
                # 计算查询集上的LDA损失
                loss_q_step1 = lda_loss(qry_embeddings_step1, y_qry[i])
                # 累加第1步更新后的查询集损失
                losses_q[1] += loss_q_step1  # 累加第i个任务在第1步更新后的查询集损失

                # 获取原始参数下的支持集嵌入(用于准确率计算)
                spt_embeddings_step0_for_acc = self.net(x_spt[i], adj_spt[i], vars=self.net.parameters(),
                                                        bn_training=True)  # 用原始参数下的spt作为参考
                # 或者用更新后的spt嵌入：
                # spt_embeddings_step1_for_acc = self.net(x_spt[i], adj_spt[i], vars=fast_weights, bn_training=True)
                # acc_step1 = calculate_embedding_accuracy(qry_embeddings_step1, y_qry[i], spt_embeddings_step0_for_acc, y_spt[i])
                # corrects[1] += acc_step1 * querysz

            # 2c. 进行更多内循环更新步骤(第2步到第update_step步)
            for k in range(1, self.update_step):
                # 使用当前fast_weights计算支持集嵌入
                spt_embeddings_k = self.net(x_spt[i], adj_spt[i], vars=fast_weights, bn_training=True)

                # 计算当前支持集的LDA损失
                loss_k = lda_loss(spt_embeddings_k, y_spt[i])

                # 计算损失对fast_weights的梯度(注意这里是对fast_weights求导)
                grad_k = torch.autograd.grad(loss_k, fast_weights)  # 注意这里是对 fast_weights 求导

                # 更新fast_weights：θ'k+1 = θ'k - α∇θ'kL
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1],
                                        zip(grad_k, fast_weights)))

                # 使用更新后的fast_weights在查询集上评估
                qry_embeddings_k_plus_1 = self.net(x_qry[i], adj_qry[i], vars=fast_weights, bn_training=True)

                # 计算查询集上的LDA损失
                loss_q_k_plus_1 = lda_loss(qry_embeddings_k_plus_1, y_qry[i])

                # 累加当前步骤的查询集损失
                losses_q[k + 1] += loss_q_k_plus_1  # 累加第i个任务在第 k+1 步更新后的查询集损失

                # 计算准确率 (可选，但推荐用于监控)
                with torch.no_grad():
                    # spt_embeddings_k_for_acc = self.net(x_spt[i], adj_spt[i], vars=fast_weights, bn_training=True) # 使用当前fast_weights下的spt
                    # acc_k_plus_1 = calculate_embedding_accuracy(qry_embeddings_k_plus_1, y_qry[i], spt_embeddings_k_for_acc, y_spt[i])
                    # corrects[k + 1] += acc_k_plus_1 * querysz
                    pass  # 准确率计算部分需要你自己实现

        # 3. 外循环更新阶段(元优化) - 更新模型的元参数

        #    使用所有任务在最后一步内循环更新后 (self.update_step) 的查询集损失进行元优化
        loss_q_final = losses_q[self.update_step] / task_num  # 平均查询集损失

        self.meta_optim.zero_grad()
        # 反向传播计算梯度
        loss_q_final.backward()  # PyTorch会自动处理链式法则，将梯度反向传播到原始的self.net.parameters()
        # self.clip_grad_by_norm_(self.net.parameters(), 10) # 可选的梯度裁剪
        # 使用元优化器更新元参数
        self.meta_optim.step()

        # accs = np.array(corrects) / (querysz * task_num) # 计算平均准确率
        # return accs
        # 由于准确率计算方式改变，这里暂时返回最终损失，你可以根据你的准确率计算来修改
        # 注意：如果你没有在上面实现并累加corrects，那么accs的计算会出错
        # 实际应用中，你需要完整地实现正确的corrects累加
        dummy_accs = np.zeros(self.update_step + 1)  # 占位符

        # 对每个步骤，计算平均准确率(如果有正确分类的样本)
        for step_idx in range(self.update_step + 1):
            if (querysz * task_num) > 0 and corrects[step_idx] > 0:  # 确保分母不为0
                dummy_accs[step_idx] = corrects[step_idx] / (querysz * task_num)
            else:
                dummy_accs[step_idx] = 0  # 或者其他默认值

        return dummy_accs

    def finetunning(self, x_spt, y_spt, adj_spt, x_qry, y_qry, adj_qry):
        """
        在新任务上进行微调和评估 (测试阶段)
        :param x_spt:   [setsz_spt, feature_dim] 或 [setsz_spt, num_nodes, node_feature_dim]
        :param y_spt:   [setsz_spt]
        :param adj_spt: [setsz_spt, num_nodes, num_nodes]
        :param x_qry:   [setsz_qry, feature_dim] 或 [setsz_qry, num_nodes, node_feature_dim]
        :param y_qry:   [setsz_qry]
        :param adj_qry: [setsz_qry, num_nodes, num_nodes]
        :return: query set上的准确率数组 (在每个微调步骤之后)
        """
        # assert len(x_spt.shape) == 4 # 原始图像数据的断言，现在可能不适用

        querysz = x_qry.size(0)  # 注意这里没有 task_num 维度，因为finetunning通常针对单个任务
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # 深拷贝网络，避免修改原始元参数，BN层状态也一同拷贝
        net = deepcopy(self.net)

        # 1. 初始评估 (不更新)
        spt_embeddings_for_eval = net(x_spt, adj_spt, vars=None, bn_training=True)
        with torch.no_grad():
            qry_embeddings_initial = net(x_qry, adj_qry, vars=net.parameters(), bn_training=True)
            # acc_initial = calculate_embedding_accuracy(qry_embeddings_initial, y_qry, spt_embeddings_for_eval, y_spt)
            # corrects[0] = acc_initial * querysz

        # 2. 第一次微调更新
        spt_embeddings = net(x_spt, adj_spt, vars=None, bn_training=True)
        loss = lda_loss(spt_embeddings, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1], zip(grad, net.parameters())))

        # 评估第一次更新后
        with torch.no_grad():
            qry_embeddings_step1 = net(x_qry, adj_qry, vars=fast_weights, bn_training=True)
            # spt_embeddings_step0_for_acc = net(x_spt, adj_spt, vars=net.parameters(), bn_training=True)
            # acc_step1 = calculate_embedding_accuracy(qry_embeddings_step1, y_qry, spt_embeddings_step0_for_acc, y_spt)
            # corrects[1] = acc_step1 * querysz

        # 3. 后续微调步骤
        for k in range(1, self.update_step_test):
            spt_embeddings_k = net(x_spt, adj_spt, vars=fast_weights, bn_training=True)
            loss_k = lda_loss(spt_embeddings_k, y_spt)
            grad_k = torch.autograd.grad(loss_k, fast_weights)
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1], zip(grad_k, fast_weights)))

            with torch.no_grad():
                qry_embeddings_k_plus_1 = net(x_qry, adj_qry, vars=fast_weights, bn_training=True)
                # spt_embeddings_k_for_acc = net(x_spt, adj_spt, vars=fast_weights, bn_training=True)
                # acc_k_plus_1 = calculate_embedding_accuracy(qry_embeddings_k_plus_1, y_qry, spt_embeddings_k_for_acc, y_spt)
                # corrects[k + 1] = acc_k_plus_1 * querysz
                pass  # 准确率计算部分

        del net  # 释放拷贝的网络

        # accs = np.array(corrects) / querysz
        # return accs
        # 占位符返回
        dummy_accs = np.zeros(self.update_step_test + 1)
        for step_idx in range(self.update_step_test + 1):
            if querysz > 0 and corrects[step_idx] > 0:
                dummy_accs[step_idx] = corrects[step_idx] / querysz
            else:
                dummy_accs[step_idx] = 0
        return dummy_accs



def lda_loss(embeddings, labels):
    # embeddings: [batch_size, embedding_dim]
    # labels: [batch_size]
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    embedding_dim = embeddings.size(1)
    device = embeddings.device

    if num_classes < 2: # LDA至少需要两个类
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 计算类均值
    class_means = torch.zeros(num_classes, embedding_dim, device=device)
    for i, label_val in enumerate(unique_labels):
        class_means[i] = embeddings[labels == label_val].mean(dim=0)

    # 1. 计算类内散度 (S_W)
    s_w = torch.zeros(embedding_dim, embedding_dim, device=device)
    for i, label_val in enumerate(unique_labels):
        class_embeddings = embeddings[labels == label_val]
        mean_centered_embeddings = class_embeddings - class_means[i].unsqueeze(0)
        s_w += mean_centered_embeddings.t().mm(mean_centered_embeddings)

    # 2. 计算类间散度 (S_B)
    overall_mean = embeddings.mean(dim=0)
    s_b = torch.zeros(embedding_dim, embedding_dim, device=device)
    for i, label_val in enumerate(unique_labels):
        num_samples_class = (labels == label_val).sum()
        mean_diff = (class_means[i] - overall_mean).unsqueeze(1) # [embedding_dim, 1]
        s_b += num_samples_class * mean_diff.mm(mean_diff.t())

    # 为了数值稳定性，可以给S_W的对角线加上一个小的epsilon
    s_w += torch.eye(embedding_dim, device=device) * 1e-4

    # 目标是最大化 tr(S_W_inv * S_B)，所以损失是 -tr(S_W_inv * S_B)
    # 或者一个更简单的形式，如 Fisher's criterion J(W) = tr(S_B) / tr(S_W)
    # 或者直接用 -(tr(S_B) - tr(S_W)) 这样的形式引导
    # 这里使用 -tr(S_W_inv * S_B) 的一个简化版，如果S_W求逆困难，可以考虑其他形式
    # 例如，loss = torch.trace(s_w) - torch.trace(s_b)  (目标是最小化类内，最大化类间)
    # 或者 loss = torch.trace(s_w) / (torch.trace(s_b) + 1e-6)

    # 一个常用的LDA损失形式是最小化类内散度，最大化类间散度的某种组合
    # 例如： loss = trace(S_W) - gamma * trace(S_B)
    # gamma是一个超参数。我们希望S_W小，S_B大。
    # 所以，如果直接优化，loss_sw = torch.trace(s_w), loss_sb = -torch.trace(s_b)
    # loss = torch.trace(s_w)
    try:
        # 计算 S_W_inv * S_B
        s_w_inv_s_b = torch.linalg.solve(s_w, s_b) # 更稳定
        # s_w_inv_s_b = torch.linalg.inv(s_w).mm(s_b)
        loss = -torch.trace(s_w_inv_s_b)
    except torch.linalg.LinAlgError: # 如果S_W奇异
        # print("S_W is singular, using alternative LDA loss")
        loss = torch.trace(s_w) - torch.trace(s_b) # 备用损失

    if torch.isnan(loss) or torch.isinf(loss):
        # print("LDA loss is NaN or Inf, returning zero loss for this batch.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss


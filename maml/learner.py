# 示例 Learner.__init__ 结构
import torch
from torch import nn
# from torch_geometric.nn import GCNConv # 如果使用 PyG

class Learner(nn.Module):
    def __init__(self, input_features, mlp_hidden, gcn_hidden, output_embedding_dim, num_joints):
        super(Learner, self).__init__()
        self.input_features = input_features # 每个关节的特征维度
        self.mlp_hidden = mlp_hidden
        self.gcn_hidden = gcn_hidden
        self.output_embedding_dim = output_embedding_dim
        self.num_joints = num_joints # 关节数量

        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList() # 如果使用BN

        # 1. MLP层
        # 假设输入是 [batch, num_frames, num_joints, coords_dim]
        # 你可能需要先 Flatten 或做一些处理，这里简化为直接处理每个节点的特征
        # 例如，一个简单的线性层将原始关节特征映射到mlp_hidden
        # 注意：这里需要仔细设计如何将骨架数据输入到MLP和GCN
        # 假设我们先对每个关节的特征独立处理
        mlp_input_dim = input_features # 假设输入已经是每个节点的特征
        self.mlp_weight = nn.Parameter(torch.ones(mlp_hidden, mlp_input_dim))
        torch.nn.init.kaiming_normal_(self.mlp_weight)
        self.vars.append(self.mlp_weight)
        self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden))
        self.vars.append(self.mlp_bias)

        # 2. GCN层 (这里用一个简化的线性变换模拟GCN权重，实际GCN层更复杂)
        # 如果使用 torch_geometric.nn.GCNConv:
        # self.gcn_conv1 = GCNConv(mlp_hidden, gcn_hidden)
        # self.vars.append(self.gcn_conv1.lin.weight)
        # self.vars.append(self.gcn_conv1.bias)
        # 手动实现或简化GCN权重:
        self.gcn_weight = nn.Parameter(torch.ones(gcn_hidden, mlp_hidden))
        torch.nn.init.kaiming_normal_(self.gcn_weight)
        self.vars.append(self.gcn_weight)
        # GCN通常没有单独的偏置，或者偏置在聚合后添加

        # 3. 输出层 (得到用于LDA的嵌入)
        self.output_weight = nn.Parameter(torch.ones(output_embedding_dim, gcn_hidden)) # 假设GCN输出后直接接一个线性层得到最终嵌入
        torch.nn.init.kaiming_normal_(self.output_weight)
        self.vars.append(self.output_weight)
        self.output_bias = nn.Parameter(torch.zeros(output_embedding_dim))
        self.vars.append(self.output_bias)

        # 如果GCN后需要对整个图的表示进行池化（例如，得到一个固定大小的图级别嵌入）
        # 你可能还需要全局平均池化层或其他池化操作

    def forward(self, x, adj, vars_list=None, bn_training=True):
        # x: 节点特征 [batch_size, num_nodes, node_feature_dim] 或者更复杂的骨架数据
        # adj: 邻接矩阵 [batch_size, num_nodes, num_nodes] 或 PyG的edge_index
        # vars_list: MAML传入的参数列表，如果为None，则使用self.vars

        if vars_list is None:
            vars_list = self.vars

        # 解析参数
        idx = 0
        mlp_w, mlp_b = vars_list[idx], vars_list[idx+1]; idx+=2
        gcn_w = vars_list[idx]; idx+=1 # 简化示例
        # 如果GCN有偏置: gcn_b = vars_list[idx]; idx+=1
        out_w, out_b = vars_list[idx], vars_list[idx+1]; idx+=2

        # --- 网络前向传播 ---
        # 0. 数据预处理 (根据你的骨架数据格式)
        #    例如: 如果 x 是 [B, Frames, Joints, Coords], 你可能需要选择一帧或聚合多帧
        #    假设 x 已经是 [B, num_joints, input_features]
        #    adj 假设是 [B, num_joints, num_joints]

        # 1. MLP (示例: 作用于每个节点特征)
        # x: [B, N, D_in] -> [B, N, D_mlp]
        x = F.relu(F.linear(x, mlp_w, mlp_b))

        # 2. GCN (简化示例: AXW)
        # x: [B, N, D_mlp], adj: [B, N, N], gcn_w: [D_gcn, D_mlp]
        # x_gcn_input = x
        # support = torch.bmm(adj, x_gcn_input) # [B, N, D_mlp]
        # output_gcn = torch.matmul(support, gcn_w.t()) # [B, N, D_gcn] (需要gcn_w转置)
        # x = F.relu(output_gcn)
        # --- 或者使用 PyG 的 GCNConv ---
        # x = self.gcn_conv1(x, edge_index, vars_for_gcn_conv1) # 需要正确传递参数

        # 假设GCN的输出是节点级别的嵌入 x: [B, N, D_gcn]
        # 如果需要图级别的嵌入用于LDA，需要进行图池化
        # 例如，全局平均池化:
        x_graph_embedding = x.mean(dim=1) # [B, D_gcn]

        # 3. 输出层
        embeddings = F.linear(x_graph_embedding, out_w, out_b) # [B, output_embedding_dim]

        return embeddings # 返回用于LDA损失的特征嵌入

    def parameters(self): # 必须提供这个方法
        return self.vars
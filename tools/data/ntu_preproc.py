import copy as cp
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
from mmcv import dump
from tqdm import tqdm

from pyskl.smp import mrlines



eps = 1e-3


def parse_skeleton_file(ske_name, root='/dadaY/xinyu/dataset/nturgbd_skeletons_s001_to_s017/'):
    #构建完整的骨架文件路径
    ske_file = osp.join(root, ske_name + '.skeleton')

    #读取骨架文件
    lines = mrlines(ske_file)
    idx = 0  #初始化行索引
    # 获取总帧数（第一行）
    num_frames = int(lines[0])
    num_joints = 25    # 设定关节点数量为25

    idx += 1  #移动到下一行

    body_data = dict()  # 用于存储每个bodyID对应的骨架数据
    fidx = 0  #记录有效帧的索引
    # 遍历每一帧
    for f in range(num_frames):
        num_bodies = int(lines[idx])   # 当前帧中的人数
        idx += 1
        if num_bodies == 0:
            continue  #若当前帧没人就跳过
        for b in range(num_bodies):
            bodyID = int(lines[idx].split()[0])   # 获取人体ID
            if bodyID not in body_data:
                kpt = []  # 初始化关节点列表
                body_data[bodyID] = dict(kpt=kpt, start=fidx)  # 存入数据结构
            idx += 1

            # 检查关节点数量（应为25）
            assert int(lines[idx]) == 25
            idx += 1
            joints = np.zeros((25, 3), dtype=np.float32)  # 初始化25个关节点的(x, y, z)

            # 读取25个关节点的位置信息
            for j in range(num_joints):
                line = lines[idx].split()
                joints[j, :3] = np.array(line[:3], dtype=np.float32)  # 只取前三个(x, y, z)
                idx += 1
            body_data[bodyID]['kpt'].append(joints) # 添加当前帧的关节点到该人的数据中
        fidx += 1 # 有效帧索引增加


    # 对每个人的骨架数据做后处理
    for k in body_data:
        # 计算该人体的运动强度（关节点位置变化的方差和）
        body_data[k]['motion'] = np.sum(np.var(np.vstack(body_data[k]['kpt']), axis=0))
        # 将关节点数据列表转换为numpy数组，shape: (有效帧数, 25, 3)
        body_data[k]['kpt'] = np.stack(body_data[k]['kpt'])

    # 确保所有行都已处理
    assert idx == len(lines)
    return body_data    #返回每个人的骨架数据


def spread_denoising(body_data_list):
    wh_ratio = 0.8
    spnoise_ratio = 0.69754

    def get_valid_frames(kpt):
        valid_frames = []
        for i in range(kpt.shape[0]):
            x, y = kpt[i, :, 0], kpt[i, :, 1]
            if (x.max() - x.min()) <= wh_ratio * (y.max() - y.min()):
                valid_frames.append(i)
        return valid_frames

    for item in body_data_list:
        valid_frames = get_valid_frames(item['kpt'])
        if len(valid_frames) == item['kpt'].shape[0]:
            item['flag'] = True
            continue
        ratio = len(valid_frames) / item['kpt'].shape[0]
        if 1 - ratio >= spnoise_ratio:
            item['flag'] = False
        else:
            item['flag'] = True
            item['motion'] = min(item['motion'],
                                 np.sum(np.var(item['kpt'][valid_frames].reshape(-1, 3), axis=0)))
    body_data_list = [item for item in body_data_list if item['flag']]
    assert len(body_data_list) >= 1
    _ = [item.pop('flag') for item in body_data_list]
    body_data_list.sort(key=lambda x: -x['motion'])
    return body_data_list


def non_zero(kpt):
    s = 0
    e = kpt.shape[1]
    while np.sum(np.abs(kpt[:, s])) < eps:
        s += 1
    while np.sum(np.abs(kpt[:, e - 1])) < eps:
        e -= 1
    return kpt[:, s: e]

#
def gen_keypoint_array(body_data):
    length_threshold = 11  # 设置关键点序列的最小长度阈值，过滤掉过短的序列

    # 深拷贝并按运动强度（motion）排序，将动作变化大的数据排在前面
    body_data = cp.deepcopy(list(body_data.values()))
    body_data.sort(key=lambda x: -x['motion'])  # 按照'运动强度'从大到小排序

    #如果只有一个骨架数据，直接返回关节点数据
    if len(body_data) == 1:
        return body_data[0]['kpt'][None]
    else:
        # 筛选掉长度小于阈值的序列
        body_data = [item for item in body_data if item['kpt'].shape[0] > length_threshold]
        # 如果只有一个骨架数据，直接返回其关节点数据
        if len(body_data) == 1:# 过滤掉运动强度小于阈值的序列
            return body_data[0]['kpt'][None]
        # 调用 `spread_denoising` 对骨架数据进行去噪处理
        body_data = spread_denoising(body_data)
        # 如果去噪后仍然只有一个骨架数据，直接返回
        if len(body_data) == 1:
            return body_data[0]['kpt'][None]
        max_fidx = 0 # 初始化最大帧索引

        # 计算所有骨架数据中的最大帧索引
        for item in body_data:
            max_fidx = max(max_fidx, item['start'] + item['kpt'].shape[0])

            # 初始化一个形状为(2, max_fidx, 25, 3)的空数组来存储关节点数据
        keypoint = np.zeros((2, max_fidx, 25, 3), np.float32)

        # 初始化起始和结束索引
        s1, e1, s2, e2 = body_data[0]['start'], body_data[0]['start'] + body_data[0]['kpt'].shape[0], 0, 0
        # 将第一个骨架数据填充到 `keypoint[0]` 中
        keypoint[0, s1: e1] = body_data[0]['kpt']
        #遍历剩下的骨架数据
        for item in body_data[1:]:
            s, e = item['start'], item['start'] + item['kpt'].shape[0]
            # 如果两个数据的时间区间有重叠或紧接，则合并到 `keypoint[0]`
            if max(s1, s) >= min(e1, e):
                keypoint[0, s: e] = item['kpt']
                s1, e1 = min(s, s1), max(e, e1)
                # 否则，合并到 `keypoint[1]`
            elif max(s2, s) >= min(e2, e):
                keypoint[1, s: e] = item['kpt']
                s2, e2 = min(s, s2), max(e, e2)

        # 调用 `non_zero` 函数去除零值（可能是填充的空数据）
        keypoint = non_zero(keypoint)
        # 如果 `keypoint[0]` 和 `keypoint[1]` 的第一个关节点的绝对和接近零，则反转 `keypoint`
        if np.sum(np.abs(keypoint[0, 0, 1])) < eps and np.sum(np.abs(keypoint[1, 0, 1])) > eps:
            keypoint = keypoint[::-1]
        return keypoint  # 返回最终的关键点数据


root = '/dadaY/xinyu/dataset/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/'   # 设置骨架文件所在的文件夹路径
skeleton_files = os.listdir(root) # 获取该文件夹下所有骨架文件的文件名列表
names = [x.split('.')[0] for x in skeleton_files]   # 去掉文件扩展名，得到纯粹的样本名（例如S001C001P001R001A001）
names.sort() # 对样本名进行排序，方便后续处理
missing = mrlines('/dadaY/xinyu/dataset/samples_with_missing_skeletons.txt')    #读取缺失的样本名列表（存在于 txt 文件中），使用自定义函数 mrlines 逐行读取
missing = set(missing)              # 转换为集合，加快查找效率
names = [x for x in names if x not in missing]     # 过滤掉缺失的样本，保留完整可用的样本名

extended = False  # 初始化标志变量，表示是否检测到 NTU120 数据
for name in names:
    if int(name.split('A')[-1]) > 60:   # 如果动作编号 A 后面的数字大于60（NTU60 只有前60类），则说明为 NTU120 数据
        extended = True
        print('NTURGB+D 120 skeleton files detected, will generate both ntu60_3danno.pkl and ntu120_3danno.pkl. ')
        break    # 一旦检测到 NTU120，就无需再继续遍历，跳出循环

if not extended:
    print('NTURGB+D 120 skeleton files not detected, will only generate ntu60_3danno.pkl. ')  #如果未检测到动作编号大于 60 的样本（即未检测到 NTU120 数据），打印提示信息。


def gen_anno(name):
    body_data = parse_skeleton_file(name, root)  # 解析骨架数据（一个样本）
    if len(body_data) == 0:
        return None   # 如果该样本中没有骨架数据，返回 None
    keypoint = gen_keypoint_array(body_data).astype(np.float16)   # # 获取关键点数据（浮点16精度）
    label = int(name.split('A')[-1]) - 1  # 提取标签编号（动作编号 Axxx 转为 0-based 索引）
    total_frames = keypoint.shape[1]   #获取总帧数
    return dict(frame_dir=name, label=label, keypoint=keypoint, total_frames=total_frames)  # 返回一个注解字典


anno_dict = {}  # 用于保存所有样本的注解
num_process = 1  # 是否使用多进程，设为 1 表示只用单线程

if num_process == 1:
    # 单线程生成注解，每个样本生成一个包含以下四个字段的字典：
    # Each annotations has 4 keys: frame_dir, label, keypoint, total_frames
    for name in tqdm(names):
        anno_dict[name] = gen_anno(name)
else:
    pool = mp.Pool(num_process)
    annotations = pool.map(gen_anno, names)
    pool.close()
    for anno in annotations:
        anno_dict[anno['frame_dir']] = anno

names = [x for x in names if anno_dict is not None]
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]     # 指定训练集参与者（Person ID，Pxxx 中的编号），用于交叉主体（xsub）划分。



if extended:
    xsub_train = [name for name in names if int(name.split('P')[1][:3]) in training_subjects]
    xsub_val = [name for name in names if int(name.split('P')[1][:3]) not in training_subjects]
    xset_train = [name for name in names if int(name.split('S')[1][:3]) % 2 == 0]
    xset_val = [name for name in names if int(name.split('S')[1][:3]) % 2 == 1]
    split = dict(xsub_train=xsub_train, xsub_val=xsub_val, xset_train=xset_train, xset_val=xset_val)
    annotations = [anno_dict[name] for name in names]
    dump(dict(split=split, annotations=annotations), '/dadaY/xinyu/dataset/ntu120_kpl/ntu120_3danno.pkl')

names = [name for name in names if int(name.split('A')[-1]) <= 60]   #筛选出 NTU60 的样本（即动作编号 ≤ 60）。
xsub_train = [name for name in names if int(name.split('P')[1][:3]) in training_subjects]
xsub_val = [name for name in names if int(name.split('P')[1][:3]) not in training_subjects]
#按照（CameraID）划分训练 / 验证集（交叉视角xview）。
xview_train = [name for name in names if 'C001' not in name]  # 除视角1外的为训练集
xview_val = [name for name in names if 'C001' in name]  # 除视角1外的为验证集
split = dict(xsub_train=xsub_train, xsub_val=xsub_val, xview_train=xview_train, xview_val=xview_val)
annotations = [anno_dict[name] for name in names]  # 所有样本的注解信息
dump(dict(split=split, annotations=annotations), '/dadaY/xinyu/dataset/ntu60_pkl/ntu60_3danno.pkl')     # 保存为 pkl 文件

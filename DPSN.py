from __future__ import annotations  
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
# import psutil
from memory_profiler import memory_usage
import os
import time
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from sklearn.preprocessing import MinMaxScaler
from enum import Enum, unique
# import logging
# import threading
from functools import wraps
import heapq





def save_metric_to_csv(dataName, counts, values, metric_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({'': counts, metric_name: values})
    file_path = os.path.join(save_dir, f"{dataName}_result.csv")
    df.to_csv(file_path, index=False)
    print(f"{metric_name}结果已保存到: {file_path}")



def data_processing(filePath: str,  # 要读取的文件位置
                    minMaxScaler=True,  # 是否进行归一化
                    drop_duplicates=True,  # 是否去重
                    shuffle= False):  # 是否进行数据打乱
    data = pd.read_csv(filePath, header=None)
    data1 = data
    
    # 将所有非数值数据转换为0
    for col in data.columns[:-1]:  # 排除最后一列(标签列)
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    true_k = data.iloc[:, -1].drop_duplicates().shape[0]
    if drop_duplicates:
        data = data.drop_duplicates().reset_index(drop=True)
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    if minMaxScaler:
        # data.iloc[:, :-1] = MinMaxScaler().fit_transform(data.iloc[:, :-1])
    # 更安全的类型转换，处理可能的异常情况
        try:
            numeric_columns = data.columns[:-1]
            # 只转换数值类型的列
            for col in numeric_columns:
                if data[col].dtype in ['int64', 'int32', 'float32', 'object']:
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
            data.iloc[:, :-1] = MinMaxScaler().fit_transform(data.iloc[:, :-1])
        except Exception as e:
            print(f"数据类型转换出错: {e}")
            # 如果转换失败，使用原始方法
            data.iloc[:, :-1] = MinMaxScaler().fit_transform(data.iloc[:, :-1])
    return np.array(data.iloc[:, :-1]), data.iloc[:, -1].values.tolist(),true_k


def print_estimate(label_true, label_pred, dataset: str, testIndex, iterIndex, time, params='无'):
    print('%15s 数据集，参数%10s，第 %2d次测试，第 %2d次聚类，真实簇 %4d个，预测簇 %4d个，用时 %10.2f s,'
          'RI %0.4f,ARI %0.4f,NMI %0.4f，'
          % (dataset.ljust(15)[:15], params.ljust(10)[:10], int(testIndex), int(iterIndex),
             len(list(set(label_true))), len(list(set(label_pred))), time,
             rand_score(label_true, label_pred),
             adjusted_rand_score(label_true, label_pred),
             normalized_mutual_info_score(label_true, label_pred)))


class Node(object):
    # 基础数据
    id: int  # 节点id
    data: tuple  # 节点数据
    label: int  # 节点标签
    labelName: str  # 节点标签名称
    # 迭代过程
    adjacentNode: dict  # 相邻节点
    degree: int  # 度
    iteration: int  # 迭代序数
    isVisited: bool  # 访问标记
    # 结果
    label_pred: int  # 预测标签
    node_uncertainty : float


    parent : Node 
    query : bool
    children : int
    root_judge : list #是否是本层迭代的根节点？  如果是，则列表最后一个值为1，如果不是，则为0
    def __init__(self, node_id, data, label, labelName):
        # 基础数据
        self.id = int(node_id)  # 节点id
        self.data = data  # 节点数据
        self.label = label  # 节点标签
        self.labelName = labelName  # 节点标签名称
        # 迭代过程
        self.adjacentNode = {}  # 相邻节点
        self.degree = 0  # 度
        self.iteration = 0  # 迭代序数
        self.isVisited = False  # 访问标记
        self.node_num = 0
        self.parent = None #双亲节点，如果没找到就是None
        self.children = 0 #存储孩子节点 
        # 结果
        self.label_pred = 0  # 预测标签
        self.node_uncertainty = 0.0 # 节点的不确定度

        self.root_judge = []   # 初始状态：不是根节点
        self.query = False  # 节点是否被查询过
    def add_adjacent_node(self, node: Node):
        self.adjacentNode[node.id] = node
        self.degree += 1

    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def set_node_num(self, node_num: int):
        self.node_num = node_num


class Graph(object):
    nodeList: list[Node]
    node_size: int

    def __init__(self):
        self.nodeList = []
        self.node_size = len(self.nodeList)

    def add_Node(self, node: Node):
        self.nodeList.append(node)
        self.node_size = len(self.nodeList)

class ClusterResult:
    def __init__(self, dataName, iteration, roots, execution_time, ri, ari, nmi):
        self.dataName = dataName  # 数据集名称
        self.iteration = iteration  # 聚类迭代次数
        self.roots = roots  # 聚类树的根节点及其结构
        self.execution_time = execution_time  # 执行时间
        self.ri = ri  # Rand Index
        self.ari = ari  # Adjusted Rand Index
        self.nmi = nmi  # Normalized Mutual Information

    def __str__(self):
        return (f"ClusterResult(dataName={self.dataName}, iteration={self.iteration}, "
                f"execution_time={self.execution_time:.2f}s, ri={self.ri:.4f}, ari={self.ari:.4f}, "
                f"nmi={self.nmi:.4f})")


class ClusterStoreWithHeap:
    def __init__(self):
        self.heap = []  # 堆，用于按优先级存储聚类树

    def add_result(self, cluster_result: ClusterResult):
        heapq.heappush(self.heap, (cluster_result.execution_time, cluster_result))  # 使用执行时间作为优先级
    def get_best_result(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]  # 返回聚类结果
        return None

    def print_results(self):
        """按执行时间打印所有聚类树结果及其结构"""
        for _, result in self.heap:
            print(result)  # 打印基础信息
            # 打印聚类树结构
            #print(f"\n聚类树结构（迭代次数：{result.iteration}）:")
            for root_id in result.roots:
                #print(f"根节点 {root_id} 的子结构:")
                #print_tree(result.roots[root_id])
                #print("----------")
                pass
        


    def __str__(self):
        return #f"ClusterResult(dataName={self.dataName}, iteration={self.iteration}, ri={self.ri}, ari={self.ari}, nmi={self.nmi}, execution_time={self.execution_time})"

def get_distribute(current_roots: list[int], nodeList: dict[int, Node], size: int):
    label_true = [-1 for i in range(size)]
    label_pred = [-1 for i in range(size)]
    i = 0
    count = 0
    for root_id in current_roots:  # 遍历根节点的ID列表
        node = nodeList[root_id]  # 通过节点ID获取实际的节点对象
        next_node = [node]
        visited = {node.id}
        while next_node:
            r = next_node.pop()
            label_pred[r.id] = i
            label_true[r.id] = r.label
            count += 1
            for n in r.adjacentNode:
                if n not in visited:
                    visited.add(n)
                    next_node.append(r.adjacentNode[n])
        i += 1
    return label_true, label_pred,count



def extract_pairs_distance(roots, iteration):
    scts = []
    for key in roots:
        sct_node = [roots[key]]
        sct_data = [roots[key].data]
        node_num = 1
        next_node = [roots[key]]
        visited = {key}
        while next_node:
            r = next_node.pop()
            for n in r.adjacentNode:
                if r.adjacentNode[n].iteration == iteration and n not in visited:
                    visited.add(n)
                    next_node.append(r.adjacentNode[n])
                    sct_node.append(r.adjacentNode[n])
                    sct_data.append(r.adjacentNode[n].data)
                    node_num += 1
        for node in sct_node:
            node.set_node_num(node_num)
        # print('%d sct共有 %d 个节点' % (key, node_num))
        scts.append(dict(sct_data=sct_data, sct_node=sct_node))
    return scts


def format_distance(distance):
    d = []
    for i in range(len(distance)):
        for j in range(i + 1, len(distance[i])):
            d.append(distance[i][j])
    d.sort()
    return d


# 计算点对之间的局部密度
def compute_local_density(nodeList, distance, cut_off_distance):
    local_density_point = [dict(node=node, local_density_node_num=0) for node in nodeList]
    for i in range(len(distance)):
        for j in range(i + 1, len(distance[i])):
            if distance[i][j] <= cut_off_distance:
                local_density_point[i]['local_density_node_num'] += 1
                local_density_point[j]['local_density_node_num'] += 1
    data = compute_degree_weigh(sorted(local_density_point, key=lambda k: k['local_density_node_num'], reverse=True))
    return data


# 衡量节点的度距离
def compute_degree_weigh(point):
    repeated_point = [p for p in point if p['local_density_node_num'] == point[0]['local_density_node_num']]
    d = repeated_point[0]['node']
    if len(repeated_point) != 1:
        base_degree = repeated_point[0]['node'].degree
        for p in repeated_point:
            if p['node'].degree > base_degree:
                d = p['node']
                base_degree = p['node'].degree
    return d


def findDensityPeak(query_times: int, roots: list[Node], cut_off=0.4, iteration=0):
    scts = extract_pairs_distance(roots, iteration)
    rootList = {}
    # 处理每个强连通分量(SCT)
    for sct in scts:
        if len(sct['sct_data']) > 1:
            # 计算密度峰值点
            distances = pairwise_distances(np.array(sct['sct_data']), metric="euclidean")
            pairs_distance = format_distance(distances)
            index = min(round(len(pairs_distance) * cut_off), len(pairs_distance) - 1)
            cut_off_distance = pairs_distance[index]
            root = compute_local_density(sct['sct_node'], distances, cut_off_distance)
            # 记录所有节点ID，确保不会丢失节点
            all_node_ids = {node.id for node in sct['sct_node']}
            # 将根节点添加到rootList
            rootList[root.id] = root
            # 创建一个临时字典来存储节点的原始连接
            original_connections = {}
            for node in sct['sct_node']:
                original_connections[node.id] = list(node.adjacentNode.keys())
            # 第一步：设置父节点关系，但不修改连接
            for node in sct['sct_node']:
                node.parent = root
                node.root_judge.append(0)  # 非根节点标记为0
                    # 更新根节点的子节点数量
                # root.children += (node.children + 1)
            # 第二步：重建连接，创建星形结构
            for node in sct['sct_node']:
                if node.id != root.id:
                    # 清除所有与同层节点的连接（除了根节点）
                    to_remove = []
                    for adj_id in node.adjacentNode:
                        adj = node.adjacentNode[adj_id]
                        if adj.iteration == iteration and adj.id != root.id:
                            to_remove.append(adj_id)
                    
                    # 删除连接
                    for adj_id in to_remove:
                        if adj_id in node.adjacentNode:
                            del node.adjacentNode[adj_id]
                            node.degree -= 1
                    
                    # 确保与根节点连接
                    if root.id not in node.adjacentNode:
                        node.add_adjacent_node(root)
                    if node.id not in root.adjacentNode:
                        root.add_adjacent_node(node)
            
            # 标记根节点
            root.root_judge[-1] = 1  # 根节点标记为1
        else:
            # 如果SCT只有一个节点，则该节点为根节点
            root = sct['sct_node'][0]
            rootList[root.id] = root
            root.parent = root
            root.root_judge.append(1)  # 根节点标记为1
    
    # 确保所有节点都被包含在结果中
    for sct in scts:
        for node in sct['sct_node']:
            if node.id not in rootList and node.parent and node.parent.id in rootList:
                # 如果节点不是根节点但其父节点是根节点，则保持连接
                continue
            elif node.id not in rootList:
                # 如果节点不是根节点且其父节点也不是根节点，则将其添加为根节点
                rootList[node.id] = node
                node.parent = node
                node.root_judge.append(1)
    print("findDensityPeak结束")
    return rootList

# 将数据对象的数据提取用于kdTree计算
# input: list[Node]
# output: list: data
def extract_data_from_Node(nodeList: dict[int, Node]):
    dataList = []
    for key in nodeList:
        dataList.append(nodeList[key].data)
    return dataList


# 寻找数据的最近邻
# input: list[Node]
# output: list: Node's NN index
def findNNs(query_times: int,nodeList: list[Node], k):
    dataList = extract_data_from_Node(nodeList)
    return kdTree(query_times,dataList, nodeList, k,return_dist  =False)


def kdTree(query_times: int, dataList, nodeList: dict[int, Node], k, return_dist=False):
    print("开始findNN")
    origin = np.array(dataList)
    k = min(k, len(dataList))
    neighbors = NearestNeighbors(n_neighbors=k).fit(origin)
        # 获取每个样本的最近邻索引，返回数组 shape=(n_samples, 3)
    indices = neighbors.kneighbors(origin, return_distance=False)
    nns = {}
    snns = {}
        # 假设 nodeList 的 key 顺序与 dataList 的顺序一致
    pos = [key for key in nodeList]
    for i, key in enumerate(nodeList):
        if k > 2:
            nns[nodeList[key].id] = pos[indices[i][1]]
            snns[nodeList[key].id] = pos[indices[i][2]]
        elif k == 2:
            nns[nodeList[key].id] = pos[indices[i][1]]
            snns[nodeList[key].id] = pos[indices[i][1]]
        else:
            nns[nodeList[key].id] = nodeList[key].id
            snns[nodeList[key].id] = nodeList[key].id
    print("完成findNN")
    return nns, snns


        



def compute_sct_num(roots: dict[int, Node], iteration: int):
    rebuild_roots = []
    for key in roots:
        sct_node = [roots[key]]
        node_num = 1
        next_node = [roots[key]]
        other_node = 0
        visited = {key}
        while next_node:
            r = next_node.pop() 
            for n in r.adjacentNode:
                if r.adjacentNode[n].iteration == iteration and n not in visited :
                    next_node.append(r.adjacentNode[n])
                    sct_node.append(r.adjacentNode[n])
                    visited.add(n)
                    other_node = n
                    node_num += 1
        if node_num == 2:
            rebuild_roots.append((key, other_node))
        for node in sct_node:
            node.set_node_num(node_num)
        # print('%d sct共有 %d 个节点' % (key, node_num))
    return rebuild_roots


def construction(nodeList: list[Node], nns: dict[int], snns:dict[int],iteration: int, query_times: int):
        nodeDict = {node.id: node for node in nodeList} 
        roots = {}
        candidates = np.array(nodeList)
        print("开始construction")
        # 记录初始节点数量，用于控制簇数量
        initial_node_count = len(candidates)
        min_clusters_required = max(1, int(initial_node_count/ 100000))  # 至少保留1/5的簇数量
        
        while len(candidates) > 0:
            # 选择度数最大的节点作为起始点
            start_node = max(candidates, key=lambda x:  x.degree)
            link = np.array([], dtype=int)
            visited = set()
            current_node = start_node
            
            while True:
                if current_node.id in visited:
                    # 如果遇到已访问的节点，将当前路径中的最后一个节点作为根节点
                    root_node = nodeDict[link[-1]]
                    roots[root_node.id] = root_node
                    break
                
                visited.add(current_node.id)
                link = np.append(link, current_node.id)
                current_node.set_iteration(iteration)
                
                # 获取最近邻节点ID
                j_id = nns[current_node.id]
                j = nodeDict[j_id]
                
                if j_id in link:
                    # 如果最近邻已在当前路径中，形成环，将当前节点作为根节点
                    roots[current_node.id] = current_node
                    break
                elif j_id not in [node.id for node in candidates]:
                    # 如果最近邻不在候选节点中，建立连接并结束当前路径
                    current_node.add_adjacent_node(j)
                    j.add_adjacent_node(current_node)
                    break
                else:
                    # 建立连接并继续
                    current_node.add_adjacent_node(j)
                    j.add_adjacent_node(current_node)
                    current_node = j
            
            # 更新candidates
            candidates = np.array([node for node in candidates if node.id not in link])
            
            # 检查簇数量是否已经达到下限
            if len(candidates) + len(roots) <= min_clusters_required:
                for node in candidates:
                    roots[node.id] = node
                    node.set_iteration(iteration)
                break
        print("完成construction")    
        return roots
    



def connect_roots(rebuild_roots, roots, snns, nodeList: list[Node], iteration: int, query_times: int):
    print("开始connect_roots")
    if query_times == 0:
        nodeDict = {node.id: node for node in nodeList}
        candidates = np.array(rebuild_roots).reshape(-1)
        for root in rebuild_roots:
            # 使用 nodeDict 访问节点，避免索引越界
            root_node_0 = nodeDict[root[0]]
            root_node_1 = nodeDict[root[1]]      
            # 检查 cannot_link 关系，使用节点 ID

            roots.pop(root[0])
            left_connect_node = nodeDict[snns[root[0]]].node_num
            right_connect_node = nodeDict[snns[root[1]]].node_num
                # 选择较大的节点作为主节点
            if left_connect_node <= right_connect_node:
                big_node = nodeDict[snns[root[1]]]
                small_node = nodeDict[root[1]]
            else:
                big_node = nodeDict[snns[root[0]]]
                small_node = nodeDict[root[0]]
                # 进行相邻节点连接
            big_node.add_adjacent_node(small_node)
            small_node.add_adjacent_node(big_node)
            # 如果小节点在 candidates 中，更新 roots
            if small_node.id in candidates:
                roots[small_node.id] = small_node
    print("完成connect_roots")
                    

def rebuild(snns: dict[int], roots: list[Node], nodeList: list[Node], iteration: int, query_times):
    """
    重建聚类结构，调用 connect_roots 函数。
    
    Args:
        snns (dict[int]): 次近邻字典
        roots (list[Node]): 根节点列表
        nodeList (list[Node]): 节点列表
        iteration (int): 当前迭代次数
        query_times (int): 查询次数
    """
    rebuild_roots = compute_sct_num(roots, iteration)
    connect_roots(rebuild_roots, roots, snns, nodeList, iteration, query_times)





class Task():
    def __init__(self, params, iterIndex: int, dataName: str, path):
        self.params = params
        self.iterIndex = str(iterIndex)
        self.dataName = str(dataName)
        self.filePath = str(path)

    def __str__(self):
        return '{}-{}'.format(self.dataName, self.iterIndex)

# 记录类型类：用于控制输出的结果类型，该名称会被用于创建输出结果的上级目录
@unique
class RecordType(Enum):
    assignment = 'assignment'
    tree = 'tree'

class Assignment():
    def __init__(self, types: str, iter: str, record: dict):
        self.type = types
        self.iter = iter
        self.record = record
# 记录类：保存每次输出的结果

class Record():
    def __init__(self):
        self.record = []
        self.cpuTime = []

    # 保存每轮聚类结果，必须明确输出类型，子迭代序数（多轮迭代结果），标签信息和预测结果
    def save_output(self, types: RecordType, label_true: list, label_pred: list, iter=0):
        assert isinstance(types, RecordType), TypeError(
            '输入类型必须为RecordType枚举类，请检查。当前类型为 ' + str(type(types)))
        assert len(label_pred) > 0, \
            TypeError('label_pred必须为list类型，且长度不为0')
        assert len(label_pred) > 0, \
            TypeError('label_true必须为list类型，且长度不为0')
        self.record.append(
            Assignment(types, str(iter), {'label_true': label_true, 'label_pred': label_pred}))

    # 保存每轮所用的时间，最终计算得到总时间。
    def save_time(self, cpuTime):
        assert isinstance(cpuTime, float), TypeError("输入类型必须为float类型，请检查。当前类型为 " + str(type(cpuTime)))
        self.cpuTime.append(cpuTime)

# logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s- %(message)s')


class ExpMonitor():
    def __init__(self, expId: str, algorithmName: str, storgePath="G:\Experiment"):
        self.task = None
        self.expId = expId
        self.algorithmName = algorithmName
        self.storgePath = storgePath
        self.stop_thread = False  # 初始化 stop_thread 属性

# ExpMonitor 类的 __call__ 方法中
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.task = kwargs['task'] if 'task' in kwargs else args[0]
            # if not self.repeat_thread_detection("monitor"):
                # t = threading.Thread(target=self.out_memory, name="monitor", daemon=True)
                # t.start()
            res = func(*args, **kwargs)
            #record: Record = res['record']  # 确保 res 中包含有效的 record
            # self.out_record(record.record)  # 此处触发断言
            # self.out_cpu_time(record.cpuTime)
            # logging.info('%15s 数据集第 %2d 轮测试完成，运行时间 %10.2f s，参数信息：%10s' %
            #             (self.task.dataName.ljust(15)[:15], int(self.task.iterIndex),
            #             sum(record.cpuTime), str(self.task.params).ljust(10)[:10]))
            return res
        return wrapper


    # def out_memory(self):
    #     p = psutil.Process(os.getpid())
    #     while not self.stop_thread:  # 加入一个退出条件
    #         cpu_percent = round(((p.cpu_percent(interval=1) / 100) / psutil.cpu_count(logical=False)) * 100, 2)
    #         mem_percent = round(p.memory_percent(), 2)
    #         mem_size = p.memory_info().rss
    #         data = dict(cpu_utilization_ratio=cpu_percent,
    #                     memory_utilization_ratio=mem_percent,
    #                     memory_size_mb=round(mem_size / 1024 / 1024, 4))
    #         path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
    #                             'memory')
    #         self.lineOutput(path, self.task.dataName, data)


    # 在实验结束后，调用这个方法停止线程
    def stop_monitor_thread(self):
        self.stop_thread = True

    # def out_record(self, records: list[Assignment]):
    #     assert len(records) > 0, \
    #         ValueError('输出结果record没有记录，请检查')
    #     for record in records:
    #         path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
    #                             'output',
    #                             self.task.dataName, record.type.value)
    #         self.makeDir(path)
    #         pd.DataFrame(record.record).to_csv(path + '/' + record.iter + '.csv', index=False)

    # def out_cpu_time(self, cpuTime: list):
    #     assert len(cpuTime) > 0, \
    #         ValueError('cputime 没有记录，请检查')
    #     path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
    #                         'cpuTime')
    #     self.makeDir(path)
    #     self.lineOutput(path, 'cpuTime', dict(dataName=self.task.dataName, cpuTime=sum(cpuTime)))

    # 输出信息工具
    def lineOutput(self, path, fileName, data: dict):
        self.makeDir(path)
        outputPath = os.path.join(path, fileName + '.csv')
        if not os.path.exists(outputPath):
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, mode='a')
        else:
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, header=False, mode='a')

    # 目录创建工具
    def makeDir(self, path):
        os.makedirs(path, exist_ok=True)

    # 查询线程是否活动
    # def repeat_thread_detection(self, tName):
    #     for item in threading.enumerate():
    #         if tName == item.name:
    #             return True
    #     return False




def query_oracle(node1, node2):
    """
    根据两个节点的真实标签判断它们之间的连接关系。
    """
    if node1.label == node2.label:
        return "must_link"
    else:
        return "cannot_link"


def process_uncertain_nodes(nodes, nodeList, current_roots, label, query_times, n_parts=1):
    """
    处理不确定节点，每个节点处理后都记录ARI值
    Args:
        nodes (list[Node]): 要处理的节点列表（已整体排序）
        nodeList (dict): 节点字典
        current_roots (list): 当前根节点ID列表
        label (list): 真实标签列表
        query_times (int): 当前查询次数
        n_parts (int): 将节点分成几份处理
    """
    ari_values = []
    nmi_values = []
    query_counts = []
    
    # 直接使用传入的已排序节点序列进行处理
    total_nodes = len(nodes)
    parts = []
    
    # 仅按照n_parts参数将节点分成几份，不再按iteration分层
    for part in range(n_parts):
        start_idx = (total_nodes * part) // n_parts
        end_idx = (total_nodes * (part + 1)) // n_parts
        if start_idx < end_idx:
            parts.append(nodes[start_idx:end_idx])
    
    # 依次处理每一份节点
    for part in parts:
        for node in part:
            query_times, nodeList, current_roots, updated = process_single_node(
                node, nodeList, current_roots, label, query_times)
            if updated:
                    # 如果查询次数增加，则记录ARI值
                    label_true, label_pred, count = get_distribute(current_roots, nodeList, len(label))
                    current_ari = adjusted_rand_score(label_true, label_pred)
                    current_nmi = normalized_mutual_info_score(label_true, label_pred)
                    ari_values.append(current_ari)
                    nmi_values.append(current_nmi)
                    query_counts.append(query_times)
                    if query_times  % 200 == 0 :  # 添加查询次数限制
                        print("当前查询次数：", query_times, "当前ARI值：", current_ari, "当前NMI值：", current_nmi)
                    if current_ari >= 0.96:
                        return query_times, nodeList, current_roots, ari_values, nmi_values,query_counts

    return query_times, nodeList, current_roots, ari_values, nmi_values,query_counts

def process_single_node(node, nodeList, current_roots, label, query_times, print_node_info=True):
    """
    处理单个不确定节点，进行查询并更新节点关系
    """
    updated = False
    node_copy = node
    while node_copy.parent.id != node_copy.id:
        node_copy = node_copy.parent
    if node.id != node_copy.id and node.query == False:
        ans = query_oracle(node_copy, node)
        query_times += 1

        updated = True
        
        if ans == "must_link":
            node.query = True
        elif ans == "cannot_link":
            node.query = True
            flag = False  # 标记是否找到可以链接的节点

            sorted_nodes = []
            for i in nodeList.values():
                if i.id == node_copy.id:
                    continue
                distance = np.linalg.norm(np.array(node.data) - np.array(i.data))
                sorted_nodes.append((i, distance))
            sorted_nodes.sort(key=lambda x: x[1])
            
            for i, _ in sorted_nodes:
                ans_2 = query_oracle(i, node)
                query_times += 1
                # 每次查询后立即计算并返回当前ARI值
                # label_true, label_pred, count = get_distribute(current_roots, nodeList, len(label))
                # current_ari = adjusted_rand_score(label_true, label_pred)
                # current_nmi = normalized_mutual_info_score(label_true, label_pred)
                
                if ans_2 == "must_link":
                    flag = True
                    true_root = i
                    node.label_pred = i.label_pred
                    break
                elif ans_2 == "cannot_link":
                    continue

            # 更新节点关系
            del node.parent.adjacentNode[node.id]
            node.parent.degree -= 1
            del node.adjacentNode[node.parent.id]
            node.degree -= 1
            if flag == False:
                current_roots.append(node.id)
                nodeList[node.id] = node
                node.parent = node
            elif flag == True:
                node.add_adjacent_node(true_root)
                true_root.add_adjacent_node(node)
                node.parent = true_root
            
            # 更新节点关系后再次计算ARI值
            # label_true, label_pred, count = get_distribute(current_roots, nodeList, len(label))
            # current_ari = adjusted_rand_score(label_true, label_pred)
            # current_nmi = normalized_mutual_info_score(label_true, label_pred)

    # 每次处理完一个节点后都返回当前的ARI值

    # label_true, label_pred, count = get_distribute(current_roots, nodeList, len(label))
    # current_ari = adjusted_rand_score(label_true, label_pred)
    # current_nmi = normalized_mutual_info_score(label_true, label_pred)
        

    return query_times, nodeList, current_roots, updated

# 修改后的 run 函数（部分）
@ExpMonitor(expId='ALDP', algorithmName='ALDP', storgePath="C:/Users/DELL/OneDrive/桌面/dezh123/123")
def run(task: Task, data_override=None, n_parts=10):
    global query_times
    query_times = 0
    record = Record()
    ari_values = []
    nmi_values = []
    query_counts = []
    # 判断是否使用外部数据
    if data_override is not None:
        data, label, K = data_override
    else:
        data, label, K = data_processing(task.filePath)
    n = len(data)
    nodeList = {i: Node(i, data[i], label[i], label[i]) for i in range(len(data))}
    iteration = 0
    start = time.time()

    iteration_roots_list = []
    while len(nodeList) > 1:
        nns, snns = findNNs(query_times, nodeList=nodeList, k= 3)
        roots = construction(nodeList=list(nodeList.values()), nns=nns, snns = snns,iteration=iteration, query_times=query_times) 
        rebuild(snns, roots, list(nodeList.values()), iteration, query_times)
        nodeList = findDensityPeak(query_times,roots, task.params, iteration=iteration)
        current_roots = list(nodeList.keys())

        iteration_roots_list.append(current_roots)
        # 获取当前聚类标
        label_true, label_pred,count = get_distribute(current_roots, nodeList, len(label))
        ari = adjusted_rand_score(label_true, label_pred)
        nmi = normalized_mutual_info_score(label_true, label_pred)
        end = time.time()
        record.save_time(end - start)
        record.save_output(RecordType.assignment, label_true, label_pred, iteration)
        elapsed_time = time.time() - start
        print_estimate(label_true, label_pred, task.dataName, task.iterIndex, 0, elapsed_time)
        # current_cluster_count_1 = len(nodeList)
        iteration += 1

    ari_values.append(ari)
    nmi_values.append(nmi)
    query_counts.append(query_times)
    

    #  收集聚类树中所有节点
    
    def get_all_nodes(roots: dict[int, Node]) -> list[Node]:
        all_nodes = {}
        for root in roots.values():
            stack = [root]
            while stack:
                node = stack.pop()
                if node.id not in all_nodes:
                    all_nodes[node.id] = node
                    for child in node.adjacentNode.values():
                        stack.append(child)
        return list(all_nodes.values())
        # 外部循环，当nodeList有更新时重新计算
    # nodeList_updated = True
    # max_iteration = max(node.iteration for node in nodeList.values())
    all_nodes = get_all_nodes(nodeList)
    top_nodes = get_top_k_uncertain_nodes(all_nodes, top_k = len(data))
    query_times, nodeList, current_roots, higher_ari_values, higher_nmi_values,  higher_query_counts = process_uncertain_nodes(
        top_nodes, nodeList, current_roots, label, query_times, n_parts=n_parts
    )
    ari_values.extend(higher_ari_values)
    nmi_values.extend(higher_nmi_values)
    query_counts.extend(higher_query_counts)
    label_true, label_pred, count = get_distribute(current_roots, nodeList, len(label))


    print_estimate(label_true, label_pred, task.dataName, task.iterIndex, 0, elapsed_time)   

    # visualize_iteration_structure(nodeList) 
    # analyze_misclassified_points(data, label_true, label_pred, task.dataName)
    print("\n")
    # 统计总运行时间和内存占用
    end = time.time()
    cpu_time = end - start
    # process = psutil.Process(os.getpid())
    # mem_usage = process.memory_info().rss / 1024 / 1024  # 单位：MB

    # 保存所有数据集的运行时间到同一个文件
    cpu_time_file = r"C:\Users\DELL\OneDrive\桌面\dezh123\ALDP_cpu_times.csv"
    cpu_time_df = pd.DataFrame([{"dataName": task.dataName, "cpu_time": cpu_time}])
    

    if not os.path.exists(cpu_time_file):
        cpu_time_df.to_csv(cpu_time_file, index=False, mode='w')
    else:
        cpu_time_df.to_csv(cpu_time_file, index=False, mode='a', header=False)



    # all_datasets_results[task.dataName] = {
    #     'ari_values': ari_values,
    #     'nmi_values': nmi_values,
    #     'query_counts': query_counts
    # }
    save_results_to_csv(task.dataName, query_counts, ari_values, nmi_values)
    return {'record': record, 'ari_values': ari_values, 'nmi_values': nmi_values, 'query_counts': query_counts}

def save_results_to_csv(dataName, query_counts, ari_values, nmi_values):
    """
    将实验结果保存到CSV文件
    
    Args:
        dataName (str): 数据集名称
        query_counts (list): 查询次数列表
        ari_values (list): 对应的ARI值列表
    """
    # 创建保存目录
    result_dir = "C:\\Users\\DELL\\OneDrive\\桌面\\dezh123\\123\\result\\DPSN"
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建结果DataFrame，只包含ARI列
    result_df = pd.DataFrame({
        ' ': query_counts,
        'ARI': ari_values,
        'NMI': nmi_values,
    })
    
    # 保存到CSV文件，index=False表示不保存索引列
    file_path = os.path.join(result_dir, f"{dataName}_result.csv")
    result_df.to_csv(file_path, index=False)
    print(f"结果已保存到: {file_path}")



# 在主程序中创建ClusterStore实例并保存结果
# cluster_store = ClusterStoreWithHeap()

########################################################################################################################################
########################################################################################################################################
# 提取聚类树
########################################################################################################################################
########################################################################################################################################











def get_top_k_uncertain_nodes(nodes, top_k= 100):
    """
    先按照层级（iteration）排序，同一层内按照到根节点的距离排序
    距离越远的节点优先级越高
    """
    def get_root_distance(node):
        # 找到根节点
        root = node
        #while root.parent.id != root.id:
        root = root.parent
            
        # 计算欧几里得距离
        node_data = np.array(node.data)
        root_data = np.array(root.data)
        distance = np.linalg.norm(node_data - root_data)
        
        return distance
    nodes_list = list(nodes)
    nodes_list = sorted(nodes_list, key=lambda node: (node.iteration, node.degree, get_root_distance(node)), reverse=True)
    return nodes_list
    # iteration越大越优先，同一iteration内distance越大越优先
    # return heapq.nlargest(top_k, nodes, 
    #                     key=lambda node: (node.iteration, node.degree, get_root_distance(node)))
                        # key=lambda node: (node.iteration, get_root_distance(node)))








    





if __name__ == '__main__':
    path = "D:/ALDP-master/data/new"
    # path = "D:/ALDP-master/data/datadata_2/合成数据集"
    cut = 0.19
    # cluster_store = ClusterStoreWithHeap()  # 确保在外部初始化

    for dataName in os.listdir(path):
        data, label, K = data_processing(path + '/' + dataName)
        task = Task(round(cut * 1, 2), 1, dataName.split('.')[0], path + '/' + dataName)
        # 用memory_profiler监控run的峰值内存
        mem_usage, result = memory_usage((run, (), {'task': task}), retval=True, interval=0.1, max_usage=True)
        peak_mem = mem_usage if isinstance(mem_usage, float) else max(mem_usage)
        print(f"{task.dataName} 内存峰值: {peak_mem} MB")

        # 保存峰值内存到CSV
        mem_file = r"C:\Users\DELL\OneDrive\桌面\dezh123\ALDP_memory.csv"
        mem_df = pd.DataFrame([{"dataName": task.dataName, "mem_usage(MB)": peak_mem}])
        if not os.path.exists(mem_file):
            mem_df.to_csv(mem_file, index=False, mode='w')
        else:
            mem_df.to_csv(mem_file, index=False, mode='a', header=False)
    # plot_all_datasets_results()
    
    # cluster_store.print_results()
    # best_result = cluster_store.get_best_result()
    print("\n")
    

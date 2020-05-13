'''
@Descripttion: 
@version: 
@Author: GolLight
@LastEditors: Gollight
@Date: 2020-05-12 17:08:03
@LastEditTime: 2020-05-13 10:12:01
'''
import SC3Units as SC3
import sys
import os
sys.path.append("dendrosplit")
from dendrosplit import split,merge,utils,preprocessing,clustering
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering,DBSCAN,KMeans,AgglomerativeClustering,AffinityPropagation
import numpy as np

def sk_tsne(X):
    model = TSNE(verbose=False)
    X_tSNE = model.fit_transform(X)
    return X_tSNE

# Kmeans
def kMeans(X,k):
    km = KMeans(n_clusters=k)
    return km.fit_predict(X)

# Map labels to integers
def str_labels_to_ints(y_str):
    y_int = np.zeros(len(y_str))
    for i,label in enumerate(np.unique(y_str)):
        y_int[y_str == label] = i
    return y_int.astype(int)
'''
@name: Safe_simple
@test: test font
@msg: Safe聚类算法的简单版本
@param {type} 
@return: 聚类标签
'''
def Safe_simple(X,cluster_num,split_score,merge_score,SC3_lables=None,den_labels=None):
    #先进行聚类，聚类算法有SC3，tsne+kmeans，dendrosplit
    if SC3_lables is None:
        SC3_lables = SC3.SC3(X,cluster_num) #避免重复计算
    if den_labels is None:
        D = split.log_correlation(self.X_pre) 
        ys,shistory = split.dendrosplit((D,self.X_pre),
                                preprocessing='precomputed',
                                score_threshold=split_score,
                                verbose=False,
                                disband_percentile=50)
        # Merge cluster labels
        ym,mhistory = merge.dendromerge((D,self.X_pre),ys,score_threshold=merge_score,preprocessing='precomputed',
                                verbose=False,outlier_threshold_percentile=90)
        den_labels = ym
    
    Xtsne = sk_tsne(X)
    kMeans_lables = kMeans(Xtsne,cluster_num)

    SC3_lables = str_labels_to_ints(SC3_lables)
    den_labels = str_labels_to_ints(den_labels)
    kMeans_lables = str_labels_to_ints(kMeans_lables)
    #构造超图，N*H的二值矩阵，H等于聚类数目之和，在第n个基因(行)
    N = np.shape(X)[0] #细胞数
    H = (int)(len(SC3_lables)+len(den_labels)+len(kMeans_lables))
    h_matrix = np.zeros((N, H), dtype=np.float) #共识矩阵初始化0
    for i in range(3):
        for n in range(N):
            if i == 0:
                s = 0
                h_matrix[n,s + SC3_lables[n]] = 1
            if i == 1:
                s = len(SC3_lables)
                h_matrix[n,s + den_labels[n]] = 1
            if i == 2:
                s = len(SC3_lables)+len(den_labels)
                h_matrix[n,s + kMeans_lables[n]] = 1
    
    #CSPA CSPA也从两两相似的计算开始。与MCLA不同的是，CSPA
    # 将两个单元格之间的相似性定义为1，如果它们总是被赋给相同的值
    # 如果它们从未分配到同一个集群，则为0。
    J = 3 #三种聚类方法
    S = h_matrix.dot(h_matrix.T)/J

    #层次聚类
    linkage = 'complete'
    clustering = AgglomerativeClustering(linkage = linkage, n_clusters = cluster_num)
    clustering.fit(S)

    return clustering.labels_


# h_matrix = np.ones((3, 4), dtype=np.float) #共识矩阵初始化0
# J = 3 #三种聚类方法
# S = h_matrix.dot(h_matrix.T)/J
# print(S)
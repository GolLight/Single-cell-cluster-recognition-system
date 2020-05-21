'''
@Descripttion: SC3算法 python 实现
@version: 
@Author: GolLight
@LastEditors: Gollight
@Date: 2020-05-11 21:40:03
@LastEditTime: 2020-05-16 17:44:41
'''
import sys
import os
sys.path.append("dendrosplit")
from dendrosplit import split,merge,utils,preprocessing,clustering
from sklearn.metrics.pairwise import pairwise_distances
"""
From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
"""

import scipy.stats as stats
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import SpectralClustering,DBSCAN,KMeans,AgglomerativeClustering,AffinityPropagation
'''
@name: 
@test: test font
@msg: 计算距离方阵，分别是皮尔森相关系数，欧氏距离，斯皮尔曼相关系数
@param X {2-D array} N细胞* M基因矩阵
@return: D N * N距离矩阵
'''
def pairwise_correlation(X): # Pearson correlation 
    return pairwise_distances(X,metric='correlation')

def pairwise_euclidean(X):#欧式距离
    return pairwise_distances(X,metric="euclidean")

def pairwise_spearmanr(X):#斯皮尔曼相关系数
    return stats.stats.spearmanr(X,axis=1).correlation


'''
@name: cal_dims
@test: test font
@msg: 计算合适的降维数范围，在论文中可以看到三种距离矩阵均是在PCA和拉普拉斯处理后降为原向量维数的4%-7%（作者给出的数据）时ARI指数出现高点，
这说明原向量维数的4%-7%是一个比较合适的目标维数。
@param  X {2-D array} N细胞* M基因矩阵 
@return:最小维数dimsmin和最大维数 dimsmax
'''
def cal_dims(X):  
    dims = np.shape(X)[0] #应该是原细胞数的4%-7%
    dimsmin = (int)(dims * 0.04)
    dimsmax = (int)(dims * 0.07)
    return dimsmin,dimsmax


# Kmeans
def kMeans(X,k):
    km = KMeans(n_clusters=k)
    return km.fit_predict(X)

'''
@name: cal_CSPA
@test: test font
@msg: 通过聚类结果计算共识矩阵，即A与B同属一类，
则共识矩阵相应点上置1，否则置0。                距离矩阵对角线是1
@param ykmeans {1*N array} kmeans聚类结果 
@return: 共识矩阵
'''
def cal_CSPA(ykmeans):
    dim = len(ykmeans)
    count_matrix = np.zeros((dim, dim), dtype=np.float) #共识矩阵初始化0
    for i in range(dim):
        count_matrix[i][i] = 1
    for i in range(1,dim):
        for j in range(dim-1):
            if ykmeans[i] == ykmeans[j]:
                count_matrix[i][j] = 1
                count_matrix[j][i] = 1
    return count_matrix
    

'''
@name: pca_for_sc3
@test: test font
@msg: 使用pca对SC3获得的三个距离方阵N*N,进行批量降维处理然后用kmeans聚类并生成最终的共识矩阵
在这里我们使用CSPA算法，该算法最终需要得到一个共识矩阵，即A与B同属一类，
则共识矩阵相应点上置1，否则置0。对每一个聚类结果都计算出相应的共识矩阵，
再将得到的所有共识矩阵相加并取平均，就得到了最终的共识矩阵。
@param D1 {N * N matrix} 皮尔森距离方阵
@param D2 {N * N matrix} 欧式距离方阵 
@param D3 {N * N matrix} 斯皮尔曼距离方阵 
@param dimsmin {int} 最小维数 
@param dimsmax {int} 最大维数
@param cluster_num {int} kmeans的聚类数   
@return: 最终的共识矩阵
'''
def pca_kmeans_consensus_for_sc3(D1,D2,D3,dimsmin,dimsmax,cluster_num):
    dim = np.shape(D1)[0]
    count = 0
    count_matrix = np.zeros((dim, dim), dtype=np.float) #共识矩阵初始化0
    for j in range(3):
        for i in range(dimsmin,dimsmax):
            D = []
            if j == 0:
                D = D1
            elif j == 1:
                D = D2
            else:
                D = D3
            pca=PCA(n_components=i)
            xdim = pca.fit_transform(D)#降维
            ykmeans = kMeans(xdim,cluster_num) #聚类
            # print(ykmeans)
            count_matrix += cal_CSPA(ykmeans)
            count += 1
    count_matrix /= count
    return count_matrix
            

'''
@name: SC3
@test: test font
@msg: 一种单细胞簇聚类方法
@param X {2-D array} N细胞* M基因矩阵（经过简单筛选） 
@param cluster_num {int} kmeans聚类数 
@param split_score {0-100}  dendrosplit分离参数
@param merge_score {0-100}  dendrosplit凝聚参数，一般为分离参数的一半
@return: 聚类标签
'''
def SC3(X,cluster_num):
    #计算距离方阵
    D1 = pairwise_correlation(X)
    D2 = pairwise_euclidean(X)
    D3 = pairwise_spearmanr(X)

    #降维
    dimsmin,dimsmax = cal_dims(X)
    count_matrix = pca_kmeans_consensus_for_sc3(D1,D2,D3,dimsmin,dimsmax,cluster_num) #最终共识矩阵
    #层次聚类
    linkage = 'complete'
    clustering = AgglomerativeClustering(linkage = linkage, n_clusters = cluster_num)
    clustering.fit(count_matrix)

    return clustering.labels_
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [1, 6, 8, 7, 10, 9, 3, 5, 2, 4]
# x1 = [[1,2,3,4],
#     [2,1,2,3],
#     [3,4,2,3]]
# D1 = pairwise_correlation(x1)
# D2 = pairwise_euclidean(x1)
# D3 = pairwise_spearmanr(x1)
# print(D1)
# print(D2)
# print(D3)
# print(stats.stats.spearmanr(x1,axis=1).correlation)
# count = 0
# for i in range((int)(1550 * 0.04),(int)(1550 * 0.07)):
#     count = count +1
# print(count)    
# x,y = (int)(49 * 0.04),(int)(49 * 0.07)
# print(x)
# print(y)
# y = [0,1,2,1,2,0,2,1,3,4]
# y1 = [0,0,2,1,2,0,2,1,3,4]
# c1 = cal_CSPA(y)
# c2 = cal_CSPA(y1)
# print(c1)
# print(c2)
# print(c1+c2)
# print((c1+c2)/2)
# for j in range(3):
#     print(j)
'''
@Descripttion: 
@version: 
@Author: GolLight
@LastEditors: Gollight
@Date: 2020-04-24 21:04:48
@LastEditTime: 2020-04-26 01:25:43
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import FastICA,NMF
from sklearn.preprocessing import StandardScaler

'''
@name: normalize_all
@msg: 将矩阵整体归一化
@param a {array} 二维矩阵 
@return: 
'''
def normalize_all(a):
    amin, amax = a.min(), a.max() # 求最大最小值
    a = (a-amin)/(amax-amin) # (矩阵元素-最小值)/(最大值-最小值)
    return a

'''
@name: 
@test: test font
@msg: Remove genes with greater than some number of expression across all cells
     Also remove genes with 'MT' in its name (these are mitochondrial genes)

@param X {array} N * M matrix N is cell_id,M is gene_id 
@return: 筛选过后的基因
'''
def gene_selet(X,cutoff=0.2,mean = 6):
    # keep_inds_thresh = np.where(np.sum(X,0) > thresh)[0]
    # keep_inds_MT = np.array([i for i in range(len(genes)) if 'MT-' not in genes[i].upper()])
    # keep_inds = np.intersect1d(keep_inds_thresh,keep_inds_MT)    #交集
    # print('Kept %d features for having > %d counts across all cells'%(len(keep_inds),thresh))
    # return X[:,keep_inds],genes[keep_inds]

    #去除所有表达都为0的基因
    # a = normalize_all(X)
    a = X
    both = np.arange(np.shape(a)[1])
    keep_inds_thresh = np.where(np.sum(a,0) != 0)[0] #不全为0的
    #have_0_col = np.where(X == 0)[1]  #有0的列坐标
    have_0_col = list(set(np.where(a == 0)[1]))
    both_zeros = np.where(np.sum(a,0) == 0)[0]  #全为0的列坐标
    none_zeros = np.array(list(set(both) - set(have_0_col))) #不含有0
    keep_inds1 = np.intersect1d(keep_inds_thresh,have_0_col)    #有些表达为0有些不为0的列坐标
    print(len(both_zeros))
    print(len(none_zeros))
    print(len(keep_inds1))
    keep_inds = np.zeros(np.shape(X)[1])
    for i in keep_inds1:
        # zscores = stats.mstats.zscore(a[:,i])
        var = np.var(a[:,i])
        mean_c = np.mean(a[:,i])
        # print(zscores)
        # print(var)
        #keep_inds[i] = zscores > cutoff  #太单调了
        keep_inds[i] = var > cutoff and mean_c > mean  #太单调了
    for i in none_zeros:
        keep_inds[i] = 1
    print('Kept %d features for all cells'%(np.sum(keep_inds)))
    return keep_inds.astype('bool')




#a = np.arange(16).reshape(4,4)
a = [0,0,1,6,
    0,1,0,2,
    0,5,0,3,
    0,1,1,4]
a = np.array(a).reshape(4,4)
#a = normalize_all(a)
print(normalize_all(a))
# bbb = list(set(np.where(a == 0)[1])) #含有0
# none = np.arange(np.shape(a)[1])
# none_z = np.array(list(set(none) - set(bbb))) #不含有0
# # keep_inds = np.zeros(np.shape(a)[1])
# keep_inds_thresh = np.where(np.sum(a,0) > 0)[0]
# both_zeros = np.where(np.sum(a,0) == 0)[0]  #全为0的列坐标
# keep = np.intersect1d(bbb,keep_inds_thresh) #有一些是零
# # print(keep_inds_thresh)
# # print(none)
# # print(bbb)
# print(both_zeros)
# print(keep)
# print(none_z)

# keep_inds = np.zeros(np.shape(a)[1])
# for i in keep:
#     keep_inds[i] = np.mean(a[:,i]) > 0.2
#     print(np.mean(a[:,i]))
# for i in none_z:
#      keep_inds[i] = 1

keep_inds = gene_selet(a,0.02,1)
print(keep_inds)
print(a[:,keep_inds])

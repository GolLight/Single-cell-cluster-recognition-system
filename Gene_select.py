'''
@Descripttion: 
@version: 
@Author: GolLight
@LastEditors: Gollight
@Date: 2020-04-24 21:04:48
@LastEditTime: 2020-05-07 00:44:44
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import FastICA,NMF
from sklearn.preprocessing import StandardScaler
#from-import可以 类比为 深拷贝。在你的模块里导入指定的模块属性
from statsmodels.nonparametric.smoothers_lowess import lowess
#局部加权回归（Lowess）的大致思路是：以一个点xxx为中心，向前后截取一段长度为fracfracfrac的数据，对于该段数据用权值函数www做一个加权的线性回归，记(x,yˆ)为该回归线的中心值，其中yˆ为拟合后曲线对应值。对于所有的n个数据点则可以做出n条加权回归线，每条回归线的中心值的连线则为这段数据的Lowess曲线。
from statsmodels.sandbox.stats.multicomp import multipletests
#多个测试的测试结果和p值校正
import multiprocessing    #multiprocessing包是Python中的多进程管理包。
# import shapely.geometry as geom      
#  #Shapely是一个Python库，用于操作和分析笛卡尔坐标系（直角坐标系和斜坐标系）中的几何对象。
from shapely.geometry import Point
from shapely.geometry import LineString
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

def project_point_to_curve_distance(XP,p):#投影点p到曲线XP距离
    curve = LineString(XP)
    point = Point(p)
    #distance from point to curve  点到曲线的距离
    dist_p_to_c = point.distance(curve)
    return dist_p_to_c    

'''
@name: 
@test: test font
@msg: Remove genes with greater than some number of expression across all cells
     Also remove genes with 'MT' in its name (these are mitochondrial genes)

@param X {array} N * M matrix N is cell_id,M is gene_id 
@return: 筛选过后的基因
'''
def gene_selet(X,k=0.2,cutoff = 2):
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
    mean = np.mean(a)
    # exist = (a > 0) * 1.0
    # num = np.sum(exist)
    sum = np.sum(a)
    # mean = sum/num
    # amin, amax = a.min(), a.max() # 求最大最小值
    for i in keep_inds1:
        # zscores = stats.mstats.zscore(a[:,i])
        # zmax,zmin = zscores.max(),zscores.min()
        var = np.var(a[:,i])
        mean_c = np.mean(a[:,i])
        cmax = a[:,i].max()
        # exist = (a[:,i] > 0) * 1.0
        # num = np.sum(exist)
        csum = np.sum(a[:,i])
        # mean_c = np.sum(a[:,i])
        # print(zscores)
        # print(var)
        #keep_inds[i] = zscores > cutoff  #太单调了
        keep_inds[i] = var > 1 and cmax > mean*cutoff and csum > (k * sum/len(keep_inds))#太单调了
    for i in none_zeros:
        keep_inds[i] = 1
    print('Kept %d features for all cells'%(np.sum(keep_inds)))
    return keep_inds.astype('bool')


def select_variable_genes(X,loess_frac=0.1,percentile=95,n_genes = None,n_jobs = multiprocessing.cpu_count()):

    """
    loess_frac：`float`，可选（默认值：0.1）                  介于0和1之间。在LOWESS函数中估计每个y值时使用的数据分数。
    percentile:百分比：`int'，可选（默认值：95）              介于0到100之间。指定选择基因的百分位数。基因根据其与拟合曲线的距离排序。
    n_genes:`int`，可选（默认：无）           指定选定基因的数目。基因是根据它与拟合曲线的距离排序的。
    n_jobs：`int`，可选（默认值：所有可用CPU）                计算每个基因到拟合曲线的距离时要运行的并行作业数
    """
        
    mean_genes = np.mean(X,axis=0)#细胞内基因表达的平均数
    #mean() 函数定义：numpy.mean(a, axis, dtype, out，keepdims )mean()函数功能：求取均值。经常操作的参数为axis，以m * n矩阵举例：
    #axis 不设置值，对 m*n 个数求均值，返回一个实数
    #axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    #axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    std_genes = np.std(X,ddof=1,axis=0)
    #对各列计算标准差 
    loess_fitted = lowess(std_genes,mean_genes,return_sorted=False,frac=loess_frac)
    #二维变量之间关系局部加权回归（lowess）主要还是处理平滑问题
    #长度frac应该截取多长的作为局部处理，frac为原数据量的比例
    residuals = std_genes - loess_fitted#残差
    
    XP = np.column_stack((np.sort(mean_genes),loess_fitted[np.argsort(mean_genes)]))
    #行合并：np.row_stack()
    #column_stack((a，b))左右根据列拼接
    #np.sort()函数的作用是对给定的数组的元素进行内排序，默认axis = 1 行内排序
    #argsort函数返回的是数组值从小到大的索引值
    
    
    mat_p = np.column_stack((mean_genes,std_genes))#均值，标准差
    
    with multiprocessing.Pool(processes=n_jobs) as pool:
    
        dist_point_to_curve = pool.starmap(project_point_to_curve_distance,[(XP,mat_p[i,]) for i in range(XP.shape[0])])
        #点到曲线距离=
        #pool.starmap这个函数，有两个参数可以传，第一个参数传的是函数，第二个参数传的是数据列表。  
    mat_sign = np.ones(XP.shape[0])
    mat_sign[np.where(residuals<0)[0]] = -1
      
    dist_point_to_curve = np.array(dist_point_to_curve)*mat_sign
    
    if(n_genes is None):
    
        cutoff = np.percentile(dist_point_to_curve,percentile)
        #numpy.percentile(a, q, axis)
        #返回点到曲线距离数组（从小到大排序）里的95%的数据大小
        #percentile=95
        
        id_var_genes = np.where(dist_point_to_curve>cutoff)[0]#可变的基因
        
        id_non_var_genes = np.where(residuals<=cutoff)[0]
        
    else:
        id_var_genes = np.argsort(dist_point_to_curve)[::-1][:n_genes]
        id_non_var_genes = np.argsort(dist_point_to_curve)[::-1][n_genes:]
    
    print(str(len(id_var_genes))+' variable genes are selected')
    return id_var_genes








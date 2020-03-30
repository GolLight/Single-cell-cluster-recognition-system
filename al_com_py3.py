# -*- coding: utf-8 -*
"""t-SNE 对手写数字进行可视化"""
import sys
sys.path.append("F:\VSCode_workspace\dendrosplit")
from time import time
import numpy as np
import matplotlib.pyplot as plt
from time import time 
#from sklearn import datasets
from sklearn.manifold import TSNE
from dendrosplit import split,merge,utils,preprocessing
import pickle,h5py
import pandas as pd
import numpy as np 
from sklearn.metrics import adjusted_rand_score
datadir = u'F:\毕业设计\dendrosplit-master\data\Zeisel\expression_mRNA_17-Aug-2014.txt'

# Save expression matrix
def save_mat_h5f(X,flname):
    h5f = h5py.File(flname, 'w')
    h5f.create_dataset('dataset_1', data=X)
    h5f.close()
# Load data
def load_mat_h5f(flname):
    h5f = h5py.File(flname,'r')
    X = h5f['dataset_1'][:]
    h5f.close()
    return X

def get_data(datadir):
    # Preparing data from Zeisel et al.
    X = pd.read_table(datadir,dtype=str,delimiter='\t',encoding='utf-8')
    X.columns = X.iloc[6] #获取第六行的数据
    labels = X.iloc[7,2:]  #提取标签
    X = X.drop(range(10))
    X.index = X['(none)']
    X = X.drop(X.columns[:2], axis=1).astype(float)
    X.columns.name = 'cell_id'
    X.index.name = 'gene_id'
    return X,labels
 
 
def plot_embedding(x1,x2,ym,ys,labels):
    #plt.figure(figsize=(16,6))

    plt.subplot(2,2,1)
    plt.scatter(x1,x2,edgecolors='none')
    _ = plt.axis('off')
    plt.title('Pre-clustering')
    # plt.show()
    
    #plt.figure()
    plt.subplot(2,2,2)
    split.plot_labels_legend(x1,x2,split.str_labels_to_ints(ys),legend_pos=(1.5,1))
    plt.title('After splitting step')
    # plt.show()
    
    plt.subplot(2,2,3)
    # plt.figure()
    split.plot_labels_legend(x1,x2,ym)
    plt.title('After merging step')

    # plt.show()
    plt.subplot(2,2,4)
    split.plot_labels_legend(x1,x2,split.str_labels_to_ints(labels))
    plt.title('true laels')
    plt.show()
 
def preprocess_funtion():
    X,labels = get_data(datadir)
    save_mat_h5f(X.values.T,'expr.h5')
    # Save genes
    np.savetxt('features.txt',X.index,fmt='%s')
    #save cell-id
    np.savetxt('cell_id.txt',X.columns,fmt='%s')
    #save labels
    np.savetxt('labels.txt',labels,fmt='%s')

    t0 = time()
    Xtsne = preprocessing.sk_tsne(X.values.T)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    print("转换成功")
    np.savetxt('reducedim_coor.txt',Xtsne)

def get_datas():
    X = load_mat_h5f('expr.h5')
    genes = np.loadtxt('features.txt',dtype=str)
    # labels = np.loadtxt('labels.txt',dtype=str) #因为一行有两列，报错
    lines = open('labels.txt').readlines() #打开文件，读入每一行
    fp = open('labels1.txt','w') #打开你要写得文件pp2.txt
    for s in lines:
        fp.write(s.replace(' ','')) # replace是替换，write是写入
    fp.close() # 关闭文件
    labels = np.loadtxt('labels1.txt',dtype=str,delimiter='RAAAWRRRR') #因为一行有两列，报错
    print('%s cells, %s features loaded'%np.shape(X))
    Xtsne = np.loadtxt('reducedim_coor.txt')
    x1,x2 = Xtsne[:,0],Xtsne[:,1]
    return X,Xtsne,x1,x2,genes,labels


def main():
    #print datadir
    #preprocess_funtion()
    X,Xtsne,x1,x2,genes,labels = get_datas()
    X,genes = split.filter_genes(X,genes,0)
    # DropSeq approach to gene selection
    keep_inds = split.dropseq_gene_selection(np.log(1+X),z_cutoff=1.5,bins=5)
    X,genes = X[:,keep_inds],genes[keep_inds]
    t0 = time()
    # Xtsne = preprocessing.sk_tsne(X)
    # x1,x2 = Xtsne[:,0],Xtsne[:,1]
    x1,x2 = preprocessing.low_dimensional_embedding(X)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    print(u"转换成功")
    # np.savetxt('reducedim_coor.txt',Xtsne)
    #plot_embedding(X)
    print('Kept %s features after DropSeq gene selection step.'%(len(X[0])))
    # Get first set of labels. Computing the distance matrix outside the algorithm is highly recommended
    D = split.log_correlation(X) 
    ys,shistory = split.dendrosplit((D,X),
                                preprocessing='precomputed',
                                score_threshold=60,
                                verbose=True,
                                disband_percentile=50)
    #plot_embedding(D)
    # Merge cluster labels
    ym,mhistory = merge.dendromerge((D,X),ys,score_threshold=30,preprocessing='precomputed',
                                verbose=True,outlier_threshold_percentile=90)
    print('Adjusted rand score (ys): %.2f'%(adjusted_rand_score(labels,ys)))
    print('Adjusted rand score (ym): %.2f'%(adjusted_rand_score(labels,ym)))
    plot_embedding(x1,x2,ym,ys,labels)

if __name__ == '__main__':
    main()
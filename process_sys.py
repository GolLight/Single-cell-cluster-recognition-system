'''
@Descripttion: 
@version: 
@Author: GolLight
@LastEditors: Gollight
@Date: 2020-05-13 10:39:24
@LastEditTime: 2020-05-22 02:01:28
'''

import pickle,h5py
import pandas as pd
import numpy as np
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


def preprocess_funtion(get_data,datadir,path):
    X,labels = get_data(datadir)
    save_mat_h5f(X.values.T,path+'expr.h5') #N cells *M genes
    # Save genes
    np.savetxt(path+'features.txt',X.index,fmt='%s')
    #save labels
    np.savetxt(path+'labels.txt',labels,fmt='%s')



def proYan(path):
    X = pd.read_csv(path) #这里自动将第一行作为列索引
    labels = (list)(X.columns[2:126])  #提取标签
    #删除空格
    for i in range(len(labels)):
        labels[i] = labels[i].replace(" ","_")
        labels_ =labels[i].split("#")
        labels[i] = labels_[0]
    #X.columns = X.iloc[0] #
    print(labels[:10])
    X.index = X["Gene_ID"]
    X.index.name = 'gene_id'
    # X = X.drop([0])
    X = X.drop(X.columns[126:], axis=1)
    X = X.drop(X.columns[:2], axis=1).astype(float)
    return X,labels

def proPlloen(path):
    X = pd.read_table(path,dtype=str,delimiter='\t',encoding='utf-8')
    labels = (list)(X.columns)  #提取标签
    #删除空格
    for i in range(len(labels)):
        labels_ =labels[i].split("_")
        labels[i] = labels_[0]+labels_[1]
    #X.columns = X.iloc[0] #
    print(labels[:10])
    # X.index = X["Hi_2338_1"]
    X.index.name = 'gene_id'
    print(X.index[:10])
    # X = X.drop([0])
    X = X.astype(float)
    print(np.shape(X))
    return X,labels



#处理Yan
Yan = r"F:\毕业设计\Yan\nsmb.2660-S2.csv"
Yanpath = r"F:\毕业设计\Yan\\"
Plloen = r"F:\毕业设计\Pollen\NBT_hiseq_linear_tpm_values.txt"
Plloenpath = r"F:\毕业设计\Pollen\\"
preprocess_funtion(proYan,Yan,Yanpath)
#preprocess_funtion(proPlloen,Plloen,Plloenpath)



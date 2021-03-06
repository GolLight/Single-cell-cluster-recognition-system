#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2020/3/26 GolLight
"""
import sys
import os
sys.path.append("dendrosplit")
sys.path.append("consensuscluster")
import wx
# from time import time
import time as times
import numpy as np
import matplotlib.pyplot as plt
from time import time 
#from sklearn import datasets
from sklearn.manifold import TSNE
from dendrosplit import split,merge,utils,preprocessing,clustering
import pickle,h5py
import pandas as pd
import numpy as np 
from sklearn.metrics import adjusted_rand_score
import Gene_select
from consensuscluster import TemplateClassifier
import SC3Units as SC3
import Safeunit as Safe


def count_labels(labels):
    len1 = len(set(labels))
    return len1

def select_samples(labels):
    set_labels = set(labels)
    keep_inst = []
    labels1 = list(labels)
    for i in set_labels:
        keep_inst.append(labels1.index(i))
    return keep_inst
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
    split.plot_labels_legend(x1,x2,ym,legend_pos=(1.5,1))
    plt.title('After merging step')

    # plt.show()
    plt.subplot(2,2,4)
    split.plot_labels_legend(x1,x2,split.str_labels_to_ints(labels),legend_pos=(1.5,1))
    plt.title('true laels')
    plt.show()


def three_plots(x1,x2,Y,ys,ym,legend_pos=(1.5,1),markersize=5,select_inds=None):
    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    split.plot_labels_legend(x1,x2,Y,show_axes=False,legend_pos=legend_pos,
                             markersize=markersize,select_inds=select_inds)
    plt.title('True labels')
    plt.subplot(1,3,2)
    split.plot_labels_legend(x1,x2,ys,show_axes=False,legend_pos=legend_pos,
                             markersize=markersize,select_inds=select_inds)
    plt.title('Labels after split')
    plt.subplot(1,3,3)
    split.plot_labels_legend(x1,x2,ym,show_axes=False,legend_pos=legend_pos,
                             markersize=markersize,select_inds=select_inds)
    plt.title('Labels after merge')
    plt.show()

def three_with_name_plot(x1,x2,ym,ysk,ykmeans,name1,name2,name3,legend_pos=(1.5,1),markersize=5,select_inds=None):
    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    split.plot_labels_legend(x1,x2,ym,show_axes=False,legend_pos=legend_pos,
                             markersize=markersize,select_inds=select_inds)
    plt.title(name1)
    plt.subplot(1,3,2)
    split.plot_labels_legend(x1,x2,ysk,show_axes=False,legend_pos=legend_pos,
                             markersize=markersize,select_inds=select_inds)
    # plt.title('DBSCAN')
    plt.title(name2)
    plt.subplot(1,3,3)
    split.plot_labels_legend(x1,x2,ykmeans,show_axes=False,legend_pos=legend_pos,
                             markersize=markersize,select_inds=select_inds)
    plt.title(name3)
    plt.show()



class HelloFrame(wx.Frame):
    """
    A Frame that says Hello World
    """

    def __init__(self, *args, **kw):
        # ensure the parent's __init__ is called
        super(HelloFrame, self).__init__(*args, **kw)

        # create a panel in the frame
        #pnl = wx.Panel(self)
        # label = wx.StaticText(self, -1, u"当前选择的文件")
        # textBox = wx.TextCtrl(self, -1, style = wx.TE_MULTILINE, size =(20, 30))    
        # sizer = wx.BoxSizer(wx.HORIZONTAL)
        # sizer.Add(label, 0, wx.ALL|wx.ALIGN_CENTRE)
        # sizer.Add(textBox, 1, wx.ALL|wx.ALIGN_CENTRE)
        # self.__TextBox = textBox
        # self.SetSizerAndFit(sizer)
        
        
        # 这个多行的文本框只是用来记录并显示events，不要纠结之
        self.logger = wx.TextCtrl(self,size=(400,500), 
                                 style=wx.TE_MULTILINE | wx.TE_READONLY)
        
        self.logger.AppendText(u"----------欢迎来到单细胞簇识别系统1.0！-------------\n")
        # # 设置sizers11
        # label = wx.StaticText(self, -1, u"当前选择的文件")
        # self.control = wx.TextCtrl(self,size=(140, -1))
        # self.sizer11 = wx.BoxSizer(wx.HORIZONTAL)
        # self.filebutton = wx.Button(self, -1, "选择文件")
        # self.sizer11.Add(label, 0, wx.ALL|wx.ALIGN_CENTRE)
        # self.sizer11.Add(self.control, 1, wx.EXPAND)
        # self.Bind(wx.EVT_BUTTON, self.OnOpenFile, self.filebutton)
        # self.sizer11.Add(self.filebutton, 1, wx.ALL|wx.ALIGN_CENTRE)    

        # 设置sizers12
        label1 = wx.StaticText(self, -1, u"当前选择的文件")
        self.control1 = wx.TextCtrl(self,size=(140, -1))
        self.sizer12 = wx.BoxSizer(wx.HORIZONTAL)
        self.filebutton1 = wx.Button(self, -1, "选择文件")
        self.sizer12.Add(label1, 0, wx.ALL|wx.ALIGN_CENTRE)
        self.sizer12.Add(self.control1, 1, wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.OnOpenFileDir, self.filebutton1)
        self.sizer12.Add(self.filebutton1, 1, wx.ALL|wx.ALIGN_CENTRE) 
        
        # 设置sizers13 prepross
        threshT = wx.StaticText(self, -1, "thresh=")
        z_cutoffT = wx.StaticText(self, -1, "z_cutoff=")
        binsT = wx.StaticText(self, -1, "bins=")
        
        self.threshstr = "0.15"
        self.z_cutoffstr = "0.1"
        self.binsstr = "2"
        
        self.thresh = wx.TextCtrl(self,size=(20, -1),value="0.15")
        self.z_cutoff = wx.TextCtrl(self,size=(20, -1),value="0.1")
        self.bins = wx.TextCtrl(self,size=(20, -1),value="2")

        self.sizer13 = wx.BoxSizer(wx.HORIZONTAL)

        self.prebutton = wx.Button(self, -1, "确定")

        self.sizer13.Add(threshT, 0, wx.ALL|wx.ALIGN_CENTRE)
        self.sizer13.Add(self.thresh, 1, wx.EXPAND)
        self.sizer13.Add(z_cutoffT, 0, wx.ALL|wx.ALIGN_CENTRE)
        self.sizer13.Add(self.z_cutoff, 1, wx.EXPAND)
        self.sizer13.Add(binsT, 0, wx.ALL|wx.ALIGN_CENTRE)
        self.sizer13.Add(self.bins, 1, wx.EXPAND)

        self.Bind(wx.EVT_TEXT, self.Onthresh, self.thresh)
        self.Bind(wx.EVT_TEXT, self.Onz_cutoff, self.z_cutoff)
        self.Bind(wx.EVT_TEXT, self.Onbins, self.bins)
        self.Bind(wx.EVT_BUTTON, self.preprocess, self.prebutton)
        self.sizer13.Add(self.prebutton, 1, wx.ALL|wx.ALIGN_CENTRE) 
        
        #size14 选择降维算法
        self.sizer14 = wx.BoxSizer(wx.HORIZONTAL)
        low_label = wx.StaticText(self, -1, "请选择降维算法")
        self.low_al = 'tsne'
        low_list = ['tsne','ICA','PCA (using SVD)','tSNE and PCA']

        self.lowComBox = wx.ComboBox(self,size=(95, -1),value="tsne", choices=low_list,style=wx.CB_DROPDOWN)
        self.lowbutton = wx.Button(self, -1, "确定降维算法")

        self.Bind(wx.EVT_COMBOBOX, self.OnLowComBox, self.lowComBox)

        self.Bind(wx.EVT_BUTTON, self.DIMENSIONALITY_REDUCTION, self.lowbutton)

        self.sizer14.Add(low_label, 0, wx.ALL|wx.ALIGN_CENTRE)
        self.sizer14.Add(self.lowComBox, 1, wx.EXPAND)
        self.sizer14.Add(self.lowbutton, 1, wx.ALL|wx.ALIGN_CENTRE) 

        #超参数
        split_scoreT = wx.StaticText(self, -1, "split_score=")
        # min_clust_sizeT = wx.StaticText(self, -1, "min_clust_size=")
        # disband_percentileT = wx.StaticText(self, -1, "disband_percentile=")
        merge_scoreT = wx.StaticText(self, -1, "merge_score=")
        # outlier_threshold_percentileT = wx.StaticText(self, -1, "outlier_threshold_percentile=")
        
        self.split_score_str = "60"
        self.merge_score_str = "30"
        
        self.split_score = wx.TextCtrl(self,size=(20, -1),value="60")
        self.merge_score = wx.TextCtrl(self,size=(20, -1),value="30")

        self.sizer15 = wx.BoxSizer(wx.HORIZONTAL)

        self.cluterbutton = wx.Button(self, -1, "确定")

        self.sizer15.Add(split_scoreT, 0, wx.ALL|wx.ALIGN_CENTRE)
        self.sizer15.Add(self.split_score, 1, wx.EXPAND)
        self.sizer15.Add(merge_scoreT, 0, wx.ALL|wx.ALIGN_CENTRE)
        self.sizer15.Add(self.merge_score, 1, wx.EXPAND)

        self.Bind(wx.EVT_TEXT, self.Onsplit_score, self.split_score)
        self.Bind(wx.EVT_TEXT, self.Onmerge_score, self.merge_score)
        
        self.Bind(wx.EVT_BUTTON, self.clutering, self.cluterbutton)
        self.sizer15.Add(self.cluterbutton, 1, wx.ALL|wx.ALIGN_CENTRE)


        #主容器
        self.mainsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer1 = wx.BoxSizer(wx.VERTICAL) 
        self.sizer2 = wx.BoxSizer(wx.HORIZONTAL)    
        self.mainsizer.Add(self.sizer1, 0, wx.GROW)
        self.mainsizer.Add(self.sizer2, 0, wx.GROW)
        # self.sizer1.Add(self.sizer11, 0, wx.GROW)
        self.sizer1.Add(self.sizer12, 0, wx.GROW)
        self.sizer1.Add(self.sizer13, 0, wx.GROW)
        self.sizer1.Add(self.sizer14, 0, wx.GROW)
        self.sizer1.Add(self.sizer15, 0, wx.GROW)      
        self.sizer2.Add(self.logger,1, 0, wx.GROW)

        # 激活sizer
        
        self.SetSizer(self.mainsizer)
        self.SetAutoLayout(True)
        self.mainsizer.Fit(self)     
        


        # create a menu bar
        self.makeMenuBar()

        # and a status bar
        self.CreateStatusBar()
        self.SetStatusText("Welcome to wxPython!")


    def makeMenuBar(self):
        """
        A menu bar is composed of menus, which are composed of menu items.
        This method builds a set of menus and binds handlers to be called
        when the menu item is selected.
        """

        # Make a file menu with Hello and Exit items
        fileMenu = wx.Menu()
        # The "\t..." syntax defines an accelerator key that also triggers
        # the same event
        helloItem = fileMenu.Append(-1, "&保存日志",
                "Help string shown in status bar for this menu item")
        newItem = fileMenu.Append(wx.ID_NEW,"打开")
        fileMenu.AppendSeparator()
        # When using a stock ID we don't need to specify the menu item's
        # label
        exitItem = fileMenu.Append(wx.ID_EXIT)

        # Now a help menu for the about item
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        # Make the menu bar and add the two menus to it. The '&' defines
        # that the next letter is the "mnemonic" for the menu item. On the
        # platforms that support it those letters are underlined and can be
        # triggered from the keyboard.
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&文件")
        menuBar.Append(helpMenu, "&帮助")

        # Give the menu bar to the frame
        self.SetMenuBar(menuBar)

        # Finally, associate a handler function with the EVT_MENU event for
        # each of the menu items. That means that when that menu item is
        # activated then the associated handler function will be called.
        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit,  exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)
        self.Bind(wx.EVT_MENU, self.OnOpenFileDir,newItem)

    def OnExit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)


    def OnHello(self, event):
        """Say hello to the user."""
        # wx.MessageBox("Hello again from wxPython")
        wx.MessageBox(self.logger.GetValue())
        fw = open(self.path+times.strftime("/%Y_%m_%d_%H_%M_%S_", times.localtime())+"logger.txt", 'w')    #将要输出保存的文件地址
        fw.write(self.logger.GetValue())

    def OnAbout(self, event):
        """Display an About Dialog"""
        messtr = u"单细胞簇识别系统\n"
        messtr =messtr +u"这是个识别单细胞簇的层次聚类系统，使用方法如下：\n" 
        messtr =messtr +u"1.将你的数据处理成三个文件expr.h5,features.txt,labels.txt,分别是用h5py.create_dataset创建的N细胞*M基因的细胞表达式矩阵和用np.savetxt函数保存的基因文件和标签文件(注意一行一个不能有空格）放在一个文件夹即可,可以参考prosess_sys.py\n"
        messtr =messtr +u"2.点击选择文件按钮选择文件夹，此时右边会提示成功与否\n"
        messtr =messtr +u"3.thresh表示过滤掉某个基因表达的细胞数少于百分比细胞数的，范围为0-1，为零时不过滤低表达基因\n" 
        messtr =messtr +u"z_cutoff是离散值，bins是分为几份，是整数，将基因按照在所有细胞表达均值分成bins份，然后去掉每一份zscore小于z_cutoff的基因\n"
        messtr =messtr +u"4.可以选择不同的降维算法进行降维\n"
        messtr =messtr +u"5.split_score和merge_score是聚类的两个超参数，一般后者是前者的一半，基于韦尔奇t检查的两个集群之间的距离度量如果大于这个split_score就分裂，小于merge_score就合并(采用的聚类方法是先分裂再合并的)\n"
        messtr =messtr +u"6.ys是层次聚类分裂的结果，ym是分裂再凝聚后的结果，ySC3是SC3算法的结果，ySafe是SAFE算法的结果，yclf是一致性聚类的结果,yKmean是kmeans算法的结果"
        wx.MessageBox(messtr,
                      "About System",
                      wx.OK|wx.ICON_INFORMATION)

    def OnOpenFile(self,event):
        """Open a file"""
        filesFilter = "All files (*.*)|*.*"
        fileDialog = wx.FileDialog(None, "选择数据集文件",
          wildcard = filesFilter, style = wx.FD_SAVE)
        if fileDialog.ShowModal() == wx.ID_OK:
            path = fileDialog.GetPath()
            print(path)
            #self.__TextBox.SetLabel(path)
            self.control.SetValue(path)
            #self.CreateStatusBar()
            self.SetStatusText(u"你已经选择文件"+path)
            self.logger.AppendText(u"你已经选择文件"+path+"\n")
            self.Show(True)
        fileDialog.Destroy()

    def get_datas(self,path):
        if os.path.isfile(path+'/expr.h5'):
            X = load_mat_h5f(path+'/expr.h5')
            self.logger.AppendText('sucessfully load expr!\n')
        elif os.path.isfile(path+'/expr.txt'):
            X = np.loadtxt(path+'/expr.txt',dtype=str).T
            X = np.array(X,dtype = 'float_')        #提取出来类型报错
            self.logger.AppendText('sucessfully load expr!\n')
        else:
            X = None
            self.logger.AppendText('faild load expr!\n')
        
        if os.path.isfile(path+'/features.txt'):
            genes = np.loadtxt(path+'/features.txt',dtype=str)
            self.logger.AppendText('sucessfully load features!\n')
        else:
            genes = None
            self.logger.AppendText('faild load features!\n')
        if os.path.isfile(path+'/labels.txt'):
            self.labels = np.loadtxt(path+'/labels.txt',dtype=str)
            self.logger.AppendText('sucessfully load labels!\n') 
        else:
            self.labels = None 
        print('%s cells, %s features loaded'%np.shape(X))
        self.logger.AppendText('%s cells, %s features loaded\n'%np.shape(X))
        return X,genes

    def OnOpenFileDir(self,event):
        """Open a file"""
        #filesFilter = "All files (*.*)|*.*"
        DirDialog = wx.DirDialog(None, "选择数据集文件目录", style=wx.DD_DEFAULT_STYLE)
        if DirDialog.ShowModal() == wx.ID_OK:
            self.path = DirDialog.GetPath()
            #print path
            #self.__TextBox.SetLabel(path)
            self.control1.SetValue(self.path)
            #self.CreateStatusBar()
            self.SetStatusText(u"你已经选择文件目录"+self.path)
            self.logger.AppendText(u"你已经选择文件目录"+self.path+"\n")
            self.X,self.genes = self.get_datas(self.path)
            self.X_pre,self.genes_pre = self.X,self.genes #不进行特征选择
            # self.Show(True)

        DirDialog.Destroy()
    
    def Onthresh(self,event):
        self.threshstr = event.GetString()
        # self.logger.AppendText('thresh: %s\n' % event.GetString())
    def Onz_cutoff(self,event):
        self.z_cutoffstr = event.GetString()
        # self.logger.AppendText('z_cutoff: %s\n' % event.GetString())
    def Onbins(self,event):
        self.binsstr = event.GetString()
        # self.logger.AppendText('bins: %s\n' % event.GetString())
    def Onsplit_score(self,event):
        self.split_score_str = event.GetString()
        # self.logger.AppendText('split_score_threshold: %s\n' % event.GetString())
    def Onmerge_score(self,event):
        self.merge_score_str = event.GetString()
        # self.logger.AppendText('merge_score_threshold: %s\n' % event.GetString())
    
    def preprocess(self,event):
        # thresh int 保留细胞表达数百分比大于thresh的基因
        # z_cutoff float 离散值
        # bins int  基因根据表达水平放在相等的箱子里
        if self.X is None or self.genes is None:
            wx.MessageBox("数据输入错误，请重新选择文件夹！","ERROR")
            event.Skip()

        self.SetStatusText(u"正在进行特征选择")
        self.logger.AppendText(u"------------正在进行特征选择,请稍后--------\n")
        try:
            thresh = float(self.threshstr)
            z_cutoff = float(self.z_cutoffstr)
            bins = int(self.binsstr)
        except ValueError:
            self.logger.AppendText("invalid input")
            event.Skip()
        self.logger.AppendText('thresh: %s\n' % self.threshstr)
        self.logger.AppendText('z_cutoff: %s\n' % self.z_cutoffstr)
        self.logger.AppendText('bins: %s\n' % self.binsstr)
        
        # X_pre,genes_pre = split.filter_genes(self.X,self.genes,thresh)
        # # DropSeq approach to gene selection
        # keep_inds = split.dropseq_gene_selection(np.log(1+X_pre),z_cutoff=z_cutoff,bins=5)
        X_pre,genes_pre = split.filter_genes(self.X,self.genes,0)
        self.X_pre,self.genes_pre = X_pre,genes_pre
        self.logger.AppendText('Kept %d features for having > %d counts across all cells\n'%(len(X_pre[0]),0))
        if thresh != 0:
            keep_inds_filter = Gene_select.filter_genes(X_pre,min_pct_cells=thresh)
            X_pre,genes_pre = X_pre[:,keep_inds_filter],genes_pre[keep_inds_filter]
        # keep_inds = Gene_select.gene_selet(X_pre,cutoff=z_cutoff,k=bins)
        # keep_inds = Gene_select.select_variable_genes(X_pre,loess_frac=z_cutoff,percentile=bins)
        #keep_inds = Gene_select.filter_genes(X_pre,min_pct_cells=thresh,min_count=bins,expr_cutoff=z_cutoff)
        # keep_inds = Gene_select.filter_and_slect_genes(X_pre,multi = thresh,min_pct_cells = z_cutoff,k=bins)
        #keep_inds = Gene_select.filter_and_slect_genes1(X_pre,z_cutoff = z_cutoff,bins=bins,min_pct_cells = thresh)
            keep_inds = split.dropseq_gene_selection(np.log(1+X_pre),z_cutoff=z_cutoff,bins=bins)
            self.X_pre,self.genes_pre = X_pre[:,keep_inds],genes_pre[keep_inds]
       
            self.logger.AppendText('Kept %s features after DropSeq gene selection step.\n'%(len(self.X_pre[0])))
        self.logger.AppendText(u"----------------特征选择完成-------------\n")
        self.SetStatusText(u"特征选择完成")
       
    def OnLowComBox(self,event):
        self.low_al = event.GetString()
        # self.logger.AppendText('选择算法: %s\n' % event.GetString())
    def DIMENSIONALITY_REDUCTION(self,event):
        self.SetStatusText(u"数据降维中")
        self.logger.AppendText(u"------------正在进行数据降维,请稍后------------\n")
        if self.X_pre is None:
            self.logger.AppendText('ERROR：请先进行特征选择！\n')
            event.Skip()
        if self.low_al == 'tsne':
            t0 = time()
            self.Xtsne = preprocessing.sk_tsne(self.X_pre)
            self.x1,self.x2 = self.Xtsne[:,0],self.Xtsne[:,1]
            t1 = time()
            self.logger.AppendText("t-SNE: %.2g sec\n" % (t1 - t0))  # 算法用时
        elif self.low_al == 'ICA':
            t0 = time()
            self.Xtsne = preprocessing.sk_ica(self.X_pre)
            self.x1,self.x2 = self.Xtsne[:,0],self.Xtsne[:,1]
            t1 = time()
            self.logger.AppendText("ICA: %.2g sec\n" % (t1 - t0))  # 算法用时
        elif self.low_al == 'PCA (using SVD)':
            t0 = time()
            self.Xtsne = preprocessing.sk_pca(self.X_pre)
            self.x1,self.x2 = self.Xtsne[:,0],self.Xtsne[:,1]
            t1 = time()
            self.logger.AppendText("PCA(using SVD): %.2g sec\n" % (t1 - t0))  # 算法用时
        elif self.low_al == 'tSNE and PCA':
            t0 = time()
            self.x1,self.x2 = preprocessing.low_dimensional_embedding(self.X_pre)
            t1 = time()
            self.logger.AppendText("tSNE and PCA: %.2g sec\n" % (t1 - t0))  # 算法用时
            
        else:
            self.logger.AppendText("ERROR:降维算法出错")  #
        self.logger.AppendText(u"------------数据降维完成--------------\n")
        self.SetStatusText(u"数据降维完成")
    
    def clutering(self,event):
        self.SetStatusText(u"正在进行聚类")
        self.logger.AppendText(u"--------正在进行聚类,请稍后-----------\n")
        try:
            split_score = float(self.split_score_str)
            merge_score = float(self.merge_score_str)
        except ValueError:
            self.logger.AppendText("invalid input")
            event.Skip()
        if self.X_pre is None:
            # self.logger.AppendText('ERROR：请先进行特征选择！\n')
            wx.MessageBox('ERROR：请先进行特征选择！\n',"ERROR")
            event.Skip()
        try:
            if self.x1 is None or self.x2 is None:
                # self.logger.AppendText('ERROR：请先进行特征选择！\n')
                wx.MessageBox('ERROR：请先进行数据降维！\n',"ERROR")
                event.Skip()
        except AttributeError:
            wx.MessageBox('ERROR：请先进行数据降维 ！\n',"ERROR")
            event.Skip()
        
        self.logger.AppendText('split_score: %s\n' % self.split_score_str)
        self.logger.AppendText('merge_score: %s\n' % self.merge_score_str)
        D = split.log_correlation(self.X_pre) 
        ys,shistory = split.dendrosplit((D,self.X_pre),
                                preprocessing='precomputed',
                                score_threshold=split_score,
                                verbose=False,
                                disband_percentile=50)
        #plot_embedding(D)
        # Merge cluster labels
        ym,mhistory = merge.dendromerge((D,self.X_pre),ys,score_threshold=merge_score,preprocessing='precomputed',
                                verbose=False,outlier_threshold_percentile=90)

        # 对比算法
        if self.labels is not None:
            cluster_num = count_labels(self.labels)
            ysk = clustering.skDBSCAN(D,eps=0.00005)
            ykmeans = clustering.kMeans(self.X_pre,cluster_num)
            
            clf = TemplateClassifier()
            keep = select_samples(self.labels)
            clf.fit(self.X_pre[keep,:],self.labels[keep])
            yclf = clf.predict(self.X_pre)
            
            t0 = time()
            ySC3 = SC3.SC3(self.X_pre,cluster_num)
            t1 = time()
            ySafe,ykmeans = Safe.Safe_simple(self.X_pre,cluster_num,split_score,merge_score,SC3_lables=ySC3,den_labels=ym)
            self.logger.AppendText("SC3: %.2g sec\n" % (t1 - t0))  # 算法用时
        else:
            ysk = clustering.skDBSCAN(D)
            ykmeans = clustering.kMeans(self.X_pre,8)
            
        
        if self.labels is not None:
            self.logger.AppendText('Adjusted rand score (ys): %.2f\n'%(adjusted_rand_score(self.labels,ys)))
            self.logger.AppendText('Adjusted rand score (ym): %.2f\n'%(adjusted_rand_score(self.labels,ym)))
            self.logger.AppendText('Adjusted rand score (ySC3): %.2f\n'%(adjusted_rand_score(self.labels,ySC3)))
            self.logger.AppendText('Adjusted rand score (ySafe): %.2f\n'%(adjusted_rand_score(self.labels,ySafe)))
            self.logger.AppendText('Adjusted rand score (yconsensus): %.2f\n'%(adjusted_rand_score(self.labels,yclf)))
            self.logger.AppendText('Adjusted rand score (ykmeans): %.2f\n'%(adjusted_rand_score(self.labels,ykmeans)))
            #plot_embedding(self.x1,self.x2,ym,ys,self.labels)
            three_plots(self.x1,self.x2,self.labels,ys,ym)
            three_with_name_plot(self.x1,self.x2,ySafe,ySC3,ykmeans,"Safe","SC3","Kmeans")
        else:
            # three_with_name_plot(self.x1,self.x2,ym,ySC3,ykmeans)
            pass
        self.logger.AppendText(u"-------------聚类完成-------------\n")
        self.SetStatusText(u"聚类完成")
 
if __name__ == '__main__':
    # When this module is run (not imported) then create the app, the
    # frame, show it, and start the event loop.
    app = wx.App()
    frm = HelloFrame(None, title='单细胞簇识别系统')
    frm.Show()
    app.MainLoop()
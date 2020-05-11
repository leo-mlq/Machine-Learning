import sys, os, os.path
import json
import numpy as np


data_set_np = np.array([[0, 0, 0, 0, 'N'],
           [0, 0, 0, 1, 'N'],
           [1, 0, 0, 0, 'Y'],
           [2, 1, 0, 0, 'Y'],
           [2, 2, 1, 0, 'Y'],
           [2, 2, 1, 1, 'N'],
           [1, 2, 1, 1, 'Y']])

def splitDataSet(dataset,featureIndex,value):
    subdataset=[]
    #迭代所有的样本
    for example in dataset:
        if example[featureIndex]==value:
            
            subdataset.append(example)
    return np.delete(subdataset,featureIndex,axis=1)

def dataset_entropy(dataset):
    """
    计算数据集的信息熵
    """
    classLabel=dataset[:,-1]
    labelCount={}
    for i in range(classLabel.size):
        label=classLabel[i]
        labelCount[label]=labelCount.get(label,0)+1     #将所有的类别都计算出来了
    #熵值(第一步)
    cnt=0
    for k,v in labelCount.items():
        cnt += -v/classLabel.size*np.log2(v/classLabel.size)
    
    return cnt


def chooseBestFeature(dataset):
    """
    选择最优特征，但是特征是不包括名称的。
    如何选择最优特征：增益率最小
    """
    #特征的个数
    featureNum=dataset.shape[1]-1
    baseEntropy=dataset_entropy(dataset)
    #print(baseEntropy)
    #设置最大增益值
    maxRatio,bestFeatureIndex=0,None
    #样本总数
    n=dataset.shape[0]  
    for i in range(featureNum):
        #指定特征的条件熵
        featureEntropy=0
        splitInfo=0
        #返回所有子集
        featureList=dataset[:,i]
        featureValues=set(featureList)
       # print(featureValues)
        for value in featureValues:

        	# print(value)
            subDataSet=splitDataSet(dataset,i,value)       
            featureEntropy += subDataSet.shape[0]/n*dataset_entropy(subDataSet) #一个的条件熵
            
            splitInfo+=-subDataSet.shape[0]/n*np.log2(subDataSet.shape[0]/n)
        #print(featureEntropy)
        gainRatio=(baseEntropy-featureEntropy)/splitInfo
        print(gainRatio)
    return bestFeatureIndex #最佳增益


data_set = json.load(open('train.json'))

train_data = data_set['data']
train_label = data_set['label']


train_data_np = np.array(train_data)
train_label_np = np.array([train_label])
#print(train_data_np.shape)
##920 X 19
#print(train_label_np.shape)
##1 X 920


#append label to the end of feature array
data_set_np = np.concatenate((train_data_np,train_label_np.T),axis=1)
#print(data_set_np[122])

# data_set_np = np.array([[0, 0, 0, 0, 'N'],
#            [0, 0, 0, 1, 'N'],
#            [1, 0, 0, 0, 'Y'],
#            [2, 1, 0, 0, 'Y'],
#            [2, 2, 1, 0, 'Y'],
#            [2, 2, 1, 1, 'N'],
#            [1, 2, 1, 1, 'Y']])

chooseBestFeature(data_set_np)
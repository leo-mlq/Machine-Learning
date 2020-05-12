import sys, os, os.path
import json
import numpy as np

data_set = json.load(open('train.json'))

train_data = data_set['data']
train_label = data_set['label']


#print(len(train_data[0]));
#----------------------------------------------
##each feature array is 19 features, 
# 0)(integer) The binary result of quality assessment. 0 = bad quality 1 = sufficient quality.
# 1)(integer) The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack.
# 2-7)(integer) The results of MA detection. Each feature value stand for the
# number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively.
# 8-15)(continuous) contain the same information as 2-7) for exudates. However,
# as exudates are represented by a set of points rather than the number of
# pixels constructing the lesions, these features are normalized by dividing the
# number of lesions with the diameter of the ROI to compensate different image
# sizes.
# 16)(continuous) The euclidean distance of the center of
# the macula and the center of the optic disc to provide important information
# regarding the patient's condition. This feature
# is also normalized with the diameter of the ROI.
# 17)(continuous) The diameter of the optic disc.
# 18)(integer) The binary result of the AM/FM-based classification.


#print(train_data[122])
#-----------------------------------------------
##feature contains both integers and continuous values: [1.0, 1.0, 19.0, 15.0, 14.0, 13.0, 12.0, 9.0, 110.744848, 61.295658, 13.508504, 1.534707, 0.073311, 0.005788, 0.001929, 0.000965, 0.474601, 0.090674, 1.0]

#print(len(train_data));
#-----------------------------------------------
##a total of 920 feature sets 

#print(train_label);
#-----------------------------------------------
##result is either 0 or 1

train_data_np = np.array(train_data)
train_label_np = np.array([train_label])
#print(train_data_np.shape)
##920 X 19
#print(train_label_np.shape)
##1 X 920


#append label to the end of feature array
data_set_np = np.concatenate((train_data_np,train_label_np.T),axis=1)
#print(data_set_np[122])

def col_sparse(dataset, col_inds, attach_labels=True):
	dataset_new = []
	for ind in col_inds:
		dataset_new.append(dataset[:,ind])
	if(attach_labels):
		dataset_new.append(dataset[:,-1])
	return np.array(dataset_new).T

def row_sparse(dataset, row_inds):
	dataset_new = []
	for ind in row_inds:
		dataset_new.append(dataset[ind,:])
	return np.array(dataset_new)

def cal_entropy(labels):
	unique, counts = np.unique(labels, return_counts=True)
	label_set = dict(zip(unique, counts))
	#ex {0:20, 1:30}
	total_labels = len(labels)
	cnt = 0
	for key in label_set:
		cnt += -label_set[key]/total_labels*np.log2(label_set[key]/total_labels)

	return cnt
#print(cal_entropy(data_set_np[:,-1]))
##entropy of whole dataset

def cal_gain_ratio(dataset, feature_set_ind, entropy):

	if(feature_set_ind<=7 or feature_set_ind==18):

		#passed in is col sparsed, first col is feature set, second col is label
		uniques = np.unique(dataset[:,0])
		feature_set_entropy=0;
		iv=0;
		samples_nums = dataset.shape[0]

		for u in uniques:
			#passed in is col sparsed, first col is feature set, second col is label
			row_inds = list(np.where(dataset[:,0]==u)[0])
			sub_dataset = row_sparse(dataset,row_inds)
			feature_set_entropy+=sub_dataset.shape[0]/samples_nums*cal_entropy(sub_dataset[:,-1])
			iv-=sub_dataset.shape[0]/samples_nums*np.log2(sub_dataset.shape[0]/samples_nums)
		gain_ratio = (entropy-feature_set_entropy)/iv
		return gain_ratio
	else:
		#sort by the fist col, feature set
		dataset_sort=dataset[np.argsort(dataset[:, 0])]
		#print(dataset_sort[:,0])
		samples_nums = dataset.shape[0]
		dataset_mid = []
		max_ratio=0
		total=0
		# for i in range(dataset.shape[0]-1):
		# 	dataset_mid.append((float(dataset_sort[i][0])+float(dataset_sort[i+1][0]))/2)
		for i in range(dataset.shape[0]):
			total+=(float(dataset[i][0]))

		##divide continuous value upon average to make it discrete
		dataset_mid.append(total/dataset.shape[0])
		# print(sum)

		#print(dataset_mid)

		for mid in dataset_mid:
			dneg_row_inds=[]
			dpos_row_inds=[]
			for l in range(dataset.shape[0]):
				
				if(float(dataset_sort[:,0][l])<mid):
					dneg_row_inds.append(l)
				elif(float(dataset_sort[:,0][l])>=mid):
					dpos_row_inds.append(l)

			dneg_subset=row_sparse(dataset_sort,dneg_row_inds)
			dpos_subset=row_sparse(dataset_sort,dpos_row_inds)
			
			dneg_ent, dpos_ent=0,0;
			dneg_iv, dpos_iv=0,0;

			# print('neg')
			# print('--------------------')
			if(dneg_subset.shape[0]!=0):
				# print(dneg_subset)
				dneg_uniques = np.unique(dneg_subset[:,0])
				for u in dneg_uniques:
					#passed in is col sparsed, first col is feature set, second col is label
					row_inds = list(np.where(dneg_subset[:,0]==u)[0])
					sub_dataset = row_sparse(dneg_subset,row_inds)
					dneg_ent+=sub_dataset.shape[0]/dneg_subset.shape[0]*cal_entropy(sub_dataset[:,-1])
					dneg_iv-=sub_dataset.shape[0]/dneg_subset.shape[0]*np.log2(sub_dataset.shape[0]/dneg_subset.shape[0])
			# print(dneg_ent)
			# print('pos')
			# print('--------------------')
			if(dpos_subset.shape[0]!=0):
				# print(dpos_subset)
				dpos_uniques = np.unique(dpos_subset[:,0])
				for u in dpos_uniques:
					#passed in is col sparsed, first col is feature set, second col is label
					row_inds = list(np.where(dpos_subset[:,0]==u)[0])
					sub_dataset = row_sparse(dpos_subset,row_inds)
					dpos_ent+=sub_dataset.shape[0]/dpos_subset.shape[0]*cal_entropy(sub_dataset[:,-1])
					dpos_iv-=sub_dataset.shape[0]/dpos_subset.shape[0]*np.log2(sub_dataset.shape[0]/dpos_subset.shape[0])
			
			#print(dpos_ent)
			gain=entropy-(dneg_subset.shape[0]/samples_nums*dneg_ent+dpos_subset.shape[0]/samples_nums*dpos_ent)
			# iv=-dneg_subset.shape[0]/samples_nums*np.log2(dneg_subset.shape[0]/samples_nums)-dpos_subset.shape[0]/samples_nums*np.log2(dpos_subset.shape[0]/samples_nums)
			iv = (dneg_subset.shape[0]/samples_nums)*dneg_iv+(dpos_subset.shape[0]/samples_nums)*dpos_iv
			# print(dneg_iv)
			# print(dpos_iv)
			# print(iv)
			gain_ratio=gain/iv
			if(gain_ratio>max_ratio):
				max_ratio=gain_ratio
		# print('-----------------')
		return (max_ratio)





def cal_best_feature(dataset,features):
	##the last col of dataset is labels
	feature_nums=dataset.shape[1]-1

	curEntropy=cal_entropy(dataset[:,-1])

	max_ratio=0
	best_feature_ind=None;
	for i in features:
		feature_set=col_sparse(dataset,[i])
		gain_ratio=cal_gain_ratio(feature_set,i,curEntropy)
		if(gain_ratio>max_ratio):
			max_ratio=gain_ratio
			best_feature_ind=i
	return best_feature_ind


def areRowsIdentical(dataset):
	return np.all(np.all(dataset == dataset[0,:], axis = 1))

def majorLabel(labels):
	uniques, counts = np.unique(labels, return_counts=True)

	maxCount=0
	maxCounts_ind=None
	for i in range(len(counts)):
		if(counts[i]>maxCount):
			maxCount=counts[i]
			maxCounts_ind=i
	return uniques[maxCounts_ind]



def TreeGenerate(dataset, features_inds):
	data=col_sparse(dataset,features_inds, attach_labels=False)

	labels=dataset[:,-1]
	labels_uniqes=np.unique(labels)
	##all results of dataset are the same
	if(len(labels_uniqes))==1:
		return labels_uniqes[0]


	if(len(features_inds))==0 or areRowsIdentical(data):
		return majorLabel(labels)
		
		

data_set_np = np.array([[0, 0, 0, 0, 'N'],
           [0, 0, 0, 1, 'N'],
           [1, 0, 0, 0, 'Y'],
           [2, 1, 0, 0, 'Y'],
           [2, 2, 1, 0, 'Y'],
           [2, 2, 1, 1, 'N'],
           [1, 2, 1, 1, 'Y']])
features_inds=[0,1,2,3]

import operator
def mayorClass(classList):
    labelCount={}
    for i in range(classList.size):
        label=classList[i]
        labelCount[label]=labelCount.get(label,0)+1
    sortedLabel=sorted(labelCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedLabel[0][0]


#print(TreeGenerate(data_set_np, features_inds))

print(majorLabel(data_set_np[:,-1]))


#features_inds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#cal_best_feature(data_set_np,features)


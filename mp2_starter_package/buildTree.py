import sys, os, os.path
import json
import numpy as np

data_set = json.load(open('train.json'))

train_data = data_set['data']
train_label = data_set['label']

train_data_np = np.array(train_data)
train_label_np = np.array([train_label])
data_set_np = np.concatenate((train_data_np,train_label_np.T),axis=1)

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

def get_averageAndindex(dataset):
	total=0
	for i in range(dataset.shape[0]):
			total+=(float(dataset[i][0]))
	average = total/dataset.shape[0]

	dneg_row_inds=[]
	dpos_row_inds=[]
	for l in range(dataset.shape[0]):
		
		if(float(dataset[:,0][l])<average):
			dneg_row_inds.append(l)
		elif(float(dataset[:,0][l])>=average):
			dpos_row_inds.append(l)
	
	return average, dneg_row_inds, dpos_row_inds
	

def cal_gain_ratio(dataset, feature_set_ind, entropy):

	if(feature_set_ind<=7 or feature_set_ind==18):
		print(feature_set_ind)

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
		print(feature_set_ind)

		#sort by the fist col, feature set
		#dataset_sort=dataset[np.argsort(dataset[:, 0])]

		samples_nums = dataset.shape[0]
		max_ratio=0
		
		# for i in range(dataset.shape[0]-1):
		# 	dataset_mid.append((float(dataset_sort[i][0])+float(dataset_sort[i+1][0]))/2)
		# dataset_mid.append(total/dataset.shape[0])


		##divide continuous value upon average to make it discrete
		average,dneg_row_inds,dpos_row_inds = get_averageAndindex(dataset)

		dneg_subset=row_sparse(dataset,dneg_row_inds)
		dpos_subset=row_sparse(dataset,dpos_row_inds)

		dneg_ent, dpos_ent=0,0;
		dneg_iv, dpos_iv=0,0;


		if(dneg_subset.shape[0]!=0):

			dneg_uniques = np.unique(dneg_subset[:,0])
			for u in dneg_uniques:
				#passed in is col sparsed, first col is feature set, second col is label
				row_inds = list(np.where(dneg_subset[:,0]==u)[0])
				sub_dataset = row_sparse(dneg_subset,row_inds)
				dneg_ent+=sub_dataset.shape[0]/dneg_subset.shape[0]*cal_entropy(sub_dataset[:,-1])
				dneg_iv-=sub_dataset.shape[0]/dneg_subset.shape[0]*np.log2(sub_dataset.shape[0]/dneg_subset.shape[0])

		if(dpos_subset.shape[0]!=0):

			dpos_uniques = np.unique(dpos_subset[:,0])
			for u in dpos_uniques:
				#passed in is col sparsed, first col is feature set, second col is label
				row_inds = list(np.where(dpos_subset[:,0]==u)[0])
				sub_dataset = row_sparse(dpos_subset,row_inds)
				dpos_ent+=sub_dataset.shape[0]/dpos_subset.shape[0]*cal_entropy(sub_dataset[:,-1])
				dpos_iv-=sub_dataset.shape[0]/dpos_subset.shape[0]*np.log2(sub_dataset.shape[0]/dpos_subset.shape[0])
		

		gain=entropy-(dneg_subset.shape[0]/samples_nums*dneg_ent+dpos_subset.shape[0]/samples_nums*dpos_ent)
		# iv=-dneg_subset.shape[0]/samples_nums*np.log2(dneg_subset.shape[0]/samples_nums)-dpos_subset.shape[0]/samples_nums*np.log2(dpos_subset.shape[0]/samples_nums)
		iv = (dneg_subset.shape[0]/samples_nums)*dneg_iv+(dpos_subset.shape[0]/samples_nums)*dpos_iv
		gain_ratio=gain/iv
		if(gain_ratio>max_ratio):
			max_ratio=gain_ratio

	return (max_ratio)





def cal_best_feature(dataset,features):
	##the last col of dataset is labels
	feature_nums=dataset.shape[1]-1

	curEntropy=cal_entropy(dataset[:,-1])

	max_ratio=0
	best_feature_ind=None;
	for i in range(len(features)):
		data=col_sparse(dataset,[features[i]])
		gain_ratio=cal_gain_ratio(data,features[i],curEntropy)
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



def TreeGenerate(dataset, features):
	# data=col_sparse(dataset,features, attach_labels=False)

	labels=dataset[:,-1]
	labels_uniqes=np.unique(labels)
	
	##all results of dataset are the same
	if(len(labels_uniqes))==1:
		return labels_uniqes[0]


	if(len(features))==0 or areRowsIdentical(col_sparse(dataset,features, attach_labels=False)):
		return majorLabel(labels)
	

	best_feature_ind = cal_best_feature(dataset,features)
	best_feature = features[best_feature_ind]

	data = dataset[:,best_feature_ind]

	dTree={best_feature:{}}
	
	data_uniques = np.unique(data)

	for u in data_uniques:
		subfeatures=features.copy()
		row_inds = list(np.where(data==u))[0]
		if(len(row_inds)==0):
			dTree[best_feature][u]=majorLabel(labels)
		else:
			subdataset=row_sparse(dataset,row_inds)
			subfeatures.pop(best_feature_ind)
			# print(features)
			dTree[best_feature][u]=TreeGenerate(subdataset,subfeatures)
	return dTree

		
		

# data_set_np = np.array([[0, 0, 0, 0, 'N'],
#            [0, 0, 0, 1, 'N'],
#            [1, 0, 0, 0, 'Y'],
#            [2, 1, 0, 0, 'Y'],
#            [2, 2, 1, 0, 'Y'],
#            [2, 2, 1, 1, 'N'],
#            [1, 2, 1, 1, 'Y']])
# features=[0,1,2,3]
features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

TreeGenerate(data_set_np,features)

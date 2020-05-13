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
	for i in range(len(dataset)):
			total+=(float(dataset[i]))
	average = total/len(dataset)

	dneg_row_inds=[]
	dpos_row_inds=[]
	for l in range(len(dataset)):
		
		if(float(dataset[l])<average):
			dneg_row_inds.append(l)
		elif(float(dataset[l])>=average):
			dpos_row_inds.append(l)
	
	return average, dneg_row_inds, dpos_row_inds
	

def cal_gain_ratio(dataset, feature_set_ind, entropy):

	# if(feature_set_ind<=7 or feature_set_ind==18):
	if(feature_set_ind<=0 or feature_set_ind==18):

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
		#dataset_sort=dataset[np.argsort(dataset[:, 0])]

		samples_nums = dataset.shape[0]
		max_ratio=0
		
		# for i in range(dataset.shape[0]-1):
		# 	dataset_mid.append((float(dataset_sort[i][0])+float(dataset_sort[i+1][0]))/2)
		# dataset_mid.append(total/dataset.shape[0])


		##divide continuous value upon average to make it discrete
		#passed in is col sparsed, first col is feature set, second col is label
		average,dneg_row_inds,dpos_row_inds = get_averageAndindex(dataset[:,0])

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



def TreeGenerate(dataset, features, depth):
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

	# data = col_sparse(dataset,[best_feature_ind])
	data = dataset[:,best_feature]
	dTree={best_feature:{}}

	#if(best_feature<=7 or best_feature==18):
	if(best_feature<=0 or best_feature==18):
	# data_uniques = np.unique(data[:,0])
		data_uniques = np.unique(data)

		for u in data_uniques:
			tmp = None
			if(u==0.0): tmp='0'
			elif(u==1.0): tmp='1'
			else:
				tmp=u

		

			subfeatures=features.copy()
			row_inds = list(np.where(data==u))[0]
			if(len(row_inds)==0):
				dTree[best_feature][tmp]=majorLabel(labels)
			else:
				subdataset=row_sparse(dataset,row_inds)
				subfeatures.pop(best_feature_ind)
				# print(features)
				dTree[best_feature][tmp]=TreeGenerate(subdataset,subfeatures, depth)
	else:
		# print(best_feature)
		new_d = depth - 1
		average,dneg_row_inds,dpos_row_inds = get_averageAndindex(data)
		cond_s="< "+str(average)
		cond_l=">= "+str(average)

		if(len(dneg_row_inds)==0 or new_d==0):
			return majorLabel(labels)
		else:
			dneg_subset=row_sparse(dataset,dneg_row_inds)
			dTree[best_feature][cond_s]=TreeGenerate(dneg_subset,features, new_d)

		if(len(dpos_row_inds)==0 or new_d==0):
			return majorLabel(labels)
		else:
			dpos_subset=row_sparse(dataset,dpos_row_inds)
			dTree[best_feature][cond_l]=TreeGenerate(dpos_subset,features, new_d)
		

	return dTree

def TreePredict(tree, test_data, features):	
		rootFeatureVal = list(tree.keys())[0]
		testValueAtRoot = test_data[rootFeatureVal]
		featureInd = features.index(testValueAtRoot)
		feature = features[featureInd]

		if(feature<=0 or feature==18):
			nextNode=tree[rootFeatureVal][str(feature)]
			## reach a leaf, make the decision
			if(type(nextNode)).__name__!="dict":
				return nextNode
      ## nextNode is the root node of a substree
			else:
				TreePredict(nextNode, test_data, features)
    

    # if(type)

# data_set_np = np.array([[0, 0, 0, 0, 'N'],
#            [0, 0, 0, 1, 'N'],
#            [1, 0, 0, 0, 'Y'],
#            [2, 1, 0, 0, 'Y'],
#            [2, 2, 1, 0, 'Y'],
#            [2, 2, 1, 1, 'N'],
#            [1, 2, 1, 1, 'Y']])


data_set_np = np.array([[0, 0.5, 'N'],
           [0, 0.9,'N'],
           [1, 1.2,'Y'],
           [2, 1,'Y'],
           [0, 0.85,'Y'],
           [0, 1.5,'N'],
           [0, 2,'Y']])


features = []
for i in range((data_set_np.shape[1]-1)):
  features.append(i)

# test_data = [0, 1, 0, 0]



depth = 3

tree = TreeGenerate(data_set_np,features, depth)
print(tree)
# print(TreePredict(tree, test_data, features))

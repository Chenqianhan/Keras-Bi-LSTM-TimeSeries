import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

#The program is designed to restore missing value, which is -1, in CSV file thatderives from time series remote sensing images. At first, the CSV file has many data type. Thus, we need to clean up the data. Then using KNN classifier to find out k nearest data arrays for target array. Finally, the Bi-LSTM module restorethe data.

#CSV reader
def csv_reader(file,chunkSize=1000,patitions=10**2):
    reader=pd.read_csv(file,iterator=True,header=None)
    chunks=[]
    with tqdm(range(patitions),'Reading...') as t:
        for _ in t:
            try:
                chunk=reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                break

    return pd.concat(chunks,ignore_index=True)

#Cleaning data to convert them into float.
def data_clean(seq):
    index=[]
    for i in range(0,len(seq)):
        if(isinstance(seq[i],str)):
            if not('#IO' in seq[i]):
                ex=seq[i]
                seq=np.delete(seq,i,axis=0)
                seq=np.insert(seq,i,float(ex))

            else:
                seq=np.delete(seq,i,axis=0)
                seq=np.insert(seq,i,-999)
                index.append(i)
    return seq,index

#The header of CSV is the date, however, we transform the date into a list of number which is easier to be handled.
def date_calculate(seq):
    date=[5]
    count=5
    for i in range(0,len(seq)-1):
        day=float(seq[i+1])-float(seq[i])
        count=count+day
        date.append(count)

    return date


#Before further operation, data arrays with missing value require preliminary restoration. 
def linearInsert(X,Y):
#    indexes=[]
    for i in range(0,len(Y)):
        if(Y[i]==-999):
#            indexes.append(i)
            Y=np.delete(Y,i,axis=0)
            coordination=X[i]
            X=np.delete(X,i,axis=0)

            newElement=np.interp(coordination,X,Y)
            Y=np.insert(Y,i,newElement)
            X=np.insert(X,i,coordination)

    return Y


def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

#Knn designed to search k nearest arrays for input 
def classify(input,dataSet,k):
    dataSize=dataSet.shape[0]  #给出训练数据的个数group.shape=（4,2）
    #print('1.Calculate dataSize:',dataSize)
    ##计算欧式距离
    diff=np.tile(input,(dataSize,1))-dataSet
    #print('2.Calculate Distance,diff:',diff)
    squarediff=diff**2
    squareDist=np.sum(squarediff,axis=1)
    #print('3.Sum of square of diff:',squareDist,'Distance done')
    dist=squareDist**0.5
    #print('dist:',dist)

    #对距离进行排序
    sortedDistIndex=np.argsort(dist) #argsort()根据元素的值从小到大进行排序，返回索引
    #print('4.Sort distance,sorteedDistIndex:',sortedDistIndex)
    resembleArray=input
    for i in range(k):
        if(sortedDistIndex[i]!=0):
        #resembleArray=np.insert(resembleArray,i,dataSet[sortedDistIndex[i]])
            resembleArray=np.vstack((resembleArray,dataSet[sortedDistIndex[i]]))
            print('The ',i,'th data is ',dataSet[sortedDistIndex[i]])
            print('Distance is ',dist[sortedDistIndex[i]])

    #axis=1 is delete column, while axis=0 is delete row
    resembleArray=np.delete(resembleArray,0,axis=0)
    return resembleArray

#a=[30,31,32,33,34,35,36,37,38]
#raw_seq=pd.read_csv('data/test.csv').iloc[1,:].values
#raw_dataset=pd.read_csv('data/test.csv').iloc[1:,:].values
#raw_seq=np.delete(raw_seq,a,axis=0)
#raw_seq=raw_seq.tolist()

#raw_dataset=np.delete(raw_dataset,a,axis=1)
#reader=raw_dataset
#date=pd.read_csv('data/test.csv').iloc[0,:].values
reader=np.array(csv_reader('data/2018NDVI.csv'))
reader=np.delete(reader,[0],axis=1)
reader=np.delete(reader,[51],axis=1)

date=reader[0].tolist()
date=date_calculate(date)

raw_seq_dirty=reader[1]
#raw_seq=data_clean(raw_seq).tolist()
raw_seq,seq_index=data_clean(raw_seq_dirty)
raw_seq=raw_seq.tolist()
raw_dataset_dirty=np.delete(reader,[0,1],axis=0)
#raw_dataset=data_clean(raw_dataset).tolist()
print('It is under cleaning process.......................')
raw_dataset=[]
indexes=[seq_index]
for data in raw_dataset_dirty:
    data,data_index=data_clean(data)
    data=data.tolist()
    raw_dataset.append(data)
    indexes.append(data_index)

print('Cleaning process is done!')
#Now parameters: raw_dataset, indexes

K=5
n_steps = 3
n_features = 1
#Process below can be written as a loop. I didn't because my PC sucks.
#for i in range(0,len(dataset)):
i=1
input=raw_dataset[i]
input_index=indexes[i]

unsatisfiedArray=[]
for j in range(0,len(indexes)):
    for b in indexes[j]:
        if(b in input_index):
            unsatisfiedArray.append(j)
            break

#training_samples=np.delete(training_samples,unsatisfiedArray,axis=0)
raw_dataset=np.delete(raw_dataset,unsatisfiedArray,axis=0)
#Now parameters:raw_dataset, input,input_index

#Linear interpolate input sequence
#The reason why read the first array and the rest seperately is to reform a intact dataset in linearInterpolation step.
#After this step, we get a interpolated dataset that fits CSV file
print('Then it is linear interpolation process..............')
#raw_seq is the first array in the dataset, acting as a corrdination.
dataset=[linearInsert(date,raw_seq)]
input=linearInsert(date,input)
#indexset=[index]

for i in raw_dataset:
    #data,index=linearInsert(date,i)
    data=linearInsert(date,i)
    #Mention that dataset is a numpy array, when indexset is not, because indexset cannot be a matrix or there will be an error
    dataset=np.vstack((dataset,data))
    #indexset.append(index)

dataset=np.delete(dataset,0,axis=0)   


print('All data have been interpolated!')
#Reconstruction of each data array in CSV

#    missingValue_indexes=indexset[i]
    #samplesNoInput=np.delete(dataset,i,axis=0)

training_samples=dataset
    #indexesNoInput=np.delete(indexset,i,axis=0)
training_indexes=indexes
    #print('Unprocess indexset: ',indexset)
    #Remove arrays that have missing values in the same timesteps with input
    #But we cannot remove them in loop, instead we should record it.

print('The input data that will be reconstructed is: ',input)

print('KNN classifier is searching k nearest data for input')
#for j in range(0,len(indexset)):
#    for b in indexset[j]:
#        if(b in indexset[i]):
#            unsatisfiedArray.append(j)
#            break


#training_samples=np.delete(training_samples,unsatisfiedArray,axis=0)
print('The k training sampels is:',training_samples)

input_X,input_y=split_sequence(input,n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
input_X = input_X.reshape((input_X.shape[0], input_X.shape[1], n_features))

print('Bidirectional')
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(input_X, input_y, epochs=200, verbose=0)

#Search k nearest data array the normalize for training
if(len(training_samples)!=0):
    k_nearest_samples=classify(input,training_samples,K)
    print('The k training sampels is:',k_nearest_samples)
    #Normalize input and training samples
    for k in k_nearest_samples:
        training_X,training_y=split_sequence(k,n_steps)
        training_X=training_X.reshape((training_X.shape[0],training_X.shape[1],n_features))
        model.fit(training_X,training_y,epochs=200,verbose=0)
        print('An training sample has been trained.')
else:
    print('No nearest data array for training')

# demonstrate prediction
plt.figure()
plt.plot(input, linewidth=1, label='interpolated data')

#Initialize output
outputData=input
round=1

for n in range(0,10):
    print('It is the ',round,' round!')
    count=1
    for j in input_index:
        x_input=np.array([outputData[j-3],outputData[j-2],outputData[j-1]])
        x_input=x_input.reshape((1,n_steps,n_features))
        yhat=model.predict(x_input,verbose=0)
        outputData=np.delete(outputData,j,axis=0)
        outputData=np.insert(outputData,j,yhat)
        print('The ',n+1,'th prediction of ',count,'th element is ',yhat)
       
        if(n==0):
            print('The linearInterpolation of it is ',input[j])
        else:
            print('The previous prediction is ',outputData[j])

        count=count+1
    
    training_X_loop,training_y_loop=split_sequence(outputData,n_steps)
    training_X_loop=training_X_loop.reshape((training_X_loop.shape[0],training_X_loop.shape[1],n_features))
    model.fit(training_X_loop,training_y_loop,epochs=200,verbose=0)
    str='The %s'%(n+3)+'th prediction'
    plt.plot(outputData, alpha=0.5, linewidth=3, label=str)


print('Reconstructed array is:',outputData)
plt.legend()
plt.show()


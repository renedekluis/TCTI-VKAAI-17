from neural_network import *
import pandas as pd
from random import shuffle

dataset 		= 'Datasets/iris.csv'

data 			= pd.read_csv(dataset).as_matrix().tolist()

shuffle(data)
# 80% of data for training
data = data[:int(len(data)*0.8)]

#20% for validation
vData 			= data[int(len(data)*0.8):]



irisNetwork 	= NeuralNetwork('Iris Network', 4, [4,5,3,3], 3, 0.1)
trainDuration 	= 5000
print('TRAINING...')
for i in range(trainDuration):
	if i%10==0:
		print(i,'/',trainDuration)
	for x in data:
		input = x[0:4]
		name = x[-1]
		if name == 'Iris-setosa':
			irisNetwork.Train(input,[0,0,1])
			
		if name == 'Iris-versicolor':
			irisNetwork.Train(input,[0,1,0])
		
		if name == 'Iris-virginica':
			irisNetwork.Train(input,[1,0,0])
	
print('RESULT:')
for x in vData:
	input = x[0:4]
	result = irisNetwork.FeedForward(input)
	a = round(result[0])
	b = round(result[1])
	c = round(result[2])
	result = [int(a),int(b), int(c)]
	if result == [0,0,1]:
		print(result,x[-1],'-->','Iris-setosa')
	if result == [0,1,0]:
		print(result,x[-1],'-->','Iris-versicolor')
	if result == [1,0,0]:
		print(result,x[-1],'-->','Iris-virginica')










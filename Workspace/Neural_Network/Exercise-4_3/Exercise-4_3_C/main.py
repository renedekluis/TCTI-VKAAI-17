from neural_network import *



trainSet = [
	[0,0,0],
	[1,0,1],
	[0,1,1],
	[1,1,0],
]
testSet = [
	[0,0],
	[1,0],
	[0,1],
	[1,1],
]

network = NeuralNetwork('Logic Xor', 2, [2,2], 1, 0.1)

#testNeuron = Neuron('test',2,1,0.1)
print(network.FeedForward([0,1]))
network.Show()

network.Train([0,1],[1])
network.Show()


'''
trainDuration = 10000
for i in range(trainDuration):
	network.Train([0,1],[1])
	print('result ',[0,1],' = ',network.FeedForward([0,1]))
	#for option in trainSet:
		#network.Train([option[0],option[1]],[option[2]])
		#print('result ',[option[0],option[1]],' = ',network.FeedForward([option[0],option[1]]))

'''	
'''
trainDuration = 1000

for times in range(trainDuration):
	for option in trainSet:
		network.Train([option[0],option[1]],[option[2]])
		#print(option,network.Run([option[0],option[1]]))
	#print('\n\n\n')
'''
	
for option in testSet:
	print(option,network.FeedForward([option[0],option[1]]))
	

'''
STATUS UPDATE

Het lijkt goed te gaan voor de opties [1,0], [0,1] en [1,1].
Echter klopt het antwoord nog niet bij [0,0]

Er gaat iets fout met het vullen van de delta errors, hier moet naar gekeken worden.

'''











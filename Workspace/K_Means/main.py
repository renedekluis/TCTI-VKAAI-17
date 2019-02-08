import numpy as np
from KMeans import *
import matplotlib.pyplot as plt

datasetcsv 	= "../Datasets/dataset1.csv"
dayscsv 	= "../Datasets/days.csv"
validationcsv = '../Datasets/validation1.csv'

testSet = np.genfromtxt(dayscsv, delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
validationSet = np.genfromtxt(validationcsv, delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

validationDates = np.genfromtxt(validationcsv, delimiter=';', usecols=[0])


#checkDict = {}
KRange = 10
RunTimes = 10

distList = [None]
for k in range(1,KRange):
	distList.append(0)
	print('K =',k,')')
	
	for times in range(RunTimes):
		km = KMeans(datasetcsv, k)
		km.SetDates(validationDates)
		km.SetLabels()
		foundSeasons = km.RunKMeans(validationSet)
		distances = km.GetDistances()
		print('='*5)
		for x in foundSeasons:
			print(x[0], round(x[1],2),'%')
		totSom = 0
		for clust in distances:
			for punt in distances[clust]:
				totSom += punt
		distList[k] += totSom
	print('\n'*5)


plt.plot(range(0,KRange),distList)
plt.show()





'''
#TestLoop
for x in range(100):
	for i in range(kRange):
		checkDict[i] = []
	procCorrect = []
	distances = []
	for i in range(1,kRange):
		km = KMeans(datasetcsv, i)
		km.SetDates(validationDates)
		km.SetLabels()

		#checkLabels = km.GetSeasons2001(validationDates)
		foundSeasons = km.RunKMeans(validationSet)
		distances.append(km.GetDistances())
		
		
'''		

print('\n'*5)





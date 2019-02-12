import numpy as np
from KMeans import *
import matplotlib.pyplot as plt

datasetcsv 	  = "../Datasets/dataset1.csv"
dayscsv 	  = "../Datasets/days.csv"
validationcsv = '../Datasets/validation1.csv'

testSet = np.genfromtxt(dayscsv, delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
validationSet = np.genfromtxt(validationcsv, delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

validationDates = np.genfromtxt(validationcsv, delimiter=';', usecols=[0])



KRange = 10
RunTimes = 10

distList = [None]
for k in range(1,KRange):
	distList.append(0)
	print('Starting: K =',k)
	
	for times in range(RunTimes):
		km = KMeans(datasetcsv, k)
		km.SetDates(validationDates)
		km.SetLabels()
		foundSeasons = km.RunKMeans(validationSet)
		distances = km.GetDistances()

		totSom = 0
		for clust in distances:
			for punt in distances[clust]:
				totSom += punt
		distList[k] += totSom
	print('\tDone\n')


plt.plot(range(0,KRange),distList)
plt.show()


 






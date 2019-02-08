import numpy as np


class KNearestNeighbours:
	def __init__(self, dataset, k):
		self.data = np.genfromtxt(dataset, delimiter=';',usecols=[1,2,3,4,5,6,7],converters={5:lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
		self.dates = np.genfromtxt(dataset,delimiter=';',usecols=[0])
		self.labels = []
		self.terms = ["FG","TG","TN","TX","SQ","DR","RH"]
		self.k = 0
		if k > len(self.data):
			self.k = len(self.data)
		else:
			self.k = k
		self.SetLabels()
		
	
	def SetLabels(self):
		for label in self.dates:
			if label < 20000301:
				self.labels.append('winter')
			elif label >= 20000301 and label < 20000601:
				self.labels.append('lente') 
			elif label >= 20000601 and label < 20000901:
				self.labels.append('zomer') 
			elif label >= 20000901 and label < 20001201: 
				self.labels.append('herfst')
			else: # from 01-12 to end of year 
				self.labels.append('winter')
	
	
	def FindNearest(self, array):
		tempData = self.data[:]

		afstanden = []
		for rowIdx, row in enumerate(self.data):
			afstanden.append(
				np.sqrt(
					np.abs(array[0]-self.data[rowIdx][0])**2 +
					np.abs(array[1]-self.data[rowIdx][1])**2 +
					np.abs(array[2]-self.data[rowIdx][2])**2 +
					np.abs(array[3]-self.data[rowIdx][3])**2 +
					np.abs(array[4]-self.data[rowIdx][4])**2 +
					np.abs(array[5]-self.data[rowIdx][5])**2 +
					np.abs(array[6]-self.data[rowIdx][6])**2
				)
			)

		indexes = []
		for idx in range(self.k):
			index = afstanden.index(min(afstanden))
			indexes.append(index)
			afstanden[index] = 999999
		return indexes	
		
		
	def GetBest(self,SeasonList):
		seasons = {'zomer':0,'winter':0,'herfst':0,'lente':0}
		for s in SeasonList:
			seasons[s] += 1
			
		key_max = max(seasons.keys(), key=(lambda k: seasons[k]))
		return key_max
		
		
	
	def Run(self, testData, k):
		self.k = k
		result = []
		for rowIdx, array in enumerate(testData):
			nearest = self.FindNearest(testData[rowIdx])

			nearestAsSeasons = []
			for n in nearest:
				nearestAsSeasons.append(self.labels[n])
				
			bestSeason = self.GetBest(nearestAsSeasons)
			result.append(bestSeason)

		return result



























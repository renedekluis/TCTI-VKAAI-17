import numpy as np
import random as r


class KMeans:
	def __init__(self, dataFile, k):
		self.data 	= np.genfromtxt(dataFile, delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
		self.dates 	= np.genfromtxt(dataFile, delimiter=';', usecols=[0])
		self.k = k
		self.labels = []
		self.points = []
		self.clusters = {}
		self.terms = ["FG", "TG" , "TN" , "TX" , "SQ" , "DR" , "RH" ]
		self.clustDists = {}
		
		#self.SetData()
		self.SetLabels()
		self.SetSeasons()

	
	def SetLabels(self):
		if self.dates[0]%20000000 > 10000:
			self.SetSeasons(10000)
		else:
			self.SetSeasons()
	
	def SetDates(self, newDates):
		self.dates = newDates
		
	def SetSeasons(self, adder=0):
		for datum in self.dates:
			if datum < 20000301+adder:
				self.labels.append( (datum,'winter'))
			elif 20000301+adder <= datum < 20000601+adder:
				self.labels.append( (datum,'lente'))
			elif 20000601+adder <= datum < 20000901+adder:
				self.labels.append( (datum,'zomer'))
			elif 20000901+adder <= datum < 20001201+adder: 
				self.labels.append( (datum,'herfst'))
			else: # from 01-12 to end of year 
				self.labels.append( (datum,'winter'))
				
		return self.labels
		
	def GetSeasons2001(self, dates):
		labs = {}
		for datum in dates:
			if datum < 20010301:
				labs.update({datum:'winter'})
				#self.labels.append( (datum,'winter'))
			elif 20010301 <= datum < 20010601:
				labs.update({datum:'lente'})
				#self.labels.append( (datum,'lente'))
			elif 20010601 <= datum < 20010901:
				labs.update({datum:'zomer'})
				#self.labels.append( (datum,'zomer'))
			elif 20010901 <= datum < 20011201: 
				labs.update({datum:'herfst'})
				#self.labels.append( (datum,'herfst'))
			else: # from 01-12 to end of year 
				labs.update({datum:'winter'})
				#self.labels.append( (datum,'winter'))
				
		return labs

		
	def getSeason(self,date):
		#print("KMeans - GetSeason()")
		seizoen = "No season"
		for tup in self.labels:
			if tup[0] == date:
				seizoen = tup[1]

		return seizoen
	
	
	def getDataSet(self):
		return self.data
	
	
	def setK(self,newK):
		self.k = newK
		
		
	def getK(self):
		return self.k
		
	def GetRandomPoints(self):
		return self.points
		

	'''
	This function will set the data needed for the algorithm to start.
	Some dictionaries that are used in this will be initialized here.
	'''
	def SetData(self, data):
		self.data = data
		
		for k in range(self.k):
			self.clustDists[k] = []

	

	'''
	This function will throw the centroids.
	It is return a list per term where the centroids are thrown.
	'''	
	def ThrowCentroids(self):
		#print('KMeans - ThrowCentroids()')
		randomNumbers = []
		for num in range(self.k):
			while True:
				rand = r.randint(0,len(self.data)-1)
				if rand not in randomNumbers:
					randomNumbers.append(rand)
					break
		
		points = []
		for number in randomNumbers:
			points.append(self.data[number][:])
		
		self.points = points
		return self.points


		
	
	'''
	This function will clustered the Data.
	From Each Data Point, the distance to each centroid is calculated.
	Where the data point has the smallest distance to a centroid,
		This data point will be set to be part of the centroid.
	'''	
	def ClusterData(self,centroids):
		#print('KMeans - ClusterData()')
		cluster = {}
		
		for k in range(self.k):
			cluster[k] = {}
			if len(self.clustDists[k])>0:
				self.clustDists[k] = []
		
		for rijIdx, rij in enumerate(self.data):
			afstanden = []
			
			for k in range(self.k):
				afstand = 0
				for termIdx, term in enumerate(self.terms):
					afstand += np.abs(self.data[rijIdx][termIdx]-centroids[k][termIdx])**2
				afstanden.append(np.sqrt(afstand))
			
			cluster[afstanden.index(min(afstanden))].update({self.dates[rijIdx]:self.data[rijIdx]})
			self.clustDists[afstanden.index(min(afstanden))].append(min(afstanden))

		return cluster
	
	
	def GetMostCommon(self,cluster):
		#print('KMeans - GetMostCommon()')
		mostCommonSeasons = []
		
		for k in range(self.k):
			seasonCnt = {}
			for season in cluster[k]:
				if season[1] in seasonCnt:
					seasonCnt[season[1]] +=1
				else:
					seasonCnt[season[1]] = 1
			#print(seasonCnt)		
			sortedList = sorted(seasonCnt, key=seasonCnt.get, reverse=True)[0]
			#print(sortedList)
			#print(seasonCnt[sortedList])
			
			tot = 0
			for idx, i in enumerate(seasonCnt):
				tot += seasonCnt[i]
			#print((sortedList,(seasonCnt[sortedList]/tot)*100))
			if len(seasonCnt) >0:
				mostCommonSeasons.append([sortedList,(seasonCnt[sortedList]/tot)*100])
				#print((sortedList,seasonCnt[sortedList]/tot))
			else:
				mostCommonSeasons.append(None)
		return mostCommonSeasons
	
	def ConvertToSeasons(self, data):
		#print('KMeans - ConvertToSeasons()')
		seasons = []
		convertedList = {}
		for k in range(self.k):
			seasons.append([])
			
		for clustIdx, clust in enumerate(data):
			for dateIdx, date in enumerate(list(data[clust])):
				for labIdx, lab in enumerate(self.labels):
					if lab[0] == date:
						convertedList.update({lab[0]:None})
						seasons[clustIdx].append((lab[0],lab[1]))

		return self.GetMostCommon(seasons)



	'''
	Move the centroids to the average of the data in the found clusters.
	'''	
	def MoveCentroids(self,clusters):
		#print('KMeans - MoveCentroids()')
		valuesClusts = []
		vals = []
		for k in range(self.k):
			valuesClusts.append([])
			vals.append([])
			for idx, term in enumerate(self.terms):
				vals[k].append([])
				
		for idx, term in enumerate(self.terms):
			vals.append([])
		
		newCentroids = []
		for k in range(self.k):
			newCentroids.append([])
			clusterValues = list(clusters[k].values())
			for line in clusterValues:
				for idx, term in enumerate(self.terms):
					vals[k][idx].append(line.tolist()[idx])
			for idx, term in enumerate(self.terms):
				if len(vals[k][idx])>0:
					newCentroids[k].append(np.average(vals[k][idx]))
				else:
					newCentroids[k].append(self.points[k][idx])
			
		return newCentroids
	
	
	def GetDistances(self):
		return self.clustDists
	
	
	
	def RunKMeans(self,data):
		self.SetData(data)
		
		while True:
			canBreak = True
			centroids = self.ThrowCentroids()
			saveCluster1 = self.ClusterData(centroids)
		
			for idx, clust in enumerate(saveCluster1):
				if len(saveCluster1[clust]) < 1:
					canBreak = False
					print('K = ',self.k,') Clust:',idx,'length:',len(saveCluster1[clust]))
				
			if canBreak:
				break
		
		times =0
		while True:
			times+=1
			centroids = self.MoveCentroids(saveCluster1)
			clusteredData2 = self.ClusterData(centroids)
			correct = True
			for idx, line1 in enumerate(saveCluster1):
				if list(saveCluster1[idx]) == list(clusteredData2[idx]):
					pass
				else:
					correct = False

			
			saveCluster1 = clusteredData2
			
			if correct:
				return self.ConvertToSeasons(saveCluster1)
			
			
			
	


			
			
				
			
























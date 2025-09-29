# -*- coding: utf-8 -*-
"""
@author: Rolf van Lieshout
"""

import numpy as np
import math   
import os
import random
import datetime
import pandas as pd

Inizio = datetime.datetime.now()
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")
print(Inizio)
print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")


class Point2D:
    """Class for representing a point in 2D space"""
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y
    
    #method that computes the rounded euclidian distance between two 2D points
    def getDistance(c1,c2): 
        dx = c1.x-c2.x
        dy = c1.y-c2.y
        return math.sqrt(dx**2+dy**2)

class TSP:
    """
    Class for representing a Traveling Salesman Problem
    
    Attributes
    ----------
    nCities : int
        the number of cities
    cities : list of ints
        the cities, all represented by integers
    distMatrix : 2D array
        matrix with all distances between cities. Distance between city i and city j is distMatrix[i-1][j]
    
    """
    def __init__(self,tspFileName):
        """
        Reads a .tsp file and constructs an instance. 
        We assume that it is an Euclidian TSP

        Parameters
        ----------
        tspFileName : str
            name of the file
        """
        points = list() #add all points to list
        f = open(tspFileName)
        for line in f.readlines()[6:-1]: #start reading from line 7, skip last line
            asList = line.split()
            floatList = list(map(float,asList))

            id = int(floatList[0])-1 #convert to int, subtract 1 because Python indices start from 0
            x = floatList[1]
            y = floatList[2]

            c = Point2D(id,x,y)
            points.append(c)
        f.close()
        
        print("Read in all points, start computing distance matrix")

        
        self.nCities = len(points)
        self.cities = list(range(self.nCities))
        
        #compute distance matrix, assume Euclidian TSP
        self.distMatrix = np.zeros((self.nCities,self.nCities)) #init as nxn matrix
        for i in range(self.nCities):
            for j in range(i+1,self.nCities):
                distItoJ = Point2D.getDistance(points[i], points[j])
                self.distMatrix[i,j] = distItoJ
                self.distMatrix[j,i] = distItoJ
        
        print("Finished computing distance matrix")


    def getTour_NN(self,start):
        """
        Performs the nearest neighbour algorithm

        Parameters
        ----------
        start : int
            starting point of the tour

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied.

        """
        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)
    
        print("Start computing NN tour")
        for i in range(self.nCities-1):
            curCity = tour[i]
            closestDist = -1 #initialize with -1
            closestCity = None #initialize with None

            #find closest city not yet in tour
            for j in notInTour:
                dist = self.distMatrix[curCity][j]
                if dist<closestDist or closestCity is None:
                    #update the closest city and distance
                    closestDist = dist
                    closestCity = j
            
            tour.append(closestCity)
            notInTour.remove(closestCity)
        
        print("Finished computing NN tour")

        return tour
    
    
    def getTour_OutlierInsertion(self, start):
        """
        Performs the Outlier Insertion algorithm

        After selecting an initial city, the algorithm creates a tour by joining it with the city with the largest distance from the initial city.
        Then, in each iteration, among all cities not in the tour, we choose the city the furthest to any city in the tour.
        Finally, we insert the selected city in the position that causes the smallest increase in tour length
        
        Parameters
        ----------
        start : int
            starting point of the tour

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied.

        """

        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)
        
        print("Start computing OI tour")
        
        # Select minimum distance city
        
        closestDistance = -1
        closestCity = None
        
        candidateDist = []
        candidateCities = []
        for i in range(self.nCities-1):
            for j in tour:
                for k in notInTour:
                    dist = self.distMatrix[j][k]
                    if dist < closestDistance or closestCity is None:
                        closestDistance = dist
                        closestCity = k

                        candidateDist.append(closestDistance)
                        candidateCities.append(closestCity)
            print("candidate cities are")
            print(candidateCities)
            print("candidate dists are")
            print(candidateDist)
            farthestCity = max(zip(candidateDist, candidateCities))[1]
            tour.append(farthestCity)
            print("tour is")
            print(tour)
            print(farthestCity)
        # Find insertion point



        print("Finished computing OI tour")

        return farthestCity

    def getCitiesCopy(self): 
        return self.cities.copy()
        
    def evaluateSolution(self,tour):
        if self.isFeasible(tour):
            costs = self.computeCosts(tour)
            print("The solution is feasible with costs "+str(costs))
            return costs
        else: 
            print("The solution is infeasible")
    
    def isFeasible(self,tour):
        """
        Checks if tour is feasible

        Parameters
        ----------
        tour : list of integers
            order in which cities are visited. For a 4-city TSP, an example tour is [3, 1, 4, 2]

        Returns
        -------
        bool
            TRUE if feasible, FALSE if infeasible.

        """
        #first check if the length of the tour is correct
        if len(tour)!=self.nCities:
            print("Length of tour incorrect")
            return False
        else: 
            #check if all cities in the tour
            for city in self.cities:
                if city not in tour:
                    return False
        return True
    
    def computeCosts(self,tour):
        """
        Computes the costs of a tour

        Parameters
        ----------
        tour : list of integers
            order of cities.

        Returns
        -------
        costs : int
            costs of tour.

        """
        costs = 0
        for i in range(len(tour)-1):
            costs += self.distMatrix[tour[i],tour[i+1]]
            
        # add the costs to complete the tour back to the start
        costs += self.distMatrix[tour[-1],tour[0]]
        return costs
    
    
##############################################################################   
  
random.seed(10)

# instFilename = "Instances/Small/berlin52.tsp"
# inst = TSP(instFilename)
# startPointNN = 0
# tour = inst.getTour_NN(startPointNN)
# inst.evaluateSolution(tour)


def selectFiles(nSmall, nMedium, nLarge):
    """
    Select a set number of files at random from the directory. It differientiate between the different sizes.

    Parameters
    ----------
    nSmall : integer
        How many small cities are being requested
    nMedium : integer
        How many medium cities are being requested
    nLarge : integer
        How many large cities are being requested

    Returns
    -------
    selectedPaths : list of strings
        a list of all paths selected.

    """

    size = list((nSmall, nMedium, nLarge))

    folderPath_small = "Instances/Small"
    folderPath_medium = "Instances/Medium"
    folderPath_large = "Instances/Large"

    pathList = list((folderPath_small, folderPath_medium, folderPath_large))

    selectedPaths = []
    for i in range(len(pathList)):
        # Select random files in each ith directory
        files = os.listdir(pathList[i])
        randomFiles = random.sample(files, size[i])

        for file in randomFiles:
            # Add the file name to the first part of the directory path
            fullPath = os.path.join(pathList[i], file)
            selectedPaths.append(fullPath)

    return(selectedPaths)


def generateRandomList(size, amount):
    rList = sorted(random.sample(range(size), amount))
    return rList


def generate_output(fileName):
    name = fileName + '.csv'
    fileName = open(name, 'w')
    fileName.write("File,Starting point, Cost\n")
    return fileName


def NNH(inst, filePath, n):
    """
    Run the Nearest Neighbour Heuristics, from n different starting cities in 'inst' instance
    
    ### Parameters
    
    - *inst* (TSP object): an instance of the TSP class
    - *filePath* (string): the path of the file corresponding to the instance
    - *n* (int): the number of starting points to try
    
    ### Returns

    - *df_results* (data frame): A Data Frame containing the following variables:
        
        - File name
        - Start city
        - Cost
        - Tour

    """
    #output_NNH = generate_output('Output_NNH')
    fileName = (filePath.split("/")[-1]).split(".")[-2]
    rList = generateRandomList(len(inst.cities), n)
    sample = [ inst.cities[i] for i in rList] # This is the list of sample cities for which we are going to solve the TSP
    print("\n My sample is " + str(sample))

    results = []
    for i in sample:
        # This "for loop" solves the TSP for each starting point in "inst" instance
        startingPoint = i
        tour = inst.getTour_NN(startingPoint)
        tour_cost = inst.evaluateSolution(tour)
        
        # Create a dictionary with the tour info (easier to convert to df)
        results.append({
            "File name" : fileName,
            "Start city" : i,
            "Cost" : tour_cost,
            "Tour" : tour
        })
        #output_NNH.write(str(fileName) + "," + str(startingPoint) + "," + str(tour_cost) + "\n")
    
    # Convert to DataFrame using pandas
    df_results = pd.DataFrame(results)

    # Save to CSV
    df_results.to_csv("Output_" + fileName + ".csv", index=False)

    print("NNH is done \n\n")
    return df_results









selectedFiles = selectFiles(5, 3, 2)
print("MY INSTANCES ARE")
print(selectedFiles)
print("")

########################################################################

# Create a TSP instance
inst_small = TSP(selectedFiles[0])
inst_medium = TSP(selectedFiles[6])
inst_large = TSP(selectedFiles[8])

## Select only a couple of instances (maybe 1s, 1m and 1l) and do point 2 on them with different starting points.
print("")
print("")
print("2) Run the nearest neighbour heuristic for different starting positions. Comment on the variation in solution quality. (1 point)")
print("")

print(str(selectedFiles[0]))
#fileName = (selectedFiles[0].split("/")[-1]).split(".")[-2]

#print(parsed)
print(selectedFiles[8])
# NNH(inst_small, selectedFiles[0],30)
# NNH(inst_medium, selectedFiles[5], 100)
# NNH(inst_large, selectedFiles[8], 100)


Fine = datetime.datetime.now()

Runtime = Fine - Inizio
print("Runtime = " + str(Runtime))


# Test instance with smallTest1.tsp

testInst_OI = TSP("smallTest1.tsp")
startPointNN = 0
tour_NN = testInst_OI.getTour_NN(startPointNN)
testInst_OI.evaluateSolution(tour_NN)

tour_OI = testInst_OI.getTour_OutlierInsertion(startPointNN)
testInst_OI.evaluateSolution(tour_OI)

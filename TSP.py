# -*- coding: utf-8 -*-
"""
@author: Matteo Meloni
"""

import numpy as np
import math   
import os
import random
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import myFunc # A collection of personal functions separated for tideness

os.system("clear")
os.system("clear")
os.system("clear")
Inizio = time.time()

print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")
print(datetime.datetime.now())
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

            id = int(floatList[0])-1 #convert to int, subtract 1 because Python indices start from 0: city 1 in list, becomes city 0 in code
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

        #print("Start computing NN tour")
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
        
        #print("Finished computing NN tour")

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
        
        #print("Start computing OI tour")

        while len(notInTour) > 0:

            farthestCity = self.findFarthestCity(notInTour, tour)  

            tour = self.cheapestInsertion(tour, farthestCity)
            notInTour.remove(farthestCity)
            
        #print("Finished computing OI tour")
        return tour

    def getTour_GRASPedInsertion(self, start, RCL_size):
        """
        Performs the GRASPed Insertion algorithm:
        This runs the Outlier Insertion algorithm, but randomizes the insertion step: instead of inserting the selected city in the position that causes the smallest increase in tour length, we insert it in a random position within a Restricted Candidate List (RCL) of good options.
        """
        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)
        
        #print("Start computing OI tour")

        while len(notInTour) > 0:

            farthestCity = self.findFarthestCity(notInTour, tour)
            tour = self.cheapestInsertion_GRASPed(tour, farthestCity, RCL_size) 
            notInTour.remove(farthestCity)
            
        #print("Finished computing OI tour")
        return tour
    
    def findFarthestCity(self, notInTour, tour):
        closestDist = -1 # Init shortest distance with lowest possible value
        closestCity = None  
        farthestCity = None
        farthestDist = 0
        candidates = {}
        # Compare, for each city i notInTour, find shortest distance between i and all j-cities in tour.
        # Save, for each i, the city in candidate
        # Then select the farthest candidate as city to insert in the tour
        for i in notInTour:
            for j in tour:
                dist = self.distMatrix[i][j]
                #print(self.distMatrix)
                if dist < closestDist or closestCity is None:
                    closestDist = dist
                    closestCity = j # This isn't actually needed
            
            candidates[i] = float(closestDist)
        
        farthestCity = max(candidates, key=candidates.get) # city chosen to be added in the tour
        

        return farthestCity

    def cheapestInsertion(self, tour, city):
        """
        Insert a given city into a tour in the position that causes the smallest increase in tour length.

        parameters
        ----------
        tour : list of ints
            The tour
        city : int
            The city to be inserted

        Returns
        -------
        tour : list of ints
            The updated tour with the new city inserted
        """
        temporaryTour = tour.copy()
        index = 0
        temporaryTour.insert(index, city)
        
        cost = self.costOfInsertion(temporaryTour, index)
        
        #self.distMatrix[tour[i],tour[i+1]]
        #cost = self.computeCosts(temporaryTour)

        for i in range(1, len(tour)):
            temporaryTour = tour.copy()
            temporaryTour.insert(i, city)
            tempoCost = self.costOfInsertion(temporaryTour, i)

            if tempoCost < cost:
                cost = tempoCost
                index = i

        tour.insert(index, city)
        return tour

    def costOfInsertion(self, tour, i):
        """
        Computes the cost of inserting a city in position i of the tour
        Parameters
        ----------
        tour : list of ints
            The tour with the new city inserted in position i
            i : int
            The position of the newly inserted city
    
        Returns
        -------
        cost : int
            The increase in cost due to the insertion  
        """

        cost = self.distMatrix[tour[i-1],tour[i]] + self.distMatrix[tour[i],tour[i+1]] - self.distMatrix[tour[i-1],tour[i+1]]
        return cost
    
    def cheapestInsertion_GRASPed(self, tour, city, RCL_size):
        """
        Insert a given city into a tour in a random position of a RCL. The RCL is built by selecting the best n positions to insert the city.
        
        Parameters
        ----------
        
        tour : list of ints
            The tour
        city : int
            The city to be inserted

        Returns
        -------

        tour : list of ints
            The updated tour with the new city inserted
        """

        RCL_size = 1 # Size of the Restricted Candidate List
        index = [0] * RCL_size
        temporaryTour = tour.copy()
        index[0] = 0
        temporaryTour.insert(index[0], city)
        
        cost = self.costOfInsertion(temporaryTour, index[0])
        
        
        #self.distMatrix[tour[i],tour[i+1]]
        #cost = self.computeCosts(temporaryTour)

        for i in range(1, len(tour)):
            temporaryTour = tour.copy()
            temporaryTour.insert(i, city)
            tempoCost = self.costOfInsertion(temporaryTour, i)
            #print(i)
            if tempoCost < cost:
                index = [None] + index[:-1] # Asked Copilot how to shift every element up one index.
                cost = tempoCost
                index[0] = i

        selectedIndex = random.sample(index, 1) # Select a random index from the RCL
        selectedIndex = int(selectedIndex[0])
        
        tour.insert(selectedIndex, city)
        return tour
    
    def isTwoOpt(self, tour, starti):
               
        for i in range(starti, len(tour)-1):
            for j in range(i, len(tour)-1):
                if i==j or j==i+1 or i==j+1:
                    continue
                
                city_i = tour[i]
                city_i1 = tour[i+1]
                city_j = tour[j]
                city_j1 = tour[j+1]

                delta = - self.distMatrix[city_i][city_i1] - self.distMatrix[city_j][city_j1] + self.distMatrix[city_i][city_j] + self.distMatrix[city_i1][city_j1]
                
                if delta < -1:
                    # print("The tour is not 2-optimal")
                    # print("Delta = " + str(delta))
                    # print("Inoptimality found at cities " + str(city_i) + " " + str(city_j) + "\n")
                    #print(delta)
                    return False, i, j
        
        if starti != 0:
            Optimality,l,m = self.isTwoOpt(tour,0)
        Optimality = True
        return Optimality,0,0

            
    def makeTwoOpt(self, tour):
        print("Starting 2-opt procedure")
        Optimality,i,j = self.isTwoOpt(tour,0)
        if Optimality == True:
            print("Tour was already 2-optimal")
            return tour
        while Optimality == False:
            #print("2-opt algorithm iteration")
            if Optimality == True:
                print("Tour is now 2-optimal")
                break

            #print("Performing 2-opt exchange between cities " + str(tour[i]) + " and " + str(tour[j]))
            newTour = []
            newTour.extend(tour[0:i+1])
            newTour.extend(reversed(tour[i+1:j+1]))
            newTour.extend(tour[j+1:])
            tour = newTour
            Optimality,i,j = self.isTwoOpt(tour, i)
        
        print("Tour is now 2 optimal")
        return tour


    def getCitiesCopy(self): 
        return self.cities.copy()
        
    def evaluateSolution(self,tour):
        if self.isFeasible(tour):
            costs = self.computeCosts(tour)
            #print("The solution is feasible with costs "+str(costs))
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
  
random.seed(200240124) # set seed for reproducibility

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
        #print(files)
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

def plot_costPerStartingPoint(df):

    cities = range(0,len(list(df["Cost"])))
    costs = list(df["Cost"])

    plt.bar(cities, costs, label='Cost by starting point')
    plt.xlabel("Iteration (starting point)")
    plt.ylabel("Cost")

    meanCost = np.mean(costs)
    meanCost = [meanCost] * 30

    plt.plot(cities, meanCost, 'r:', label='Mean cost')
    plt.legend()
    plt.tight_layout()
    plt.show()

def runForNStartingPoints(filePath, n, method):

    """
    Run the Nearest Neighbour Heuristics, from n different starting cities in 'inst' instance
    
    ### Parameters
    
    - *filePath* (string): the path of the file corresponding to the instance
    - *n* (int): the number of starting points to try
    - *method* (str): What method to use to generate the tour (NN: Neigherst Neighbour, OI: Outlier Insertion)
    
    ### Returns

    - *df_results* (data frame): A Data Frame containing the following variables:
        
        - File name
        - Start city
        - Cost
        - Tour
    """
    
    inst = TSP(filePath)
    #output_NNH = generate_output('Output_NNH')
    fileName = (filePath.split("/")[-1]).split(".")[-2]
    fileSize = (filePath.split("/")[-2])
    rList = generateRandomList(len(inst.cities), n)
    sample = [ inst.cities[i] for i in rList] # This is the list of sample cities for which we are going to solve the TSP
    print("\nSolving for instance: " + str(fileName) + " of size: " + str(fileSize) + " (" + str(len(inst.cities)) + ")")
    print("Applying method: " + str(method))
    print("The selected starting points are: " + str(sample) + "\n")
    RCL_size = 1
    results = []
    counter = 0
    for i in sample:
        AlgorithmStartTime = time.time()
        counter += 1
        if counter % 10 == 0:
            print("Calculating different starting points: " + str(counter) + "/" + str(n))
        # This "for loop" solves the TSP for each starting point in "inst" instance
        startingPoint = i
        tour_cost_beforeLS = None
        runningLS = False
        if method == "NN":
            tour = inst.getTour_NN(startingPoint)
        elif method == "OI":
            tour = inst.getTour_OutlierInsertion(startingPoint)
        elif method == "OI_GRASP":
            if RCL_size is 1:
                RCL_size = input("!!!! Select the size of the RCL (Restricted Candidate List): Type an integer \n !!! ")
            tour = inst.getTour_GRASPedInsertion(startingPoint, RCL_size)  
        elif method == "NN+2OPT":
            runningLS = True
            tour_beforeLS = inst.getTour_NN(startingPoint)
            tour = inst.makeTwoOpt(tour_beforeLS)
        elif method == "OI+2OPT":
            runningLS = True
            tour_beforeLS = inst.getTour_OutlierInsertion(startingPoint)
            tour = inst.makeTwoOpt(tour_beforeLS)
        elif method == "OI_GRASP+2OPT":
            runningLS = True
            if RCL_size is 1:
                RCL_size = input("!!!! Select the size of the RCL (Restricted Candidate List): Type an integer \n !!! ")
            tour_beforeLS = inst.getTour_GRASPedInsertion(startingPoint, RCL_size)
            tour = inst.makeTwoOpt(tour_beforeLS)
        else:
            print(f"Selected method ({method}) is not supported or incorrectly written")
            return
        #print("Done computing from starting point " + str(startingPoint))
        #print(type(tour_co t_beforeLS))
        if runningLS == True:
            #print("Cost before local search was:")
            tour_cost_beforeLS = inst.evaluateSolution(tour_beforeLS)
        tour_cost = inst.evaluateSolution(tour)
        
        AlgorithmEndtTime = time.time()
        # Create a dictionary with the tour info (easier to convert to df)
        results.append({
            "File name" : fileName,
            "Start city" : i,
            "Partial cost" : tour_cost_beforeLS,
            "Cost" : tour_cost,
            "Computation time" : str((AlgorithmEndtTime - AlgorithmStartTime)),
            "RCL size" : RCL_size,
            "Tour" : tour
        })
        #output_NNH.write(str(fileName) + "," + str(startingPoint) + "," + str(tour_cost) + "\n")
    
    # Convert to DataFrame using pandas
    df_results = pd.DataFrame(results)

    # Save to CSV
    df_results.to_csv("Results/Output_" + fileSize + "_" + fileName + "_" + str(method) + ".csv", index=False)

    print("runForNStartingPoints is done \n\n")
    return df_results



selectedFiles = selectFiles(5, 3, 2)
print("MY INSTANCES ARE")
print(selectedFiles)
print("")

########################################################################

## Select only a couple of instances (maybe 1s, 1m and 1l) and do point 2 on them with different starting points.
"""
Point 2: Run the nearest neighbour heuristic for different starting positions. Comment on the variation in solution quality. (1 point)
"""

# Instances are created directly in the function runForNStartingPoints. it takes as input the path of the file, the number of starting points to try and the method to use (NN or OI).

#df_large_NN_100 = runForNStartingPoints(selectedFiles[9], 100, "NN")
#df_medium_OI_100 = runForNStartingPoints(selectedFiles[5], 100, "NN")
#df_small_NN_30 = runForNStartingPoints(selectedFiles[0], 30, "NN")


"""
Point 3: Run the outlier insertion heuristic for different starting positions. Comment on the variation in solution quality. (1 point)
"""

#df_small_OI_30 = runForNStartingPoints(selectedFiles[0], 30, "OI")
#df_medium_OI_100 = runForNStartingPoints(selectedFiles[5], 100, "OI")

def statistics(df):
    print("Statistics for instance: " + str(df["File name"][0]) + "\n")
    print("If GRASP was used, the RCL size was: " + str(df["RCL size"][0]) + "\n")
    print("The mean cost is: " + str(np.mean(list(df["Cost"]))))
    print("The median cost is: " + str(np.median(list(df["Cost"]))))
    print("The standard deviation of the costs is: " + str(np.std(list(df["Cost"]))))
    print("The minimum cost is: " + str(np.min(list(df["Cost"]))))
    print("The maximum cost is: " + str(np.max(list(df["Cost"]))))
    print("Total computation time: " + str(np.sum(list(map(float, df["Computation time"])))) + " seconds")
    print("")

# Comparing results for medium instance from NN and OI
# df_medium_NN_30 = pd.read_csv('Results/Output_Medium_ali535_NN.csv')
# statistics(df_medium_NN_30)
# df_medium_OI_30 = pd.read_csv('Results/Output_Medium_ali535_OI.csv')
# statistics(df_medium_OI_30)


# Test instance with smallTest1.tsp

#testInst_OI = TSP("smallTest1.tsp")
# startPointNN = 0
# tour_NN = testInst_OI.getTour_NN(startPointNN)
# testInst_OI.evaluateSolution(tour_NN)
# print("Tour with NN is " + str(tour_NN))

# tour_OI = testInst_OI.getTour_OutlierInsertion(startPointNN)
# testInst_OI.evaluateSolution(tour_OI)
# print("Tour with OI is " + str(tour_OI))

# Let's try to plot something

#NN_results = []
#OI_results = []

#print(selectedFiles)

# for i in range(0, len(selectedFiles)):
#     print("Running instance: " + str(i))
#     NN_results.append(runForNStartingPoints(selectedFiles[i],10, "NN"))
#     OI_results.append(runForNStartingPoints(selectedFiles[i],10, "OI"))

# runForNStartingPoints(selectedFiles[8],10, "OI")



# plot_costPerStartingPoint(df)



#print(selectedFiles)

"""
Point 4:
    There are two steps in the outlier insertion that can be randomized. Identify these two steps, and develop a GRASP algorithm by randomizing one of them, or both. Implement your algorithm in a method within the TSP class called getTour_GRASPedInsertion. (1 point)
"""

#df_small_GRASPedOI_1 = runForNStartingPoints(selectedFiles[0],1, "OI_GRASP")


# df_medium_GRASPedOI_1_RCL1 = runForNStartingPoints(selectedFiles[6],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL1.to_csv("Results/Output_medium_GRASPedOI_1_RCL1.csv", index=False)

# df_medium_GRASPedOI_1_RCL2 = runForNStartingPoints(selectedFiles[6],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL2.to_csv("Results/Output_medium_GRASPedOI_1_RCL2.csv", index=False)

# df_medium_GRASPedOI_1_RCL3 = runForNStartingPoints(selectedFiles[6],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL3.to_csv("Results/Output_medium_GRASPedOI_1_RCL3.csv", index=False)

# df_medium_GRASPedOI_1_RCL4 = runForNStartingPoints(selectedFiles[6],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL4.to_csv("Results/Output_medium_GRASPedOI_1_RCL4.csv", index=False)

# df_medium_GRASPedOI_1_RCL5 = runForNStartingPoints(selectedFiles[6],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL5.to_csv("Results/Output_medium_GRASPedOI_1_RCL5.csv", index=False)

# df_medium_GRASPedOI_1_RCL10 = runForNStartingPoints(selectedFiles[6],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL10.to_csv("Results/Output_medium_GRASPedOI_1_RCL10.csv", index=False)

# df_medium_GRASPedOI_1_RCL15 = runForNStartingPoints(selectedFiles[6],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL15.to_csv("Results/Output_medium_GRASPedOI_1_RCL15.csv", index=False)


#df_large_GRASPedOI_10 = runForNStartingPoints(selectedFiles[8],1, "OI_GRASP")


statistics(pd.read_csv('Results/Output_medium_GRASPedOI_1_RCL1.csv'))
statistics(pd.read_csv('Results/Output_medium_GRASPedOI_1_RCL2.csv'))
statistics(pd.read_csv('Results/Output_medium_GRASPedOI_1_RCL3.csv'))
statistics(pd.read_csv('Results/Output_medium_GRASPedOI_1_RCL4.csv'))
statistics(pd.read_csv('Results/Output_medium_GRASPedOI_1_RCL5.csv'))
statistics(pd.read_csv('Results/Output_medium_GRASPedOI_1_RCL10.csv'))
statistics(pd.read_csv('Results/Output_medium_GRASPedOI_1_RCL15.csv'))


# df_medium_GRASPedOI_1_RCL1 = runForNStartingPoints(selectedFiles[7],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL1.to_csv("Results/Output_medium2_GRASPedOI_1_RCL1.csv", index=False)

# df_medium_GRASPedOI_1_RCL2 = runForNStartingPoints(selectedFiles[7],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL2.to_csv("Results/Output_medium2_GRASPedOI_1_RCL2.csv", index=False)

# df_medium_GRASPedOI_1_RCL3 = runForNStartingPoints(selectedFiles[7],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL3.to_csv("Results/Output_medium2_GRASPedOI_1_RCL3.csv", index=False)

# df_medium_GRASPedOI_1_RCL4 = runForNStartingPoints(selectedFiles[7],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL4.to_csv("Results/Output_medium2_GRASPedOI_1_RCL4.csv", index=False)

# df_medium_GRASPedOI_1_RCL5 = runForNStartingPoints(selectedFiles[7],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL5.to_csv("Results/Output_medium2_GRASPedOI_1_RCL5.csv", index=False)

# df_medium_GRASPedOI_1_RCL10 = runForNStartingPoints(selectedFiles[7],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL10.to_csv("Results/Output_medium2_GRASPedOI_1_RCL10.csv", index=False)

# df_medium_GRASPedOI_1_RCL15 = runForNStartingPoints(selectedFiles[7],1, "OI_GRASP")
# df_medium_GRASPedOI_1_RCL15.to_csv("Results/Output_medium2_GRASPedOI_1_RCL15.csv", index=False)


# statistics(pd.read_csv('Results/Output_medium2_GRASPedOI_1_RCL1.csv'))
# statistics(pd.read_csv('Results/Output_medium2_GRASPedOI_1_RCL2.csv'))
# statistics(pd.read_csv('Results/Output_medium2_GRASPedOI_1_RCL3.csv'))
# statistics(pd.read_csv('Results/Output_medium2_GRASPedOI_1_RCL4.csv'))
# statistics(pd.read_csv('Results/Output_medium2_GRASPedOI_1_RCL5.csv'))
# statistics(pd.read_csv('Results/Output_medium2_GRASPedOI_1_RCL10.csv'))
# statistics(pd.read_csv('Results/Output_medium2_GRASPedOI_1_RCL15.csv'))

#print(runForNStartingPoints(selectedFiles[0],1, "OI_GRASP"))
print("--------------------------\n--------------------------\n")


"""
Point 5: Write a method in the TSP class called isTwoOpt that takes a tour as input and checks whether it is 2-optimal. Next, write a method called makeTwoOpt that takes in a tour as input and applies 2-exchanges until the tour is 2-optimal (i.e. isTwoOpt returns True). (1 point)
"""

# Test code


# twoOptTest = TSP("point5_test.tsp")
# initialTour = [0, 1, 4, 3, 2, 5, 6]
# print("Initial tour is " + str(initialTour))
# #print(isTwoOpt(twoOptTest, initialTour))


# point5test = TSP("point5_test2.tsp")
# initialTour2 = [0, 1, 2, 3, 9, 8, 6, 7, 5, 4, 10, 11, 12, 13]
# print("Initial tour is " + str(initialTour2))
# point5test.evaluateSolution(initialTour2)

# NewTour = point5test.makeTwoOpt(initialTour2)
# print("Final tour is " + str(NewTour))
# point5test.evaluateSolution(NewTour)
# print("--------------------------\n--------------------------\n")

# print("Trying 2-opt on a small instance\n")
# tour = runForNStartingPoints(selectedFiles[0],1, "NN")["Tour"][0]
# tour = inst_small0.makeTwoOpt(tour)
# inst_small0.evaluateSolution(tour)
# print("--------------------------\n--------------------------\n")

# Try 2-opt on a medium instance
print("Trying 2-opt on a medium instance\n")
#tour_m = runForNStartingPoints(selectedFiles[6],1, "NN+2OPT") # This is a medium instance
#inst_medium6.evaluateSolution(newTour_m)
print("--------------------------\n--------------------------\n")

# Try 2-opt on a large instance
# print("Trying 2-opt on a large instance\n")
tour_l = runForNStartingPoints(selectedFiles[8],1, "NN+2OPT")["Tour"][0] # This is a large instance
# inst_large8.evaluateSolution(tour_l)
# print("--------------------------\n--------------------------\n")



"""
Point 6: Develop an algorithm that combines the GRASPed insertion heuristic with 2-opt and implement this algorithm in the TSP class. Make a figure similar to the one on Slide 26 of the lecture slides on Local Search, with the costs before local search on the x-axis, and after local search on the y-axis. (1 point)
"""

# print("Trying GRASP + 2-opt on a medium instance\n")
# tour_m = runForNStartingPoints(selectedFiles[6],1, "OI_GRASP+2OPT") # This is a medium instance

#df_medium_GRASPedOI_LS_30_RCL3 = runForNStartingPoints(selectedFiles[7],30, "OI_GRASP+2OPT")
#df_medium_GRASPedOI_LS_30_RCL3x2 = runForNStartingPoints(selectedFiles[7],30, "OI_GRASP+2OPT")

#df_twoOpt = pd.read_csv('Results/Output_Medium_rat575_OI_GRASP+2OPT.csv')

#print(df_twoOpt)

def comparisonPlot_beforeAfterLS(df):

    costs_before = list(df_twoOpt["Partial cost"])
    costs_after = list(df_twoOpt["Cost"])

    best_before_x = min(costs_before)
    index_best_before_x = costs_before.index(best_before_x)
    best_before_y = costs_after[index_best_before_x]


    best_after_y = min(costs_after)
    index_best_after_y = costs_after.index(best_after_y)
    best_after_x = costs_before[index_best_after_y]



    plt.plot(costs_before, costs_after, 'bo')
    plt.xlabel("Cost before 2-opt local search")
    plt.ylabel("Cost after 2-opt local search")

    plt.plot(best_before_x, best_before_y, 'ro', label="Best before LS")
    plt.plot(best_after_x, best_after_y, 'go', label="Best after LS")


    plt.legend()
    plt.tight_layout()
    plt.savefig("GRASPed+2opt.png")
    plt.show()

#comparisonPlot_beforeAfterLS(df_twoOpt)




#tour_l = runForNStartingPoints(selectedFiles[8],1, "NN")["Tour"][0] # This is a large instance


################################################################
################################################################
################################################################
########################## RUNTIME #############################
################################################################
################################################################
################################################################

Fine = time.time()
Runtime = Fine - Inizio
print("Runtime: " + str(Runtime) + " seconds")
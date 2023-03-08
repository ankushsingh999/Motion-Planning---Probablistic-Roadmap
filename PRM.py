# Standard Algorithm Implementation
# Sampling-based Algorithms PRM

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math
import numpy as np 
from scipy.spatial import KDTree

# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path

        
    def get_rand_points(self):
        x = np.random.randint(0,self.size_row)
        y = np.random.randint(0,self.size_col)
        point = [int(x), int(y)]
        return point


    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        ### YOUR CODE HERE ###
        #the line between pairs divided into 300 parts
        check = 300
        #finding out the step increment
        incx = (p2[0] - p1[0])/check
        incy = (p2[1] - p1[1])/check
        #initializing the x and y points
        xpt = p1[0]
        ypt = p1[1]
        for i in range(check):
            #check if the point is an obstacle
            if(self.map_array[int(xpt)][int(ypt)] == 0):
                return True
            #incrementing x and y 
            xpt = xpt+incx
            ypt = ypt+incy
        return False


    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        ### YOUR CODE HERE ###
        d = math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))
        return d


    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        ### YOUR CODE HERE ###

        #initializing the x and y
        x = 0
        y = 0        
        #checking for a and y to be inside bounds and incrementing the steps by 100
        while x < self.size_row:
            x += 10
            y = 0
            while y<self.size_col:
                y += 10
                if (x!=300 and y!=300):   
                    #Appending the point only if it is free
                    if (self.map_array[x][y] == 1 ):
                        self.samples.append((x,y))

        

    
    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        #creating an random array of size n_pts x 2
        point = np.random.rand(n_pts,2)
        for i in range(len(point)) :
            #calling function to get a random point 
            p = self.get_rand_points()
            #checking if the sampled point is a obstacle
            if (self.map_array[int(p[0])][int(p[1])] == 1):
                self.samples.append((p[0],p[1]))

    
    def noise_points(self,q1,s):
        q2 = [int(q1[0]+np.random.normal(0,scale=s)), 
              int(q1[1]+np.random.normal(0,scale=s))]
        return q2

    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
        not the number of final sampled points
        check collision and append valid points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        
        for i in range(n_pts):
            #calling the function to get a random point 
            q1 = self.get_rand_points()
            #sampling a gausian noise point around the point with scale = 25 
            q2 = self.noise_points(q1,25)
            #checking if the code is inside bounds
            if((q2[0] > self.size_row-1) or (q2[1] > self.size_col-1)):
                continue
            #checking if either point1 ot point2 is an obstacle = and appending the obstacle free node
            if(self.map_array[q1[0]][q1[1]] == 1 and self.map_array[q2[0]][q2[1]] == 0):
                self.samples.append(tuple(q1))
            elif(self.map_array[q1[0]][q1[1]] == 0 and self.map_array[q2[0]][q2[1]] == 1):
                self.samples.append(tuple(q2))

        
        
            
        ### YOUR CODE HERE ###
        #self.samples.append((0, 0))


    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        for i in range(n_pts):
            #calling a function to get a random point 
            q1 = self.get_rand_points()
            #sampling a gausian noise point around the point with scale = 14
            q2 = self.noise_points(q1,14)
            #checking if the points are inside bounds
            #if(q2[0]<0 and q2[1] <0):
             #   continue
            if((q2[0] > self.size_row-1) or (q2[1] > self.size_col-1)):
                continue
            #checking if both the points are obstacles
            if(self.map_array[q2[0]][q2[1]] == 0 and self.map_array[q1[0]][q1[1]] == 0):
                #finding the midpoint of the obstacles
                mid = [int((q1[0]+q2[0])/2),int((q1[1]+q2[1])/2)]
                #checking if the midpoint is not an obstacle
                if(self.map_array[mid[0]][mid[1]]==1):
                    self.samples.append(tuple(mid))
        ### YOUR CODE HERE ###
        #self.samples.append((0, 0))


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()


    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []
       
        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02), 
        #          (p_id1, p_id2, weight_12) ...]
        pairs = []
        #defining upto how much the graph should search
        radius = 15
        #KDTree for samples
        kdtree = KDTree(self.samples)
        idxs = kdtree.query_pairs(radius)
        #for loop to append the path pairs while checking for collision 
        for p_idxs in idxs:
            dist = self.dis(self.samples[p_idxs[0]], self.samples[p_idxs[1]])
            if self.check_collision((self.samples[p_idxs[0]]),(self.samples[p_idxs[1]])):
                continue
            pairs.append((p_idxs[0], p_idxs[1], dist))     

        # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01), 
        #                                     (p_id0, p_id2, weight_02), 
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of 
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        self.graph.add_nodes_from([])
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))


    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])
        

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, weight_s0), (start_id, p_id1, weight_s1), 
        #                (start_id, p_id2, weight_s2) ...]
        
        start_pairs = []
        goal_pairs = []
        #no.of neghibouring nodes to be explored
        explore = 30
        kdtree = KDTree(self.samples)
        #query for the KDtree returns distance from the querying nodes and the indexes
        dist, p = list(kdtree.query([start, goal], explore))
        # dist = self.dis(self.samples[p1], self.samples[p2])
        #appending the pairs
        for i in range(explore):
            start_pairs.append(('start', p[0][i], dist[0][i]))
            goal_pairs.append(('goal', p[1][i], dist[1][i]))

        # Add the edge to graph
        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)
        
        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        
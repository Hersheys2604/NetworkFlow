from collections import deque

'''
UNLESS OTHERWISE STATED, ASSUME ALL TIME/SPACE COMPLEXITIES MENTIONED ARE WORST CASE.
'''

'''
Author: Harshath Muruganantham
'''
def maxThroughput(connections, maxIn, maxOut, origin, targets):
    '''
    Function Description:
    This function calculates the maximum data throughput from the data centre origin to the data centres specified in
    targets.

    Approach Description:
    First the connections list given is scoured to find the maximum data centre (node) present. This is then used to initialise a network flow graph (based on an adjacency list format).
    The general approach for this solution is to create and intermediary node for each data-centre node, that acts as a bottle-neck in terms of maxIn/maxOut value of that node. this
    intermediary node connects to all the other data centre that the original data centre connected to, with the same capacity.
    For example,
    With the example given, the nodes are [0,1,2,3,4]. We will be creating new nodes, [5,6,7,8,9] which act as intermediary nodes. There will be a connection between each node in the
    original list to a node in the intermediary nodes list (depending on position i.e. 0 -> 5, 1 -> 6, 2 -> 7 etc.). The capactity of this connection will be the bottle-neck value, i.e. the
    minimum of the maximum input of that original node and the maximum output value of the original node. The intermediary nodes will then connect to the corresponding node that the
    orginal node connected to. (i.e using the previos example, if 0 originally had a connection to 1, then 5 would have that SAME connection to 1 RATHER than 0 having it). The capacity of this connection will
    be the same as the capacity of the original connection. To re-iterate the original connection will be maintained through creating a connection from the intermediary node of the data centre from which 
    the communication channel departs to the proper (original node of the data centre to which the communication channel arrives. 
    This is true for all nodes (data-centres) that are not the origin or the targets. For data-centre nodes that are the origin, the same steps above are applied, but with one change.
    This change is that the bottle-neck capacity value from the original node to the intermediary node will NOT be the minimum of the maximum input of that original node and
    the maximum output value of the original node, BUT will rather be just the maximum Output value of that node.
    For the target nodes, the bottle-neck capacity value from the original node to the intermediary node will just be the maximum Input value of the target node.
    The intermediary nodes connected to the target nodes will be connected to the super sink node created using edges with a capacity of infinity. The origin node acts as the source node.
    After this new flow network is set up, then we would have to run Edmond-Karps algorithm to find the maximum possible flow of data between the origin and target nodes. Edmond Karps is just a 
    variation of the Ford-Fulkerson method, but we use BFS to traverse through the graph rather than DFS. Edmond Karps method, like Ford Fulkerson, uses a residual graph network as well, which are added in
    when the Edges are added in. After Edmond Karps' BFS algorithm traverses through the graph as long as there is a possibility for flow to be pushed through. When there is no more possibility for flow to be pushed through 
    from the origin to the target nodes, then the BFS ends. Each time BFS runs, we find the maximum possible flow that can be pushed through those particual edges that were traversed. At the end of the BFS, the maximum flow value
    of that BFS is added to the overall maximum flow value. When Edmond Karps finishes running, this maximum value is then returned. 

    Let |D| denote data centres
    Let |C| denote communication channels

    The worst case time complexity is O(|C|) + O(|2*D|) + O(|D|) + O(|D|) + O(|C|) + O(|wD|*|C|^2) = O(|D|*|C|^2)
    The worst case space complexity is O(2*|D| + |D| + |C|) = O(|D| + |C|)


    Inputs:
        connections: a list of tuples representing the direct communication channels between the data centres.
        maxIn: a list of integers in which maxIn[i] specifies the maximum amount of incoming data that data centre i can process per second
        maxOut: maxOut is a list of integers in which maxOut[i] specifies the maximum amount of outgoing data that data centre i can process per second
        origin: integer of the data centre where the data to be backed up is located
        targets:  a list of data centres that are deemed appropriate locations for the backup data to be stored
    Outputs:
        maximumPossibleFlow: maximum possible data throughput from the data centre origin to the data centres specified in targets.
    '''

    #Initialise Maximum number for nodes (data-centres)
    max = 0

    #Find the maximum number of data centres
    for connection in connections: #O(|C|)
        if connection[0] > max:
            max = connection[0]
        elif connection[1] > max:
            max = connection[1]

    #Create the Flow Network (using the form of an adjacency list)
    graph = FlowNetwork(max, origin)

    #Add the indermediate edge for the orgin node. acts as an acivator node with the capacity between this intermediary node and the origin bening the maximum output value of the origin.
    graph.addIntermediateEdges(origin, maxOut[origin])

    #Add the intermediate edges for the target nodes. Capicity of these edges are the maximum possible input value for these target nodes
    for target in targets: #O(|D|)
        graph.addIntermediateEdges(target, maxIn[target])

        #Create an edge between the intermediary nodes of the targets and the super sink node.
        graph.graph[max + 1 + target].append(EdgeData(max + 1 + target, 2 * max + 2, float("inf"), 0))


    #Add indermediary nodes for all the data centres that are not the origin or the targets.
    for i in range (0, max+1): #O(|D|)
        graph.addIntermediateEdges(i, min(maxIn[i], maxOut[i]))
        
    #Create the connectiosn given to us in our flow network. The connection will be created from the intermediary node of the data centre from which the communication channel departs to the proper node 
    #of the data centre to which the communication channel arrives.
    for connection in connections: #O(|C|)
        graph.addDirectedEdge(connection[0] + max + 1, connection[1], connection[2])
    
    #Run Edmond-Karps Algorithm for calculating the maximum possible flow for the data-centre flow netwrok created. 
    maximumPossibleFlow = 0
    while True:
        currentFlow = graph.bfs() #Run BFS in O(DC^2) time complexity
        if currentFlow == False:
            break #If no more possible pathway, then break this loop of Edmond Karps Algorithm
        else:
            maximumPossibleFlow += currentFlow #Increment maximum flow if BFS returns a flow value


    return maximumPossibleFlow



class FlowNetwork:
    '''
    This class represents the creation of the Flow Network used for question one.
    '''

    def __init__(self, vertices: int, source: int) -> None:
        '''
        Initialises the Flow Network adjacency list and  sets the source and super skink node's locations in the adjacency list.

        Let |D| denote data centres
        Let |C| denote communication channels

        The worst case time complexity is O(|2*D|) = O(|D|)
        The worst case space complexity is O(2*|D| + 2*|C|) = O(|D| + |C|) overall for the graph at the end.

        Inputs:
            vertices: maximum number of data centers
            source: origin data centre (from where the communication is transfered from at the begining)
        '''
        self.graph = [None] * (2*vertices + 3) #Create an adjacency list of size 2* verices + 3 to account for all intermediary nodes and the super sink node.
        self.maxV = vertices
        self.visited = [False for _ in range (0, 2* vertices + 3)] #create visited array needed for BFS
        self.source = source
        self.superSink = 2 * vertices + 2


    def addDirectedEdge(self,dataCentreFrom:int, dataCentreTo: int, capacity: int):
        '''
        This is used for adding edges and capacities as connection. This is used to add the connection between an intermediary node and an original node
        that represenets the original connections given.
        
        Let |D| denote data centres
        Let |C| denote communication channels

        The worst case time complexity is O(1)
        The worst case space complexity is O(2*|D| + 2*|C|) = O(|D| + |C|) overall for the graph at the end.

        Inputs:
            dataCentreFrom: Integer representing the data centre where this connection is from (a intermediary node).
            dataCentreTo: Integer representing the data centre where this connection is to (a original node).
            capacity: capaciity integer value of this connection (maximum data throughput).
        
        '''
        #Initialise this position in the adjacency list array
        if self.graph[dataCentreFrom] is None:
            self.graph[dataCentreFrom] = []
        if self.graph[dataCentreTo] is None:
            self.graph[dataCentreTo] = []
        #Create edges and residual edges to be added to the adjacency list
        edgeNormal = EdgeData(dataCentreFrom, dataCentreTo, capacity,0)
        edgeResidual = EdgeData(dataCentreTo, dataCentreFrom, 0, 0)
        #Store the residual edges for each edge in each edge
        edgeNormal.setResidual(edgeResidual)
        edgeResidual.setResidual(edgeNormal)
        #Store the residual edges for each edge in each edge
        self.graph[dataCentreFrom].append(edgeNormal)
        self.graph[dataCentreTo].append(edgeResidual)
    
    def addIntermediateEdges(self, dataCentreFrom: int, capacity: int):
        '''
        This is used for adding edges between the original node and its corresponding intermediary Node. The capacity value will be the maximum output value for the origin node,
        the maximum input value for the target nodes and the minimum of the maximum output values and the maximum input values for the other nodes.

        Let |D| denote data centres
        Let |C| denote communication channels

        The worst case time complexity is O(1)
        The worst case space complexity is O(2*|D| + 2*|C|) = O(|D| + |C|) overall for the graph at the end.

        Inputs:
            dataCentreFrom: Integer representing the data centre where this connection is from (the original node).
            capacity: capaciity integer value of this connection (maximum data throughput).
        '''
        #The indermediary node position will always be equal to the position of the original node after the maximum position of all original nodes.
        dataCentreTo = self.maxV + 1 + dataCentreFrom

        #Initialise this position in the adjacency list array
        if self.graph[dataCentreTo] is None:
            self.graph[dataCentreTo] = []
        else: #If there is already an intermediary edge there, then end this function.
            return
        if self.graph[dataCentreFrom] is None:
            self.graph[dataCentreFrom] = []
        #Create edges and residual edges to be added to the adjacency list
        edgeNormal = EdgeData(dataCentreFrom, dataCentreTo, capacity,0)
        edgeResidual = EdgeData(dataCentreTo, dataCentreFrom, capacity, capacity)
        #Store the residual edges for each edge in each edge
        edgeNormal.setResidual(edgeResidual)
        edgeResidual.setResidual(edgeNormal)
        #Store the residual edges for each edge in each edge
        self.graph[dataCentreFrom].append(edgeNormal)
        self.graph[dataCentreTo].append(edgeResidual)
        

    def ifVisited(self, index:int) -> bool:
        '''
        This function returns true if the node in the index value inputed has already been visitied in this iteration of the BFS. False is returned otherwise

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Inputs:
            index: Integer representing the data source to be checked for
        Outputs:
            boolean true if the node in the index value inputed has already been visitied in this iteration of the BFS. False is returned otherwise

        '''
        return self.visited[index]
    
    def clearVisited(self):
        '''
        Resets the visited array for use in the next BFS iteration.

        Let |D| denote data centres
        Let |C| denote communication channels

        The worst case time complexity is O(2*|D|) = O(|D|)
        The worst case space complexity is O(2*|D|) = O(|D|)

        '''
        self.visited = [False for _ in range (0, 2* self.maxV + 3)]
    
    def addVisisted(self, index: int):
        '''
        Sets the visited value for the index value to be true.

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Inputs:
            index: Integer representing the data source to be modified in visited array
        
        '''
        self.visited[index] = True

    def getConnections(self, index: int):
        '''
        Returns all the connections departing from that data-centre node

        The worst case time complexity is O(1)

        Inputs:
            index: Integer representing the data source for whose connection we need
        Outputs:
            all the connections departing from that data centre. (None is not any)
        '''
        return self.graph[index]


    def bfs(self):
        '''
        Run BFS, used to iterate through the flow netwrok to from the orgin to the super sink node (if possible) and return the maximum possible flow that can be pushed in this direction

        Let |D| denote data centres
        Let |C| denote communication channels

        The worst case time complexity is  O(2|D|) + O(|2*D|*|2*C|^2) = O(|D|*|C|^2)

        Outputs:
            minflow: the maximum possible flow (minimum capacity) that can be pushed in from the origin(Source) node to the super sink node.
        '''
        #Initiate/Clear Visited Array
        self.clearVisited()
        queue = deque()
        self.addVisisted(self.source)
        #Start at Source Node
        queue.append(self.source)
        #Initaiate a predecessor array used for backtracking to find chosen/given path
        pred = [None for _ in range (0, 2* self.maxV + 3)]
        while (len(queue) > 0):
            current = queue.popleft()

            if current == self.superSink: #End of traversal, reached super sink node
                break
            
            #For each edge connected from the current node, see if the edge the node leads to has already been visited in this iteration.
            #If it has, then continue with the loop.
            #If it has not been visited, then add this node to the visited array, add this node to the queue, and update the predecessor array with this edge at the position of this node.
            for edge in self.getConnections(current):
                if edge.getResidualFlow() > 0 and not self.ifVisited(edge.getDestinationVertex()):
                    self.addVisisted(edge.getDestinationVertex())
                    queue.append(edge.getDestinationVertex())
                    pred[edge.getDestinationVertex()] = edge

        #If the super sink node cannot be reached, end of Edmond Karps Traversal
        if pred[self.superSink] == None:
            return False

        #Find the maximum possible flow (the minimum capacity, so hence was named the minimum flow) of the path we took to reach the super sink node.
        #by backtracking using the predecessor array
        minFlow = float("inf")
        edge = pred[self.superSink]
        while edge != None:
            minFlow = min(minFlow, edge.getResidualFlow())
            edge = pred[edge.getThisVertex()]
               
        #Update the flow values in the edges and the residual edges for all node we visited in the current path from source (origin) to super-sink.
        edge = pred[self.superSink]
        while edge != None:
            edge.updateFlowValues(minFlow)
            edge = pred[edge.getThisVertex()]


        return minFlow #Return the flow value calculated.
        
class EdgeData:
    '''
    This class represents the connections between data centres.
    '''

    def __init__(self, this: int, vertex: int, capacity: int, flow: int):
        '''
        Initialise the connection values stored in this connection
        
        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Inputs:
            this: The DataCentre this connection is from
            vertex: the data centre this connection is to
            capacity: the capacity of this connection
            flow: The initial flow of this connection
        '''
        self.this = this
        self.vertex = vertex
        self.capacity = capacity
        self.checkCapacity = capacity
        self.flow = flow
        self.residualEdge = None

    def hasResidual(self) -> bool:
        '''
        Returns true of this node has an residual edge.

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Outputs:
            boolean value True if this ege has an residual edge. False if it doesnt
        '''
        return not self.residualEdge == None
    
    def getResidualFlow(self):
        '''
        Returns the residual flow of this connection. Used for BFS.

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Outputs:
            Residual flow of this connection (integer value of capacity - flow)
        '''
        return self.getCapacity() - self.getFlow()
    
    def updateFlowValues(self, minFlow: int):
        '''
        Update the flows of the edges with the flow taken up by the path. Increase in flow in normal edge means decrease in flow in residual edge.

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Inputs:
            minFlow: the maximum possible flow (minimum capacity) that can be pushed in from the origin(Source) node to the super sink node.
        '''
        self.flow += minFlow
        if self.hasResidual():
            self.residualEdge.flow -= minFlow
        
    def setResidual(self, residual):
        '''
        Sets the residual edge of this connection.

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Inputs:
            residual: Edge Data object containing this edge's residual edge
        '''
        self.residualEdge = residual
    
    def getThisVertex(self) -> int:
        '''
        Returns the DataCentre this connection is from

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Outputs:
            this: The DataCentre this connection is from
        '''
        return self.this

    def getDestinationVertex(self) -> int:
        '''
        Returns the data centre this connection is to

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Outputs:
            vertex: The DataCentre this connection is to

        '''
        return self.vertex
    
    def getCapacity(self) -> int:
        '''
        Returns the capacity of this connection
        
        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Outputs:
            capacity: the capacity of this connection
        '''
        return self.capacity
    
    def getFlow(self) -> int:
        '''
        returns the initial flow of this connection

        The worst case time complexity is O(1)
        The worst case space complexity is O(1)

        Outputs:
            flow: the flow of this connection
        '''
        return self.flow






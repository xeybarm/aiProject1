
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import random
from collections import defaultdict

alg = input("Enter D or A (Dijkstra or A*): ")
start = int(input("Enter start node: "))
dest = int(input("Enter destination node: "))

nodesStr = []
nodes = []
labels=[]
edgesStr = []
edges = []

#read nodes and edges from the file
with open('p1_graph.txt', 'r') as f: 
    for line in f:
        if not line.startswith("#"):
            numCommas = line.count(',')
            if numCommas == 1:
                nodesStr.append(line) #store nodes in an array
            else:
                edgesStr.append(line) #store edges in an array

#neglect unneccesary parts in the file
nodesStr = nodesStr[:len(nodesStr)-2] 
edgesStr = edgesStr[1:len(edgesStr)-1] 

#add vertexID and squareID of each node to a list
for i in range(len(nodesStr)):
    label = nodesStr[i].split(',')[0] #vertexID of each node
    loc = int(nodesStr[i].split(',')[1]) #squareID of each node

    #genarate random numbers for placing nodes randomly in each square 
    randomNumX = random.randint(0, 10) 
    randomNumY = random.randint(0, 10)

    #change squareID to (x, y) format 
    if loc%10 == 0:
        x = randomNumX
        y = (90-loc)+randomNumY
    else:
        x = (loc%10)*10+randomNumX
        y = ((100-loc)-(100-loc)%10)+randomNumY
    nodes.append((x, y)) #add locations of nodes to a list
    labels.append(label) #add labels of nodes to a list

#add edges and their weights to a list
for j in range(len(edgesStr)): 
   v1 = int(edgesStr[j].split(',')[0])
   v2 = int(edgesStr[j].split(',')[1])
   w = int(edgesStr[j].split(',')[2])
   edges.append((str(v1), str(v2), w))

#add nodes and edges to created graph 
G = nx.Graph()
G.add_nodes_from(labels) 
G.add_weighted_edges_from(edges) 

#match vertexID and squareID of each node
pos = {label: nodes[int(label)] for label in labels}

#draw 10x10 grid (100 squares)
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(10))

#label edges
edgeLabels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels, ax=ax, font_size=5)

#draw network
nx.draw(G,pos=pos, node_size=100, ax=ax, with_labels=True, font_size=8)

#Dijkstra algorithm 
def dijkstra(graph, s, d):
    vertices = []
    for i in graph:
        vertices.append(i)
        vertices += [x[0] for x in graph[i]]

    p = set(vertices)
    vertices = list(p)
    wSum = dict()
    cameFrom = dict()

    for i in vertices:
        wSum[i] = float('inf')
        cameFrom[i] = None

    wSum[s] = 0

    while p:
        u = min(p, key=wSum.get)
        p.remove(u)

        if d is not None and u == d:
            return wSum[d], cameFrom

        for v, w in graph.get(u, ()):
            alt = wSum[u] + w
            if alt < wSum[v]:
                wSum[v] = alt
                cameFrom[v] = u

    return wSum, cameFrom

#A-Star algorithm
def heuristic(start, goal):
    start = str(pos[str(start)])
    goal = str(pos[str(goal)])

    startX = int(start.split(',')[0][1:])
    startY = int(start.split(',')[1][1:-1])

    goalX = int(goal.split(',')[0][1:])
    goalY = int(goal.split(',')[1][1:-1])
   
    dx = abs(startX - goalX)
    dy = abs(startY - goalY)

    return pow((dx+dy),1/2)

def neighbours(v):
    if v in graph:
        return graph[v]
    else:
        return None

def astar(graph, s, d):
    G = {}
    F = {}

    G[s] = 0
    F[s] = heuristic(s, d)
    nodes = []
    for node in graph:
        nodes.append(node)
        nodes += [x[0] for x in graph[node]]

    p = set(nodes)
    nodes = list(p)
    route = dict()
    pr = dict()
    for node in nodes:
        route[node] = float('inf')
        pr[node] = None

    route[s] = 0
    closedNodes = set()
    openNodes = set([start])
    prev = {}

    while p:
        current = None
        currentF = None

        u = min(p, key=route.get)
        p.remove(u)

        if d is not None and u == d:
            return route[d], pr

        if current == d:
            path = [current]
            while current in prev:
                current = prev[current]
                path.append(current)
            path.reverse()
            return path, F[d]

        closedNodes.add(current)

        for v, w in graph.get(u, ()):
            candG = route[u] + w
            if candG < route[v]:
                route[v] = candG
                G[v] = candG
                H = heuristic(v, d)
                F[v] = G[v] + H
                route[v] = F[v]

    return route, pr

graph = defaultdict(list)
seen_edges = defaultdict(int)
for src, dst, weight in edges:
    seen_edges[(src, dst, weight)] += 1
    if seen_edges[(src, dst, weight)] > 1:
        continue
    graph[src].append((dst, weight))
    graph[dst].append((src, weight))
        
if(alg == "D" or alg == "d"):
    d, prev = dijkstra(graph, str(start), str(dest))
    print(str(start) + " -> " + str(dest) + ": distance = {}".format(d))

elif(alg == "A" or alg == "a"):
    d, prev = astar(graph, str(start), str(dest))
    print(str(start) + " -> " + str(dest) + ": distance = {}".format(d))
else:
    print('Please enter either D or A')

#show graph
plt.xlim([-3, 103])
plt.ylim([-3, 103])
plt.axis("on") 
plt.gca().set_aspect('equal', adjustable='box')
plt.grid() 
plt.show() 

"""
    Author: Lasse Regin Nielsen
"""

from __future__ import print_function
import os, csv
import numpy as np
filepath = os.path.dirname(os.path.abspath(__file__))

def read_data(filename, has_header=True):
    """
        Read data from file.
        Will also return header if header=True
    """
    data, header = [], None
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        if has_header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

def load_graphs(filename):
    """
        Loads graphs from file
    """
    data, _ = read_data(filename, has_header=False)
    graphs = []
    for line in data:
        if line[0] == 't':
            G = Graph(id=int(line[2]))
            graphs.append(G)
        else:
            if line[0] == 'v':
                v = Vertex(id=int(line[1]), label=line[2])
                graphs[len(graphs)-1].add_vertex(vertex=v)
            elif line[0] == 'e':
                e = Edge(label=line[3],
                         from_vertex=graphs[len(graphs)-1].get_vertex(id=int(line[1])),
                         to_vertex=graphs[len(graphs)-1].get_vertex(id=int(line[2])))
                graphs[len(graphs)-1].add_edge(edge=e)
    return graphs

#################################################
#                    Classes                    #
#################################################
class Queue(object):
    """
        Implementation of a simple queue data structure
    """
    def __init__(self, queue=None):
        if queue is None:
            self.queue = []
        else:
            self.queue = list(queue)
    def dequeue(self):
        return self.queue.pop(0)
    def enqueue(self, element):
        self.queue.append(element)
    def is_empty(self):
        return len(self.queue) == 0
    def empty(self):
        self.queue = []

class Vertex():
    """
        Implementation of an Vertex in a graph
    """
    visited = False
    dfs_id = 0
    def __init__(self, id, label):
        self.id = id
        self.label = label

class Edge():
    """
        Implementation of an Edge in a graph
    """
    def __init__(self, label, from_vertex, to_vertex):
        self.label = label
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex

    def connected_to(self, vertex):
        return vertex.id == self.from_vertex.id or \
               vertex.id == self.to_vertex.id

class Graph():
    """
        Implementation of a Graph
    """
    edges, vertices = [], []
    def __init__(self, id):
        self.id = id
        self.edges = []
        self.vertices = []
    def add_vertex(self, vertex):
        self.vertices.append(vertex)
    def add_edge(self, edge):
        self.edges.append(edge)
    def get_vertex(self, id):
        for v in self.vertices:
            if v.id == id:
                return v
        raise KeyError('No vertex with the id was found in graph')
    def adjacent_edges(self, vertex):
        adj_edges = []
        for e in self.edges:
            if e.connected_to(vertex):
                adj_edges.append(e)
        return adj_edges
    def adjacent_vertices(self, vertex):
        adj_edges = self.adjacent_edges(vertex)
        adj_vertices = []
        for e in adj_edges:
            if e.from_vertex.id == vertex.id:
                adj_vertices.append(e.to_vertex)
            else:
                adj_vertices.append(e.from_vertex)
        return adj_vertices
    def adjacent_connections(self, vertex):
        adj_edges = self.adjacent_edges(vertex)
        adj_connections = []
        for e in adj_edges:
            if e.from_vertex.id == vertex.id:
                adj_connections.append((e, e.to_vertex))
            else:
                adj_connections.append((e, e.from_vertex))
        # Sort according to node index
        ids = [w.id for e,w in adj_connections]
        idx = np.argsort(ids)
        adj_connections = [adj_connections[i] for i in idx]
        return adj_connections
    def generate_vertices(self):
        for e in self.edges:
            for v in [e.from_vertex, e.to_vertex]:
                v.id = v.dfs_id
                if not v in self.vertices:
                    self.add_vertex(vertex=v)
    def get_max_vertex(self):
        ids = [v.id for v in self.vertices]
        idx = np.argsort(ids)[::-1]
        return self.vertices[idx[0]]
    def get_max_dfs_id_vertex(self):
        vertices_id = []
        for i, v in enumerate(self.vertices):
            if not v.dfs_id is None:
                vertices_id.append(i)
        if len(vertices_id) > 0:
            ids = [self.vertices[i].id for i in vertices_id]
            idx = np.argsort(ids)[::-1]
            return self.vertices[idx[0]]
        else:
            return []
    def get_min_vertex(self):
        ids = [v.id for v in self.vertices]
        idx = np.argsort(ids)
        return self.vertices[idx[0]]
    def contains_vertex_id(self, id):
        for v in self.vertices:
            if v.id == id:
                return True
        return False
    def contains_edge(self, from_id, to_id):
        for e in self.edges:
            if (e.from_vertex.id == from_id and e.to_vertex.id == to_id) or \
               (e.to_vertex.id == from_id and e.from_vertex.id == to_id):
               return True
        return False
    def reverse_graph(self):
        for e in self.edges:
            tmp_from = e.from_vertex
            e.from_vertex = e.to_vertex
            e.to_vertex = tmp_from
        self.edges = self.edges[::-1]
        self.vertices = self.vertices[::-1]
    def print_graph(self):
        DFScode = G2DFS(self)
        for line in DFScode:
            print(line)
    def get_edge(self, from_id, to_id):
        for e in self.edges:
            if (e.from_vertex.id == from_id and e.to_vertex.id == to_id) or \
               (e.to_vertex.id == from_id and e.from_vertex.id == to_id):
               return e
        return None
    def reset(self):
        for v in self.vertices:
            v.visited = False
            v.dfs_id = None

#################################################
#                   Functions                   #
#################################################
def DFS(G, v):
    """
        Depth-first search recursive algorithm:
        Input:
            G   Graph object containing vertices and edges
            v   Root vertex of the graph G (Vertex object)
        Output:
            p   Graph making a DFS spanning tree
    """
    G.reset() # Reset search parameters
    edges = []
    recursive_call_DFS(G, v, edges)
    p = Graph(-1)
    for e in edges:
        p.add_edge(e)
    p.generate_vertices()
    return p

def recursive_call_DFS(G, v, edges):
    """
        Helper function for recursive DFS
    """
    v.visited = True
    v.dfs_id = len(edges)
    neighbors = G.adjacent_connections(vertex=v)
    for e, w in G.adjacent_connections(vertex=v):
        if not w.visited:
            edges.append(e)
            recursive_call_DFS(G, w, edges)

def rightmost_path_BFS(G, v, v_target):
    """
        Get rightmost path using Breadth-First search algorithm on DFS path:
        Input:
            G           Graph object containing vertices and edges
            v           Root vertex of the graph G (Vertex object)
            v_target    Target vertex
        Output:
            p           Graph of shortest path from v to v_target
    """
    G.reset() # Reset search parameters
    for _v in G.vertices:
        _v.dfs_id = float('inf')
        _v.parent = None
    Q = Queue()
    v.dfs_id = 0
    Q.enqueue(v)
    while not Q.is_empty():
        current = Q.dequeue()
        for e, w in G.adjacent_connections(vertex=current):
            if w.dfs_id == float('inf'):
                w.dfs_id = current.dfs_id + 1
                w.parent = current
                Q.enqueue(w)
                if(w == v_target):
                    Q.empty()
                    break
    tmp = v_target
    p = Graph(id=-1)
    while tmp.parent is not None:
        e = Edge(label='_', from_vertex=tmp, to_vertex=tmp.parent)
        p.add_edge(edge=e)
        p.add_vertex(vertex=tmp)
        tmp = tmp.parent
    p.add_vertex(vertex=tmp)
    return p

def get_rightmost_path(G):
    """
        Returns the rightmost-path of the graph G
    """
    v_root = G.get_min_vertex()
    v_target = G.get_max_vertex()
    T_G = DFS(G=G, v=v_root)
    v_target = G.get_max_dfs_id_vertex()
    R = rightmost_path_BFS(T_G, v_root, v_target)
    for v in R.vertices:
        v.id = v.dfs_id
    R.reverse_graph()
    return R

def G2DFS(G):
    """
        Converts a graph object into a DFScode tuple sequence
    """
    DFScode = []
    for e in G.edges:
        DFScode.append((e.from_vertex.id, e.to_vertex.id,
            e.from_vertex.label, e.to_vertex.label, e.label))
    return DFScode

def DFS2G(C):
    """
        Converts a DFScode tuple sequence C into a graph G
    """
    G = Graph(id=-1)
    vertices = []
    for u,v,L_u,L_v,L_uv in C:
        for vertex, label in [(u, L_u), (v, L_v)]:
            if not (vertex, label) in vertices:
                vertices.append((vertex, label))
    for v_id, v_label in vertices:
        # Create and add vertex
        v = Vertex(id=v_id, label=v_label)
        G.add_vertex(vertex=v)
    # Add edges
    for t in C:
        # Expand tuple
        u, v, L_u, L_v, L_uv = t
        # Get vertices
        _u, _v = G.get_vertex(id=u), G.get_vertex(id=v)
        # Add edge
        e = Edge(label=L_uv, from_vertex=_u, to_vertex=_v)
        G.add_edge(edge=e)
    return G

def tuple_is_smaller(t1,t2):
    """
        Checks whether the tuple t1 is smaller than t2
    """
    t1_forward = t1[1] > t1[0]
    t2_forward = t2[1] > t2[0]
    i,j,x,y = t1[0], t1[1], t2[0], t2[1]
    # Edge comparison
    if t1_forward and t2_forward:
        if j < y or (j == y and i > x):
            return True
        elif j > y or (j == y and i < x):
            return False
    elif (not t1_forward) and (not t2_forward):
        if i < x or (i == x and j < y):
            return True
        elif i > x or (i == x and j > y):
            return False
    elif t1_forward and (not t2_forward):
        if j <= x:
            return True
        else:
            return False
    elif (not t1_forward) and t2_forward:
        if i < y:
            return True
        elif i > y: # Maybe something missing here
            return False
    # Lexicographic order comparison
    a1,b1,c1 = t1[2], t1[3], t1[4]
    a2,b2,c2 = t2[2], t2[3], t2[4]

    if not a1.isdigit():
        a1,b1,c1 = ord(a1),ord(b1),ord(c1)
        a2,b2,c2 = ord(a2),ord(b2),ord(c2)
    else:
        a1,b1,c1 = int(a1),int(b1),int(c1)
        a2,b2,c2 = int(a2),int(b2),int(c2)

    if a1 < a2:
        return True
    elif a1 == a2:
        if b1 < b2:
            return True
        elif b1 == b2:
            if c1 < c2:
                return True

    #if ord(t1[2]) < ord(t2[2]):
    #    return True
    #elif ord(t1[2]) == ord(t2[2]):
    #    if ord(t1[3]) < ord(t2[3]):
    #        return True
    #    elif ord(t1[3]) == ord(t2[3]):
    #        if ord(t1[4]) < ord(t2[4]):
    #            return True
    return False
        #raise KeyError('Wrong key type in tuple')

#def compare_DFScodes
def tuples_are_smaller(G1, G2):
    """
        Checks if tuples in G1 are less than tuples in G2
    """
    DFScodes_1, DFScodes_2 = G1, G2
    if len(DFScodes_1) != len(DFScodes_2):
        raise Exception('Size of the two graphs are not equal')
    for i in range(0, len(DFScodes_1)):
        t1, t2 = DFScodes_1[i], DFScodes_2[i]
        is_smaller = tuple_is_smaller(t1,t2)
        if is_smaller:
            return True
    return False

def get_minimum_DFS(G_list):
    """
        Finds the graph with smallest DFS code i.e. the canonical graph
    """
    # Initialize first one as minimum
    min_G = G_list[0]
    min_idx = 0
    counts = np.zeros(len(G_list))
    for i in range(0, len(G_list)):
        for j in range(0, len(G_list)):
            if i == j:
                continue
            is_smaller = tuples_are_smaller(G_list[i], G_list[j])
            if not is_smaller:
                counts[i] += 1
    min_idx = np.argmin(counts)
    min_G = G_list[min_idx]
    return min_G, min_idx

def subgraph_isomorphisms(C, G):
    """
        Returns the set of all isomorphisms between C and G
    """
    # Initialize set of isomorphisms by mapping vertex 0 in C
    # to each vertex x in G that shares the same label as 0s
    #G.print_graph()
    phi_c = []
    G_C = DFS2G(C)
    v0 = G_C.get_min_vertex()
    for v in G.vertices:
        if v.label == v0.label:
            phi_c.append([(v0.id, v.id)])
    for i, t in enumerate(C):
        u, v, L_u, L_v, L_uv = t    # Expand extended edge
        phi_c_prime = []            # partial isomorphisms
        for phi in phi_c:
            # phi is a list of transformations
            if v > u:
                # Forward edge
                try:
                    phi_u = transform_vertex(u, phi)
                except Exception as e:
                    continue
                # Find neighbors of transformed vertex
                vertex = G.get_vertex(phi_u)
                neighbors = G.adjacent_connections(vertex)
                for e, x in neighbors:
                    # Check if an inverse transformation exists
                    inv_trans_exists = check_inv_exists(x.id, phi)
                    if (not inv_trans_exists) and \
                        (x.label == L_v) and \
                        (e.label == L_uv):
                        phi_prime = list(phi)
                        phi_prime.append((v, x.id))
                        phi_c_prime.append(list(phi_prime))
            else:
                # Backward edge
                try:
                    phi_u = transform_vertex(u, phi)
                    phi_v = transform_vertex(v, phi)
                except Exception as e:
                    #print('abe2')
                    continue
                # Find neighbors of transformed vertex
                vertex = G.get_vertex(phi_u)
                neighbors = G.adjacent_connections(vertex)
                for e, x in neighbors:
                    if phi_v == x.id:
                        phi_c_prime.append(list(phi))
                        break
        phi_c = list(phi_c_prime)
    return phi_c

def check_inv_exists(v, phi):
    """
        Given a vertex id u and a set of partial isomorphisms phi.
        Returns True if an inverse transformation exists for v
    """
    for _phi in phi:
        if _phi[1] == v:
            return True
    return False

def inv_transform_vertex(x, phi):
    """
        Given a vertex id x and a set of partial isomorphisms phi.
        Returns the inverse transformed vertex id
    """
    for _phi in phi:
        if _phi[1] == x:
            return _phi[0]
    raise Exception('Could not find inverse transformation')

def transform_vertex(u, phi):
    """
        Given a vertex id u and a set of partial isomorphisms phi.
        Returns the transformed vertex id
    """
    for _phi in phi:
        if _phi[0] == u:
            return _phi[1]
    raise Exception('u couldn\' be found in the isomorphisms')

def RMPE(C, D):
    """
        Implements the RightMostPath-Extensions algorithm.
        Given a frequent canonical DFS code C and a list of graphs D, a
        set of possible rightmost path extensions from C, along with
        their support values are computed.
    """
    # Create graph of C -> G(C)
    G_C = DFS2G(C=C)
    # Only if C is not empty
    if len(C) > 0:
        # Compute rightmost path
        R = get_rightmost_path(G_C)
        u_r = R.vertices[len(R.vertices)-1].dfs_id
        L_u_r = R.vertices[len(R.vertices)-1].label
    E = [] # set of extensions from C
    for i, G in enumerate(D):
        if len(C) == 0: # If C is empty
            # add distinct label tuples in G_i as forward extensions
            for e in G.edges:
                L_x, L_y, L_xy = e.from_vertex.label, e.to_vertex.label, e.label
                f = (0, 1, L_x, L_y, L_xy)
                E.append((i, f))
                f = (0, 1, L_y, L_x, L_xy)
                E.append((i, f))
        else:
            # Get subgraph isomorphisms
            phi_c_i = subgraph_isomorphisms(C, G)
            for phi in phi_c_i:
                ############################################
                # Backward extensions from rightmost child #
                ############################################
                phi_u_r = transform_vertex(u_r, phi)
                # Find neighbors of transformed vertex
                vertex = G.get_vertex(phi_u_r)
                neighbors = G.adjacent_connections(vertex)
                for e, x in neighbors:
                    if check_inv_exists(x.id, phi):
                        v = inv_transform_vertex(x.id, phi)
                        if R.contains_vertex_id(id=v) and \
                           not G_C.contains_edge(from_id=u_r, to_id=v):
                            _e = G.get_edge(transform_vertex(v, phi), phi_u_r)
                            if _e is None:
                                raise Exception('Couldn\'t find edge')
                            L_v = G_C.get_vertex(id=v).label
                            b = (u_r, v, L_u_r, L_v, _e.label) # What label?
                            E.append((i, b))
                ###################################################
                # Forward extensions from nodes on rightmost path #
                ###################################################
                for u in R.vertices:
                    phi_u = transform_vertex(u.id, phi)
                    # Find neighbors of transformed vertex
                    vertex = G.get_vertex(phi_u)
                    neighbors = G.adjacent_connections(vertex)
                    for e, x in neighbors:
                        if not check_inv_exists(x.id, phi):
                            f = (u.id, u_r + 1, vertex.label, x.label, e.label)
                            E.append((i, f))
    # Only use distinct tuples
    E = list(set(E))
    extensions = list(set([e for i,e in E]))
    extensions = sort_tuples(extensions)
    pairs = []
    for ext in extensions:
        sup = 0
        for s in E:
            _, tup = s
            if tup == ext:
                sup += 1
        pairs.append((ext, sup))
    return pairs

def sort_tuples(E):
    """
        Sort a list of tuples using the get_minimum_DFS function.
    """
    sorted_tuples = []
    tuples = [[t] for t in E]
    for i in range(0, len(tuples)):
        min_G, min_idx = get_minimum_DFS(tuples)
        sorted_tuples.append(tuples[min_idx][0])
        del tuples[min_idx]
    return sorted_tuples

def compute_support(C, D):
    """
        Computes the support of subgraph C in set of graphs D
    """
    sup = 0
    for i, G in enumerate(D):
        phi_c_i = subgraph_isomorphisms(C, D[i])
        if len(phi_c_i) > 0:
            sup += 1
    return sup

def is_canonical(C):
    """
        Checks if C is canonical
    """
    D_C = [DFS2G(C)]    # graph corresponding to code C
    C_star = []         # initialize canonical DFScode
    k = len(C)
    for i in range(0, k):
        E = RMPE(C_star, D_C)
        if len(E) == 0:
            break
        G_list = [[_e[0]] for _e in E]
        min_G, min_idx = get_minimum_DFS(G_list)
        s_i = E[min_idx][0]
        sup_s_i = E[min_idx][1]
        if tuple_is_smaller(s_i, C[i]):
            return False
        C_star.extend([s_i])
    return True # no smaller code exists -> C is canonical

def g_span(C, D, min_sup, extensions):
    """
        Finds possible frequent and canonical extensions of C in D, using
        min_sup as lowest allowed support value.
        Results are stored in extensions
    """
    #extensions.append(C)
    E = RMPE(C, D)
    for t, sup_t in E:
        # extend the code with extended edge tuple t
        C_prime = list(C)
        C_prime.extend([t])
        # record the support of new extension
        sup_C_prime = sup_t
        # recursively call gSpan if code is frequent and canonical
        if (sup_C_prime >= min_sup) and is_canonical(C_prime):
            extensions.append(C)
            g_span(C_prime, D, min_sup, extensions)

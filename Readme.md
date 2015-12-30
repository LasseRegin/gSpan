# gSpan algorithm

Implementation of the depth-first gSpan algorithm for frequent graph mining in graphs data set.

Finds possible frequent and canonical extensions of a given graph from a given set of graphs.

## Supported python versions:
* Python 2.7
* Python 3.4

## Python package dependencies
* Numpy        (http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)

# Documentation

## Graph file format

Following shows an example of the format of a text file containing a set of graphs. Each line denodes a vertex (v) or edge (e) with a given label (end of line).

```
t # 1
v 10 a
v 20 b
v 30 a
v 40 b
e 10 20 _
e 10 30 _
e 20 30 _
e 30 40 _
t # 2
v 50 b
v 60 a
v 70 b
v 80 a
e 50 60 _
e 50 70 _
e 60 70 _
e 60 80 _
e 70 80 _
```

## Running the algorithm

Import algorithms
```python
from algorithms import g_span as gSpan
from algorithms import load_graphs
```

Load graphs from graph file and run gSpan algorithm
```python
graphs = load_graphs(filename)
n = len(graphs)
extensions = []
gSpan([], graphs, min_sup=min_sup, extensions=extensions)
for i, ext in enumerate(extensions):
    print('Pattern %d' % (i+1))
    for _c in ext:
        print(_c)
    print('')
```

## Example output

```
Pattern 1

Pattern 2
(0, 1, 'a', 'a', '_')

Pattern 3
(0, 1, 'a', 'a', '_')
(1, 2, 'a', 'b', '_')

Pattern 4
(0, 1, 'a', 'a', '_')
(1, 2, 'a', 'b', '_')
(2, 0, 'b', 'a', '_')

Pattern 5
(0, 1, 'a', 'a', '_')
(1, 2, 'a', 'b', '_')

Pattern 6
(0, 1, 'a', 'a', '_')
(1, 2, 'a', 'b', '_')

Pattern 7

Pattern 8
(0, 1, 'a', 'b', '_')

Pattern 9
(0, 1, 'a', 'b', '_')
(1, 2, 'b', 'a', '_')

Pattern 10
(0, 1, 'a', 'b', '_')

Pattern 11
(0, 1, 'a', 'b', '_')
(0, 2, 'a', 'b', '_')
```

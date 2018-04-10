from __future__ import print_function
import os, sys
import numpy as np
from algorithms import g_span as gSpan
from algorithms import load_graphs
filepath = os.path.dirname(os.path.abspath(__file__))

def main(filename='data/exampleG.txt', min_sup=2):
    filename = os.path.join(filepath, filename)
    graphs = load_graphs(filename)
    n = len(graphs)
    extensions = []
    gSpan([], graphs, min_sup=min_sup, extensions=extensions)
    for i, ext in enumerate(extensions):
        print('Pattern %d' % (i+1))
        for _c in ext:
            print(_c)
        print('')

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("")
        print("Finds possible frequent and canonical extensions of C in D, using")
        print("min_sup as lowest allowed support value.")
        print("Usage: %s FILENAME minsup" % (sys.argv[0]))
        print("")
        print("FILENAME: Relative path of graph data file.")
        print("minsup:   Minimum support value.")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['filename'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['min_sup'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            sys.exit("Not correct arguments provided. Use %s -h for more information" % (sys.argv[0]))
        main(**kwargs)

import os
import sys

thisdir, f = os.path.split(os.path.realpath(__file__))
basedir = os.path.abspath(os.path.join(thisdir, '..'))

if basedir not in sys.path:
    sys.path.insert(0, basedir)

import unittest
import random

import numpy as np

import _path
import spline

class TestSome(unittest.TestCase):

    def setUp(self):
        self.nrKnots = random.randint(6,30)
    
    def test_addsToOne(self):
        knots = np.hstack((np.zeros(3), np.linspace(0,1,self.nrKnots-4),
                           np.ones(3)))
        Ns = []
        for i in xrange(len(knots)-4):
            Ns.append(spline.basisFunction(i,knots))
        
        x = np.random.random(1000)
        y = np.zeros(len(x))
        for n in Ns:
            y += n(x)
            
        for val in y:
            self.assertAlmostEqual(val,1.)
        
    def test_matrix(self):
        knots = np.hstack((np.zeros(3), np.linspace(0,1,self.nrKnots-4),
                           np.ones(3)))
        xi=(knots[1:-3]+knots[2:-2]+knots[3:-1])/3
        nMatrix=np.zeros((len(xi),len(xi)))
        for i in xrange(len(xi)):
            fun = spline.basisFunction(i,knots)
            nMatrix[:,i]=fun(xi,3)
        revMatrix = nMatrix[::-1,::-1]
        for i in xrange(nMatrix.size):
            self.assertAlmostEqual(revMatrix.flat[i],nMatrix.flat[i])
            
if __name__ == '__main__':
    unittest.main()

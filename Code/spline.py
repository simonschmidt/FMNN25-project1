from scipy import *
from matplotlib import *
from pylab import *


class Spline(object):

    #TODO Interpolation stuff
    def __init__(self,ctrlP,knotP=None):
        '''
            Arguments:
                ctrlP: control points, (L x 2) matrix
                knotP: knot points, (L+4 x 1) matrix
                       default: equidistant on [0,1]
        '''
        self.ctrlP = ctrlP
        self.ncp = len(ctrlP)

        # Check and set knot points
        # Equidistant on [0,1] by default
        if knotP == None:
            knotP = linspace(0,1,self.ncp+4)

        # Convert lists/tuples
        if not isinstance(knotP,ndarray):
            knotP = array(knotP)

        # Verify shape
        # TODO be less restrictive about number of knot points
        if shape(knotP) != (self.ncp+4,):
            raise NotImplemented

        # Make sure it's sorted
        if not all(knotP[1:] >= knotP[:-1]):
            knotP = sort(knotP)

        self.knotP = knotP

    def __call__(self,x):
        '''
           de-Boor algorithm to evaluate BSpline
           x:     point to evaluate
        '''

        # k: s.t. u[k]<x<u[k+1]
        # p: degree (smoothness)
        # s: multiplicity of point (TODO)
        # cp[j,r]: result of de-boor algorithm at step j,r
        #          TODO: no need for matrix, change to vector or something
        k = searchsorted(self.knotP,x)-1
        if k<3 or k>self.ncp-1: # TODO find correct region
            raise Exception
        p = 3
        h = p
        s = 0
        cp = zeros((k-s+1 - (k-p),h+1,2))
        cp[:,0] = self.ctrlP[k-p:k-s+1]
        for r in xrange(1,h+1):
            j=r # Nobody likes indices
            for i in xrange(k-p+r,k-s+1):
                # TODO 0/0
                a = (x - self.knotP[i])/(self.knotP[i+p-r+1]-self.knotP[i])
                cp[j,r]=(1-a)*cp[j-1,r-1] + a*cp[j,r-1]
                j=j+1
        return cp[-1,-1]

    def plot(self,points=200,region=None):
        '''
            Plots the spline together with control points
            points: number of points to evaluate in region
            region: region to evaluate
                    default: full region
        '''
        if region==None:
            # TODO what is the maximum region?
            region=[self.knotP[3]+0.0001,self.knotP[-4]-0.001]
        x = linspace(region[0],region[1],200)
        y = array([self(xi) for xi in x])
        plot(y[:,0],y[:,1],'--')
        hold(True)
        plot(self.ctrlP[:,0],self.ctrlP[:,1])
        show()




# Some examples
def ex1():
    cp = array([ [0,0],[0,2],[2,3],[4,0],[6,3],[8,2],[8,0]])
    gp = array([0,0,0,0,1,1,1,2,2,2,2])
    s=Spline(cp,gp)
    s.plot()

def ex2(k=12):
    cp = randn(k,2)
    gp = linspace(0,1,k+4)
    s = Spline(cp,gp)
    s.plot()

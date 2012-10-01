from scipy import *
from matplotlib import *
from pylab import *
import bisect


def deBoor(ctrlP,u,x):
    '''
       de-Boor algorithm to evaluate BSpline
       ctrlP: (Lx2) control points
       u:     (L+4 x1) grid points
       x:     point to evaluate
    '''

    # k: s.t. u[k]<x<u[k+1]
    # p: degree (smoothness)
    # s: multiplicity of point (TODO)
    # cp[j,r]: result of de-boor algorithm at step j,r
    #          TODO: no need for matrix, change to vector or something
    k = bisect.bisect_left(u,x)-1
    if k<3: # TODO other cases
        raise Exception
    p = 3
    h = p
    s = 0
    cp = zeros((k-s+1 - (k-p),h+1,2))
    cp[:,0] = ctrlP[k-p:k-s+1]
    for r in xrange(1,h+1):
        j=r
        for i in xrange(k-p+r,k-s+1):
            a = (x - u[i])/(u[i+p-r+1]-u[i])
            #print (a,i,r,j)
            cp[j,r]=(1-a)*cp[j-1,r-1] + a*cp[j,r-1]
            j=j+1
    return cp[-1,-1]

#cp = array([ [cos(x),sin(x)] for x in linspace(0.51,2.13,6)])
#gp = array([0,0,0,0,0.25,0.5,0.75,1,1,1,1])


# Some examples
def ex1():
    cp = array([ [0,0],[0,2],[2,3],[4,0],[6,3],[8,2],[8,0]])
    gp = array([0,0,0,0,1,1,1,2,2,2,2])
    x=linspace(0.1,1.9,250)
    y=array([deBoor(cp,gp,i) for i in x])
    plot(y[:,0],y[:,1],'.--')
    hold(True)
    plot(cp[:,0],cp[:,1],'.-')
    show()

def ex2(k=12):
    cp = randn(k,2)
    gp = linspace(0,1,k+4)
    x=linspace(gp[3]+0.0001,gp[-4],200)
    y=array([deBoor(cp,gp,i) for i in x])
    plot(y[:,0],y[:,1],'--')
    hold(True)
    plot(cp[:,0],cp[:,1],'.-')
    show()

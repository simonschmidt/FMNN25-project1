import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl

class Spline(object):
    """
    
    A class that produces splines using de Boors algorithm and takes control
    points as input and can take knot point if the user want to specify them,
    otherwise they are set to be equidistant. 
    
    The class has three functions:  
        * __init__:   initialize the class and set some global variables
        * __call__:   evaluates the spline at a point x or a vector x
        * plot:       plots the curve with a given set of points. 
    
    """

    def __init__(self, ctrlPs, knots=None):
        """

        Arguments:   
            * ctrlP: a (L x n) matrix with control points that determines the 
                     curve and n is the dimension of the system.
            * knots: a (L+2) array, if left empty equidistant points will be 
                     taken instead with first 3 equal and same for the last 3.
                * default is set to empty
        
        Initialize a object of the class and sets the following variables:
            * da: a matrix with the denominators of alpha in the de Boor 
                  algorithm.
            * d0: an array to make vectorization of the __call__ method to work
        
        .. testcode::
            
            cp = array([ [0,0],[0,2],[2,3],[4,0],[6,3],[8,2],[8,0]])
            s=Spline(cp) 

        """

        self.cp = np.array(ctrlPs,dtype='float')
        if self.cp.size < 3:
            raise ValueError('There mest be at least 3 control points.')
        if self.cp.size == self.cp.shape[0]:
            self.cp = self.cp.reshape(-1,1)
        
        if knots != None:
            self.knots = np.array(knots,dtype='float')
            if len(ctrlPs) + 2 != len(knots):
                raise ValueError('knots is of the wrong size')

        else:
            self.knots = np.hstack((np.zeros(2),
                                    np.linspace(0,1,len(ctrlPs)-2),
                                    np.ones(2)))

            #self.knots = np.zeros((len(ctrlPs)+3))
            #self.knots[2:-2] = np.linspace(0,1,len(ctrlPs))[1:-2]
            #self.knots[0] = self.knots[2]
            #self.knots[1] = self.knots[2]
            #self.knots[-1] = self.knots[-3]
            #self.knots[-2] = self.knots[-3]
            #self.knots = np.linspace(0,1,len(ctrlPs)+2)

        self.da = self._calcDenomAlpha()
        self.d0 = np.array([-2,-1,0,1])

    def _calcDenomAlpha(self):
        """
        Private function that calculates the denominator in the alphas.
        """
        m = np.zeros((len(self.knots)-3,3))

        for i in xrange(3):
            m[:len(m)-i,i] = self.knots[3:len(self.knots)-i] - \
                             self.knots[i:len(m)]

        indx = (m != 0)
        m[indx] = 1./m[indx]

        return m

    def __call__(self,u):
        """
        Uses a vectorized version of the de Boor algorithm to calculate the 
        position of a specific u.       
        
        Arguments:
            * u: either a number or an array to be evaluated, must be in [0,1] 
                 or [knot[1], knot[-1]]
        
        Example
        -------
        .. testcode::
            
            >>> u = s(0.5)
            >>> v = s(linspace(0,1,5)[1:-2])
            >>> print u
            >>> print v
            
        .. testoutput::
            
            [[ 4.  1.]]
            
            [[ 0.33333333  1.83333333]
             [ 4.          1.        ]]
            
        """
        
        if not isinstance(u,np.ndarray):
            u = np.array([u])
            
        if (u < self.knots[2]).any() or (u >= self.knots[-3]).any():
            raise ValueError('u out of range!')

        I = np.searchsorted(self.knots[1:-3], u,side='right')
        leI = len(I)

        d = self.cp[np.repeat(I,4) + np.tile(self.d0,leI)]
        indx = np.ones(4,dtype='bool')
        indx[-1] = False

        for k in xrange(3):
            rpI = np.repeat(I,3-k)
            a = ((self.knots[rpI + np.tile(np.arange(1,4-k),leI)] - 
                np.repeat(u,3-k)) * 
                self.da[:,k][rpI + np.tile(self.d0[:-k-1],leI)]).reshape(-1,1)

            d = a * d[np.tile(indx[k:],leI)] + \
                (1 - a) * d[np.tile(np.roll(indx[k:],1),leI)]

        return d

    def plot(self,showCP=True,npoints=200):
        """
        A method to plot the spline by calling a range of points. Plotting is only 
        possible when the dimension of the control points is equal to 2. 
        Arguments:
            
            * showCP: boolean to control the plotting of control points
                * default is set to True
                
            * npoints: number of points to plot
                * default is set to 200

        """
        if not self.cp.shape[1] == 2:
            raise ValueError('Sorry but I only plot 2d')
        
        x = self(np.linspace(self.knots[2],self.knots[-3],npoints,endpoint=0))
        k = self(np.hstack((self.knots[2:-3],self.knots[-3]-1e-15)))
        plt.plot(x[:,0],x[:,1],color='red')
        plt.hold(True)
        plt.plot(k[:,0],k[:,1],'x',color='red')
        if showCP:
            plt.plot(self.cp[:,0],self.cp[:,1],'--',color='blue')
            plt.plot(self.cp[:,0],self.cp[:,1],'o',color='blue')
        xmin,xmax = self.cp[:,0].min(), self.cp[:,0].max()
        ymin,ymax = self.cp[:,1].min(), self.cp[:,1].max()
        xbrdr = (xmax - xmin)*0.05
        ybrdr = (ymax - ymin)*0.05
        plt.xlim(xmin - xbrdr, xmax + xbrdr)
        plt.ylim(ymin - ybrdr, ymax + ybrdr)
        plt.show()

def basisFunction(index, knotP):
    """
    Evaluates the basis function N for j given the knot points and returns
    a function
    Arguments:
        * index: index
        * knotP: knot points, (L+4 x 1) matrix
            * default: equidistant on [0,1]
    """

    def n(x,k=3,i=index):
        if k==0:
            return 1.*(n.knotP[i]<=x<n.knotP[i+1])
        den1 = (n.knotP[i+k] - n.knotP[i])
        den2 = (n.knotP[i+k+1] - n.knotP[i+1])
        if den1 != 0:
            den1 = 1./den1
        if den2 != 0:
            den2 = 1./den2
        return (x - n.knotP[i])*den1*n(x,k-1,i) + (1 - (x-n.knotP[i+1])*den2)*n(x,k-1,i+1)
    n.knotP = knotP

    return n
    
def interpolation(interP,knots=None):
    """
        Interpolates the given points and returns an object of the Spline class 
        Arguments:
            * interP: interpolation points, (L x 2) matrix
            * knotP: knot points, (L+4 x 1) matrix
                * default: equidistant on [0,1]
    """
    nip=len(interP)
    ctrlP=np.zeros((nip,2))
    if knots != None:
            knots = np.array(knots,dtype='float')
            if len(ctrlP) + 2 != len(knots):
                raise ValueError('knots is of the wrong size')

    else:
        knots = np.hstack((np.zeros(2),
                                    np.linspace(0,1,len(ctrlP)+2),
                                    np.ones(2)))
    xi=(knots[2:-4]+knots[3:-3]+knots[4:-2])/3
    nMatrix=np.zeros((len(xi),len(xi)))
    for i in xrange(len(xi)):
        fun=basisFunction(i,knots)
        for k in xrange(len(xi)):
            nMatrix[k,i]=fun(xi[k],3)
    print nMatrix

    ctrlP[:,0]=sl.solve(nMatrix,interP[:,0])
    ctrlP[:,1]=sl.solve(nMatrix,interP[:,1])
    
    print
    print ctrlP
    
    return Spline(ctrlP)

    
# Some examples
def ex1():
    cp = np.array([ [0,0],[0,2],[2,3],[4,0],[6,3],[8,2],[8,0]])
    s=Spline(cp)
    s.plot()

def ex2(k=12):
    cp = randn(k,2)
    s = Spline(cp)
    s.plot()
    
def ex3():
    cp = np.array([ [0,0],[0,2],[2,3],[4,0],[6,3],[8,2],[8,0]])
    s=interpolation(cp)
    s.plot()
    plot(cp[:,0],cp[:,1])
import warnings

import numpy as np
import scipy.linalg as sl
try:
    import matplotlib.pyplot as plt
    isplot2d = True
except ImportError:
    warnings.warn('matplotlib not found, 2D plotting not possible.')
    isplot2d = False
try:
    from enthought.mayavi import mlab as ml
    isplot3d = True
except ImportError:
    warnings.warn('Mayavi not found, 3D plotting not possible.')
    isplot3d = False


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
            * ctrlP: array_like (L x n) object with control points that
                     determines the curve in n dimensions and where L >= 3.
            * knots: optional array_like (L+2) object, if left empty
                     equidistant points will be taken instead with first 3
                     equal and same for the last 3. If left None, knot points
                     will be generated.
                * default is set to None
        
        Initialize a object of the class and sets the following variables:
            * knots: (L+2) numpy array instance of float64 type holding the
                     knots.
            * cp: (L x n) numpy array instance of float64 type holding the
                  control points.
            * da: a matrix with the inverse of the denominators of alpha in the
                  de Boor algorithm.
            * d0: an array to make vectorization of the __call__ method to work
        
        .. testcode::
            
            cp = array([ [0,0],[0,2],[2,3],[4,0],[6,3],[8,2],[8,0]])
            s=Spline(cp)

        """
        #If ctrlPs isn't array_like this will return an exception
        self.cp = np.array(ctrlPs,dtype='float')
        if self.cp.size < 3:
            raise ValueError('There must be at least 3 control points.')
        if self.cp.size == self.cp.shape[0]:
            self.cp = self.cp.reshape(-1,1)

        if knots != None:
            self.knots = np.array(knots,dtype='float')
            if (self.knots[1:] > self.knots[:-1]).any():
                raise ValueError('The knots should be in ascending order.')
            if len(ctrlPs) + 2 != len(knots):
                raise ValueError('knots is of the wrong size')

        else:
            self.knots = np.hstack((np.zeros(2),
                                    np.linspace(0,1,len(ctrlPs)-2),
                                    np.ones(2)))


        self.da = self._calcDenomAlpha()
        self.d0 = np.array([-2,-1,0,1])

    def _calcDenomAlpha(self):
        """
        Private function that calculates the inverse of the denominator in the
        alphas. The case of 0/0 = 0 is taken care of here.
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
        Calculates the de Boor algorithm in the following manner:
            - For every value in u, finds the index of the 'hot' interval I.
            - Finds the corresponding control points d_{I-2},...,d_{I+1}.
            - Calculates from the formula:
                
                d_{i}^{k} = a_{i}^{k-1} * d_{i}^{k-1} + 
                            (1 - a_{i}^{k-1}) * d_{i+1}^{k-1}
                          
              where
              
                                knot[i+3-k] - u 
                a_{i}^{j} = -------------------------
                             knot[i+3-k] - knot[i+k]

            - repeats for k = 0,1,2.

        Note that some of the indexing has been specially picked to fit memmory
        array indexing in the best possible way.
        
        Since the inverse of the denominator in alpha is already calculated,
        stored and the exception that 0/0 = 0 taken into consideration there
        is no need for a check here since it's  simply a multiplication and
        0*0 = 0 by defult.
        
        
        Arguments:
            * u: either a number or an array to be evaluated, must be inside
                 the interval [knot[2], knot[-2]).
        
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
        
        #If input is not array already it's converted for convenience.
        if not isinstance(u,np.ndarray):
            if isinstance(u,(int,float,long)):
                u = np.array([u],dtype='float')
            else:
                u = np.array(u,dtype='float')
            
        if (u < self.knots[2]).any() or (u >= self.knots[-3]).any():
            raise ValueError('u out of range!')

        #I is array with index of the 'hot' interval for every value past in.
        I = np.searchsorted(self.knots[1:-2], u,side='right')
        leI = len(I)

        #For every u_i we have that d[i*4:(i+1)*4] = d_{I[i]-2},...,d_{I[i]+1}
        d = self.cp[np.repeat(I,4) + np.tile(self.d0,leI)]

        #Used for indexing: take 3, skip 1, repeat
        indx = np.ones(4,dtype='bool')
        indx[-1] = False

        for k in xrange(3):
            rpI = I.repeat(3-k)

            a = ((self.knots[rpI + np.tile(np.arange(1,4-k),leI)] - 
                u.repeat(3-k)) * 
                self.da[:,k][rpI + np.tile(self.d0[:-k-1],leI)]).reshape(-1,1)

            d = a * d[np.tile(indx[k:],leI)] + \
                (1 - a) * d[np.tile(np.roll(indx[k:],1),leI)]

        return d

    def plot(self, axes=None, showCP=True, npoints=200):
        if axes == None:
            if self.cp.shape[1] == 2:
                self._plot2d((0,1),showCP,npoints)
            elif self.cp.shape[1] == 3:
                self._plot3d((0,1,2),showCP,npoints)
            elif self.cp.shape[1] == 1:
                pass
            else:
                raise ValueError("Can't plot %i dimensions, specify what axes"
                                 " should be used." % self.cp.shape[1])
        else:
            if len(axes) == 2:
                self._plot2d(axes,showCP,npoints)
            elif len(axes) == 3:
                self._plot3d(axes,showCP,npoints)
            elif len(axes) == 1:
                self._plot1d(axes,showCP,npoints)
            else:
                raise ValueError("Can't plot %i dimensions, number of axes"
                                 " must be 1, 2 or 3." % len(axes))

    def _plot1d(self, a, showCP, npoints):
    
        if not isplot2d:
            raise NotImplementedError('Lacking matplotlib to do the plotting.')
            
        x = np.linspace(self.knots[2],self.knots[-3],npoints,endpoint=0)
        k = np.hstack((self.knots[2:-3],self.knots[-3]-1e-15))
        plt.plot(x,self(x)[:,a[0]], c='blue')
        plt.hold(True)
        plt.plot(k,self(k)[:,a[0]],'*', c='blue')
        
        if showCP:
            plt.plot(np.ones(len(self.cp))*x[0], self.cp[:,a[0]],
                     'o', color='blue')
        plt.show()

    def _plot2d(self, a, showCP, npoints):
        """
        A method to plot the spline by calling a range of points. Plotting is only 
        possible when the dimension of the control points is equal to 2. 
        Arguments:
            
            * showCP: boolean to control the plotting of control points
                * default is set to True
                
            * npoints: number of points to plot
                * default is set to 200

        """
        if not isplot2d:
            raise NotImplementedError('Lacking matplotlib to do the plotting.')
        
        x = self(np.linspace(self.knots[2],self.knots[-3],npoints,endpoint=0))
        k = self(np.hstack((self.knots[2:-3],self.knots[-3]-1e-15)))
        plt.plot(x[:,a[0]],x[:,a[1]],color='red')
        plt.hold(True)
        plt.plot(k[:,a[0]],k[:,a[1]],'*',color='red')
        if showCP:
            plt.plot(self.cp[:,a[0]],self.cp[:,a[1]],'--',color='blue')
            plt.plot(self.cp[:,a[0]],self.cp[:,a[1]],'o',color='blue')
        xmin,xmax = self.cp[:,a[0]].min(), self.cp[:,a[0]].max()
        ymin,ymax = self.cp[:,a[1]].min(), self.cp[:,a[1]].max()
        xbrdr = (xmax - xmin)*0.05
        ybrdr = (ymax - ymin)*0.05
        plt.xlim(xmin - xbrdr, xmax + xbrdr)
        plt.ylim(ymin - ybrdr, ymax + ybrdr)
        plt.show()
        
    def _plot3d(self, a, showCP, npoints):
        
        
        if not isplot3d:
            raise NotImplementedError('Lacking Mayavi to do the plotting.')
            
        x = self(np.linspace(self.knots[2],self.knots[-3],npoints,endpoint=0))
        k = self(np.hstack((self.knots[2:-3],self.knots[-3]-1e-15)))
        ml.plot3d(x[:,a[0]],x[:,a[1]],x[:,a[2]],color=(.5,.2,.3))
        ml.points3d(k[:,a[0]],k[:,a[1]],k[:,a[2]],
                    color=(.5,.2,.3),scale_factor=.3)
        
        if showCP:
            ml.plot3d(self.cp[:,a[0]], self.cp[:,a[1]],
                      self.cp[:,a[2]], color=(.2,.4,.4))
            ml.points3d(self.cp[:,a[0]], self.cp[:,a[1]],
                        self.cp[:,a[2]], color=(.2,.4,.4), scale_factor=.3)
        ml.show()
        
    def getInterval(self):
        return self.knots[2],self.knots[-3]

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
            return 1.*((n.knotP[i] <= x) * (x < n.knotP[i+1]) +
                       (n.knotP[i+1] == n.knotP[-1]) * (x == n.knotP[-1]))
        den1 = (n.knotP[i+k] - n.knotP[i])
        den2 = (n.knotP[i+k+1] - n.knotP[i+1])
        if den1 != 0:
            den1 = 1./den1
        if den2 != 0:
            den2 = 1./den2
        return (x - n.knotP[i])*den1*n(x,k-1,i) + \
               (n.knotP[i+k+1]-x)*den2*n(x,k-1,i+1)
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
                raise ValueError('Knots is of the wrong size')

    else:
        knots = np.hstack((np.zeros(3), np.linspace(0,1,len(ctrlP)-2),
                           np.ones(3)))
    xi=(knots[1:-3]+knots[2:-2]+knots[3:-1])/3
    nMatrix=np.zeros((len(xi),len(xi)))
    for i in xrange(len(xi)):
        fun=basisFunction(i,knots)
        nMatrix[:,i]=fun(xi,3)
    #nMatrix[-1,-1] = 1.
    print nMatrix

    ctrlP[:,0]=sl.solve(nMatrix,interP[:,0])
    ctrlP[:,1]=sl.solve(nMatrix,interP[:,1])
    
    print
    print ctrlP
    
    return Spline(ctrlP)

def getN(k, knots=None):
    """
    Uses the Spline class to calculate the k:th basis function and returns it
    as a one variable function on the intervall (knot[1],knot[-1]).
    Arguments:
        * k: which basis function is wanted. Must be an integer in the
             intervall [0,len(knots)-1]
        * knots: an array of knot points for the basis function.
            * default: 34 equidistant points in [0,1]
    """
    if k % 1:
        raise ValueError("expected k to be integer")
    if knots == None:
        ncp = 30
    else:
        ncp = len(knots) - 2
    ctrlP = np.zeros((ncp,2))
    ctrlP[k,:] = np.array([1,1])
    return Spline(ctrlP, knots)


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

def ex4():
    n=getN(10)
    plt.plot(linspace(0,1,100), n(linspace(0.2,0.7,100)))
    

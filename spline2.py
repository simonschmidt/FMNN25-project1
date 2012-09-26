import numpy as np
import matplotlib.pyplot as plt


class Spline(object):

    def __init__(self, ctrlPs, knots=None):
        """

        """

        self.cp = np.array(ctrlPs,dtype='float')

        if knots != None:
            self.knots = np.array(knots,dtype='float')

        else:
            self.knots = np.linspace(0,1,len(ctrlPs) + 2)

        self.da = self._calcDenomAlpha()
        self.d0 = np.array([-2,-1,0,1])

    def _calcDenomAlpha(self):
        m = np.zeros((len(self.knots)-3,3))

        for i in xrange(3):
            m[:len(m)-i,i] = self.knots[3:len(self.knots)-i] - \
                             self.knots[i:len(m)]

        indx = (m != 0)
        m[indx] = 1./m[indx]

        return m

    def __call__(self,u):

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
                self.da[:,k][rpI + np.tile(self.d0[:-(k+1)],leI)]).reshape(-1,1)

            d = a * d[np.tile(indx[k:],leI)] + \
                (1 - a) * d[np.tile(np.roll(indx[k:],1),leI)]

        return d

    def plot(self,showCP=True,npoints=200):
        x = self(np.linspace(self.knots[2],self.knots[-3],npoints,endpoint=False))
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

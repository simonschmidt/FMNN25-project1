import numpy as np


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
            m[:len(m)-i,i] = self.knots[3:len(self.knots)-i] - self.knots[i:len(m)]
        
        indx = (m != 0)
        m[indx] = 1./m[indx]
        
        return m
        
    def __call__(self,u):
        
        if not isinstance(u,np.ndarray):
            u = np.array([u])
        
        I = np.searchsorted(self.knots[1:-1], u,side='left')
        
        if (I < 2).any() or (I > (len(self.cp) -1)).any():
            raise ValueError('u out of range!')
        
        d = self.cp[np.repeat(I,4) + np.tile(self.d0,len(I))]
        indx = np.ones(4,dtype='bool')
        indx[-1] = False
        
        for k in xrange(3):
            a = (self.knots[np.repeat(I,3-k) + np.tile(np.arange(1,4-k),len(I))] - \
                np.repeat(u,3-k)) * \
                self.da[:,k][np.repeat(I,3-k) + np.tile(self.d0[:-(k+1)],len(I))]
            
            
            d = (a).reshape(-1,1) * d[np.tile(indx[k:],len(I))] + \
                (1 - a).reshape(-1,1) * d[np.tile(np.roll(indx[k:],1),len(I))]
            
        return d
            
            
            

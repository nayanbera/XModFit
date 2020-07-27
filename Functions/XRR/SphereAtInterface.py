####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines/'))
from functools import lru_cache
####Please do not remove lines above####

####Import your modules below if needed####
from xr_ref import parratt


class SphereAtInterface: #Please put the class name same as the function name
    def __init__(self,x=0.1,lam=1.0,Rc=10,Rsig=0.0,rhoc=4.68,D=60.0, cov=100,Zo=20.0,decay=3.0,rho_up=0.333,
                 rho_down=0.38,zmin=-50,zmax=100,dz=1,roughness=3.0,rrf=True,mpar={},qoff=0):
        """
        Calculates X-ray reflectivity from a system of nanoparticle at an interface between two media
        x         	: array of wave-vector transfer along z-direction
        lam       	: wavelength of x-rays in invers units of x
        Rc        	: Radius of nanoparticles in inverse units of x
        rhoc      	: Electron density of the nanoparticles
        cov       	: Coverate of the nanoparticles in %
        D         	: The lattice constant of the two dimensional hcp structure formed by the particles
        Zo        	: Average distance between the center of the nanoparticles and the interface
        decay     	: Assuming exponential decay of the distribution of nanoparticles away from the interface
        rho_up    	: Electron density of the upper medium
        rho_down	: Electron density of the lower medium
        zmin      	: Minimum z value for the electron density profile
        zmax      	: Maximum z value for the electron density profile
        dz       	: minimum slab thickness
        roughness	: Roughness of the interface
        rrf      	: True for Frensnel normalized refelctivity and False for just reflectivity
        qoff      	: offset in the value of qz due to alignment errors
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.Rc=Rc
        self.lam=lam
        self.rhoc=rhoc
        self.Zo=Zo
        self.cov=cov
        self.D=D
        self.decay=decay
        self.rho_up=rho_up
        self.rho_down=rho_down
        self.zmin=zmin
        self.zmax=zmax
        self.dz=dz
        self.roughness=roughness
        self.rrf=rrf
        self.qoff=qoff
        self.__mpar__ = mpar
        self.choices={'rrf':[True,False]}
        self.__fit__=False
        self.init_params()



    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value=0,vary=0)
        """
        self.params=Parameters()
        self.params.add('Rc',value=self.Rc,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('rhoc',value=self.rhoc,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('Zo',value=self.Zo,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('D',value=self.D,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('cov',value=self.cov,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('decay',value=self.decay,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('roughness',value=self.roughness,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('qoff',value=self.qoff,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)


    @lru_cache(maxsize=10)
    def decayNp(self,z,Rc=10,D=30.0,z0=0.0,xi=1.0,cov=100.0,rhoc=4.65,rhos=(0.334,0.38),sig=1.0):
        z=np.array(z)
        if sig<1e-3:
            z2=z
        else:
            zmin=z[0]-5*sig
            zmax=z[-1]+5*sig
            z2=np.arange(zmin,zmax,self.dz)
        intf=np.where(z2<=0,rhos[0],rhos[1])
        if z0<=0:
            z1=np.linspace(-5*xi+z0,z0,101)
            dec=np.exp((z1-z0)/xi)/xi
        else:
            z1=np.linspace(z0,z0+5*xi,101)
            dec=np.exp((z0-z1)/xi)/xi
        rhoz=np.zeros_like(z2)
        for i in range(len(z1)):
            rhoz=rhoz+self.rhoNPz(tuple(z2),z0=z1[i],rhoc=rhoc,Rc=Rc,D=D,rhos=rhos)*dec[i]/sum(dec)
        rhoz=cov*rhoz/100.0+(100-cov)*intf/100.0
        x=np.arange(-5*sig,5*sig,self.dz)
        if sig>1e-3:
            rough=np.exp(-x**2/2.0/sig**2)/np.sqrt(2*np.pi)/sig
            res=np.convolve(rhoz,rough,mode='valid')*self.dz
            if len(res)>len(z):
                return res[0:len(z)]
            else:
                return res
        else:
            return rhoz

    @lru_cache(maxsize=10)
    def rhoNPz(self,z,z0=0,rhoc=4.65,Rc=10.0,D=28.0,rhos=(0.334,0.38)):
        z=np.array(z)
        rhob=np.where(z>0,rhos[1],rhos[0])
        #D=D/2
        return np.where(np.abs(z-z0)<=Rc,(2*np.pi*(rhoc-rhob)*(Rc**2-(z-z0)**2)+1.732*rhob*D**2)/(1.732*D**2),rhob)


    def y(self):
        """
        Define the function in terms of x to return some value
        """
        self.output_params = {'scaler_parameters': {}}
        rhos=(self.rho_up,self.rho_down)
        lam=self.lam
        z=np.arange(self.zmin,self.zmax,self.dz)
        d=np.ones_like(z)*self.dz
        edp=self.decayNp(tuple(z),Rc=self.Rc,z0=self.Zo,xi=self.decay,cov=self.cov,rhos=rhos,rhoc=self.rhoc,sig=self.roughness,D=self.D)
        if not self.__fit__:
            self.output_params['EDP']={'x':z,'y':edp}
        beta=np.zeros_like(z)
        rho=np.array(edp,dtype='float')
        refq,r2=parratt(self.x+self.qoff,lam,d,rho,beta)
        if self.rrf:
            ref,r2=parratt(self.x,lam,[0.0,1.0],rhos,[0.0,0.0])
            refq=refq/ref
        return refq


if __name__=='__main__':
    x=np.linspace(0.001,1.0,100)
    fun=SphereAtInterface(x=x)
    print(fun.y())

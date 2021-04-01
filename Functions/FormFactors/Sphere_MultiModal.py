import numpy as np
from lmfit import Parameters
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
from utils import find_minmax
from PeakFunctions import Gaussian, LogNormal

class Sphere_MultiModal:
    def __init__(self, x=0.001, dist='Gaussian', N=50, rhoc=1.0, rhosol=0.0, norm=1.0, bkg=0.0,
                 mpar={'Distributions':{'Dist':['Dist1'],'R':[1.0],'Rsig':[1.0],'Norm':[1.0]}}):
        """
        Calculates the form factor of a solid sphere with Multimodal size distribution
        x     	: Array of q-values in the same reciprocal unit as R and Rsig
        R     	: Mean radius of the solid spheres
        Rsig  	: Width of the distribution of solid spheres
        dist  	: Gaussian or LogNormal
        N     	: No. of points on which the distribution will be calculated
        integ   : The type of integration ('Normal' or 'MonteCarlo') Default: 'Normal'
        rhoc  	: Electron density of the particle
        rhosol	: Electron density of the solvent or surrounding environment
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.dist=dist
        self.rhoc=rhoc
        self.rhosol=rhosol
        self.norm=norm
        self.bkg=bkg
        self.N=N
        self.__mpar__=mpar
        self.__mkeys__=list(self.__mpar__.keys())
        self.choices={'dist':['Gaussian','LogNormal']}
        self.init_params()
        self.output_params = {'scaler_parameters': {}}

    def init_params(self):
        self.params=Parameters()
        self.params.add('rhoc',value=self.rhoc,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('rhosol',value=self.rhosol,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('norm',value=self.norm,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('bkg',value=self.bkg,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key !='Dist':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)

    def update_params(self):
        mkey = self.__mkeys__[0]
        key='R'
        self.__Nl__ = len(self.__mpar__[mkey][key])
        self.__R__ = [self.params['__%s_%s_%03d' % (mkey, key, i)] for i in range(self.__Nl__)]
        key='Rsig'
        self.__Rsig__ = [self.params['__%s_%s_%03d' % (mkey, key, i)] for i in range(self.__Nl__)]
        key='Norm'
        self.__Norm__ = [self.params['__%s_%s_%03d' % (mkey, key, i)] for i in range(self.__Nl__)]

    def y(self):
        self.update_params()
        rho=self.rhoc-self.rhosol
        if self.dist=='Gaussian':
            rmin, rmax = max(0.0001, self.__R__[0]-5*self.__Rsig__[0]),self.__R__[-1]+5*self.__Rsig__[-1]
            r=np.linspace(rmin,rmax,self.N)
        else:
            rmin, rmax = max(0.0001, np.exp(np.log(self.__R__[0]) - 5 * self.__Rsig__[0])), np.exp(np.log(self.__R__[-1]) + 5 * self.__Rsig__[-1])
            r = np.logspace(np.log10(rmin), np.log10(rmax), self.N)
        dist=np.zeros_like(r)
        for i in range(self.__Nl__):
            if self.dist == 'Gaussian':
                gau=Gaussian.Gaussian(x = r, pos = self.__R__[i], wid = self.__Rsig__[i])
                gau.x=r
                tdist=self.__Norm__[i]*gau.y()
                self.output_params[self.__mpar__['Distributions']['Dist'][i]] = {'x':r,'y':tdist/np.sum(tdist)}
                dist = dist + tdist
            else:
                lgn=LogNormal.LogNormal(x = r, pos = self.__R__[i], wid = self.__Rsig__[i])
                lgn.x = r
                tdist = self.__Norm__[i]*lgn.y()
                self.output_params[self.__mpar__['Distributions']['Dist'][i]] = {'x':r,'y':tdist/np.sum(tdist)}
                dist = dist + tdist
        sumdist = np.sum(dist)
        self.output_params['Distribtuion']={'x':r,'y':dist/sumdist}
        mean = np.sum(r*dist)/sumdist
        self.output_params['scaler_parameters']['Rmean'] = mean
        self.output_params['scaler_parameters']['Rwidth'] = np.sqrt(np.sum((r-mean)**2*dist)/sumdist)
        ffactor=[]
        for x in self.x:
            f = np.sum(16*np.pi**2*(np.sin(x * r) - x * r * np.cos(x * r)) ** 2 * dist / x ** 6)
            ffactor.append(f / sumdist)
        return self.norm*rho**2*np.array(ffactor)+self.bkg


if __name__=='__main__':
    x=np.linspace(0.001,1.0,500)
    fun=Sphere_MultiModal(x=x)
    print(fun.y())

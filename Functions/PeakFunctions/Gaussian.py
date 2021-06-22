import numpy as np
from lmfit import Parameters

class Gaussian:
    def __init__(self,x=0.0,pos=0.5,wid=0.1,norm=1.0,bkg=0.0,mpar={}):
        """
        Provides Gaussian function

        x    	: Scalar or array of values
        pos  	: Peak position
        wid  	: Width
        norm 	: Normalization constant
        bkg  	: Constant background
        """
        self.x=x
        self.pos=pos
        self.wid=wid
        self.norm=norm
        self.bkg=0.0
        self.__mpar__=mpar
        self.choices={}
        self.output_params={'scaler_parameters':{}}
        self.init_params()

    def gau(self,x,pos,wid,norm):
        """
        Gaussian function
        """
        return norm*np.exp(-(x-pos)**2/wid**2)/2.5066/wid

    def init_params(self):
        self.params=Parameters()
        self.params.add('pos',value=self.pos,vary=1,min=-np.inf,max=np.inf,expr=None,brute_step=max(0.1*self.pos,0.1))
        self.params.add('wid',value=self.wid,vary=1,min=-np.inf,max=np.inf,expr=None,brute_step=max(0.1*self.wid,0.1))
        self.params.add('norm',value=self.norm,vary=1,min=-np.inf,max=np.inf,expr=None,brute_step=max(0.1*self.norm,0.1))
        self.params.add('bkg',value=self.bkg,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=max(0.1*self.bkg,0.1))


    def y(self):
        self.output_params = {'scaler_parameters': {}}
        self.output_params['scaler_parameters']['peak_position']=self.pos
        self.output_params['scaler_parameters']['peak_width']=2.35428*self.wid
        return self.gau(self.x,self.pos,self.wid,self.norm)+self.bkg

if __name__=='__main__':
    x=np.arange(0.001,1.0,0.01)
    fun=Gaussian(x=x)
    print(fun.y())

####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_rountines'))
from functools import lru_cache
####Please do not remove lines above####

####Import your modules below if needed####
from FormFactors.Sphere import Sphere
from ff_sphere import ff_sphere_ml
from PeakFunctions import LogNormal, Gaussian
from utils import find_minmax, calc_rho


class Sphere_Uniform_Edep: #Please put the class name same as the function name
    def __init__(self, x=0, Np=10, flux=1e13, bkg=0.0,dist='Gaussian', relement='Au', Energy=None, NrDep='True', norm=1.0,
                 D=1.0, phi=0.1, U=-1.0, SF='None',Rsig=0.0,
                 mpar={'Multilayers':{'Material':['Au','H2O'],'Density':[19.32,1.0],'SolDensity':[1.0,1.0],'Rmoles':[1.0,0.0],'R':[1.0,0.0]}}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered nanoparticles with different materials

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Np          : No. of points with which the size distribution will be computed. Default: 10
        Energy      : Energy of the X-rays
        NrDep       : Energy dependence of the non-resonant element. Default= 'True' (Energy Dependent), 'False' (Energy independent)
        dist        : The probablity distribution fucntion for the radii of different interfaces in the nanoparticles. Default: Gaussian
        norm        : The density of the nanoparticles in Molar (Moles/Liter)
        flux        : Total X-ray flux to calculate the errorbar to simulate the errorbar for the fitted data
        Rsig        : Widths of the distributions ('Rsig' in Angs) of radii of all the interfaces present in the nanoparticle system.
        bkg         : In-coherrent scattering background
        D           : Hard Sphere Diameter
        phi         : Volume fraction of particles
        U           : The sticky-sphere interaction energy
        SF          : Type of structure factor. Default: 'None'
        mpar        : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Radii ('R' in Angs), and
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.D=D
        self.phi=phi
        self.U=U
        self.SF=SF
        self.norm=norm
        self.dist=dist
        self.Np=Np
        self.relement=relement
        self.NrDep=NrDep
        self.Energy=Energy
        #self.rhosol=rhosol
        self.flux=flux
        self.Rsig=Rsig
        self.bkg=bkg
        self.__mpar__=mpar #If there is any multivalued parameter
        self.choices={'dist':['Gaussian','LogNormal'],'NrDep':['True','False']} #If there are choices available for any fixed parameters
        self.__fit__=False
        self.output_params={'scaler_parameters':{}}
        self.__mkeys__=list(self.__mpar__.keys())
        self.init_params()

    def init_params(self):
        """
        Define all the fitting parameters like
        self.params.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params=Parameters()
        self.params.add('norm',value=self.norm,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('Rsig',value=self.Rsig,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('D',value=self.D,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('U',value=self.U,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('phi',value=self.phi,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key!='Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)


    @lru_cache(maxsize=10)
    def calc_Rdist(self, R, Rsig, dist, N):
        R = np.array(R)
        totalR = np.sum(R[:-1])
        if Rsig > 0.001:
            fdist = eval(dist + '.' + dist + '(x=0.001, pos=totalR, wid=Rsig)')
            rmin, rmax = find_minmax(fdist, totalR, Rsig)
            dr = np.linspace(rmin, rmax, N)
            fdist.x = dr
            rdist = fdist.y()
            sumdist = np.sum(rdist)
            rdist = rdist / sumdist
            self.output_params['Distribution'] = {'x': dr, 'y': rdist}
            return dr, rdist, totalR
        else:
            self.output_params['Distribution'] = {'x': [totalR], 'y': [1.0]}
            return [totalR], [1.0], totalR

    @lru_cache(maxsize=10)
    def new_sphere(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10):
        q = np.array(q)
        dr, rdist, totalR = self.calc_Rdist(R, Rsig, dist, Np)
        form = np.zeros_like(q)
        eiform = np.zeros_like(q)
        aform = np.zeros_like(q)
        cform = np.zeros_like(q)
        pfac = (4 * np.pi * 2.818e-5 * 1.0e-8) ** 2
        for i in range(len(dr)):
            r = np.array(R) * (1 + (dr[i] - totalR) / totalR)
            ff, mff = ff_sphere_ml(q, r, rho)
            form = form + rdist[i] * ff
            eiff, meiff = ff_sphere_ml(q, r, eirho)
            eiform = eiform + rdist[i] * eiff
            aff, maff = ff_sphere_ml(q, r, adensity)
            aform = aform + rdist[i] * aff
            cform = cform + rdist[i] * meiff * maff
        return pfac * form, pfac * eiform, pfac * aform, np.abs(pfac * cform)  # in cm^2

    @lru_cache(maxsize=2)
    def new_sphere_dict(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, key='SAXS-term'):
        form, eiform, aform, cform = self.new_sphere(q, R, Rsig, rho, eirho, adensity, dist=dist, Np=Np)
        if key == 'SAXS-term':
            return eiform
        elif key == 'Resonant-term':
            return aform
        elif key == 'Cross-term':
            return cform
        elif key == 'Total':
            return form


    def update_params(self):
        mkey=self.__mkeys__[0]
        key='Density'
        Nmpar=len(self.__mpar__[mkey][key])
        self.__density__=tuple([self.params['__%s_%s_%03d'%(mkey,key,i)].value for i in range(Nmpar)])
        key='SolDensity'
        self.__solDensity__=tuple([self.params['__%s_%s_%03d'%(mkey,key,i)].value for i in range(Nmpar)])
        key='Rmoles'
        self.__Rmoles__=tuple([self.params['__%s_%s_%03d'%(mkey,key,i)].value for i in range(Nmpar)])
        key='R'
        self.__R__=tuple([self.params['__%s_%s_%03d'%(mkey,key,i)].value for i in range(Nmpar)])
        key='Material'
        self.__material__=tuple([self.__mpar__[mkey][key][i] for i in range(Nmpar)])


    def y(self):
        """
        Define the function in terms of x to return some value
        """
        self.update_params()
        if type(self.x) == dict:
            sqf={}
            for key in self.x.keys():
                sq=[]
                Energy=float(key.split('_')[1].split(':')[1])
                rho,eirho,adensity,rhor,eirhor,adensityr=calc_rho(R=self.__R__,material=self.__material__,
                                                                       density=self.__density__, sol_density=self.__solDensity__,
                                                                       Energy=Energy, Rmoles= self.__Rmoles__, NrDep=self.NrDep)
                sqf[key] = self.norm * 6.022e20 * self.new_sphere_dict(tuple(self.x[key]), tuple(self.__R__),
                                                                       self.Rsig, tuple(rho), tuple(eirho),
                                                                       tuple(adensity), key=key, dist=self.dist,
                                                                       Np=self.Np)  # in cm^-1
            if self.SF is None:
                struct = np.ones_like(self.x[key])  # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x[key], D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x[key], D=self.D, phi=self.phi, U=self.U, delta=0.01)

            if not self.__fit__:
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1]}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1]}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]}

        else:
            rho, eirho, adensity, rhor, eirhor, adensityr = calc_rho(R=self.__R__,material=self.__material__,
                                                                       density=self.__density__, sol_density=self.__solDensity__,
                                                                       Energy=self.Energy, Rmoles= self.__Rmoles__, NrDep=self.NrDep)
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)

            tsqf, eisqf, asqf, csqf = self.new_sphere(tuple(self.x), tuple(self.__R__), self.Rsig, tuple(rho),
                                                      tuple(eirho), tuple(adensity), dist=self.dist, Np=self.Np)
            sqf = self.norm * np.array(tsqf) * 6.022e20 * struct  # in cm^-1
        return sqf



if __name__=='__main__':
    x=np.arange(0.001,1.0,0.1)
    fun=Sphere_Uniform_Edep(x=x)
    print(fun.y())

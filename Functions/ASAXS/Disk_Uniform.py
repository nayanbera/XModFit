####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
from scipy.special import j1
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines'))
from functools import lru_cache
####Please do not remove lines above####

####Import your modules below if needed####
from Chemical_Formula import Chemical_Formula
from PeakFunctions import LogNormal, Gaussian
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf
#from ff_cylinder import ff_cylinder_ml_asaxs
from utils import find_minmax, calc_rho, create_steps

from numba import jit

@jit(nopython=True)
def disk_ml_asaxs(q, H, R, RvvgtH, rho, eirho, adensity, Nalf):
    #RvvgtH: R>>H  Disk with infinite Radius
    dalf = np.pi/Nalf
    fft = np.zeros_like(q)
    ffs = np.zeros_like(q)
    ffc = np.zeros_like(q)
    ffr = np.zeros_like(q)
    Nlayers=len(H)
    tH=np.cumsum(H)
    V = np.pi*R**2*2*tH[:-1]
    drho=2.0*np.diff(np.array(rho))*V
    deirho=2.0*np.diff(np.array(eirho))*V
    dadensity=2.0*np.diff(np.array(adensity))*V
    for i, q1 in enumerate(q):
        for ialf in range(Nalf):
            alf = ialf*dalf + 1e-6
            tft = np.complex(0.0,0.0)
            tfs = 0.0
            tfr = 0.0
            for k in range(Nlayers-1):
                qh=q1 * tH[k] * np.cos(alf)
                fach=np.sin(qh)/(qh)
                qr=q1*R*np.sin(alf)
                facR=(1.0-RvvgtH)*j1(qr)/(qr)\
                     +RvvgtH*np.sin(qr-np.pi/2)/(qr)
                fac =  fach*facR
                tft = tft + drho[k] * fac
                tfs = tfs + deirho[k] * fac
                tfr = tfr + dadensity[k] * fac
            fft[i] = fft[i] + np.abs(tft) ** 2 * np.sin(alf)
            ffs[i] = ffs[i] + tfs ** 2 * np.sin(alf)
            ffc[i] = ffc[i] + tfs * tfr * np.sin(alf)
            ffr[i] = ffr[i] + tfr ** 2 * np.sin(alf)
        fft[i] = fft[i] * dalf
        ffs[i] = ffs[i] * dalf
        ffc[i] = ffc[i] * dalf
        ffr[i] = ffr[i] * dalf
    return fft,ffs,ffc,ffr




class Disk_Uniform: #Please put the class name same as the function name
    def __init__(self, x=0, Np=10, flux=1e13, dist='Gaussian', Energy=None, relement='Au', NrDep='False', R=1.0,RvvgtH=False,
                 Hsig=0.0, norm=1.0, sbkg=0.0, cbkg=0.0, abkg=0.0, D=1.0, phi=0.1, U=-1.0, SF='None',Nalf=200,term='Total',
                 mpar={'Layers': {'Material': ['Au', 'H2O'], 'Density': [19.32, 1.0], 'SolDensity': [1.0, 1.0],
                                  'Rmoles': [1.0, 0.0], 'H': [1.0, 0.0]}}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered centro-symmetric disk with different materials

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Energy      : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np          : No. of points with which the size distribution will be computed. Default: 10
        R           : Radius of te disk in Angs
        RvvgtH      : True R>>H and False otherwise
        NrDep       : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy independent)
        dist        : The probablity distribution fucntion for the radii of different interfaces in the nanoparticles. Default: Gaussian
        Hdist       : Width of distribution the thickness of the central layer
        Nalf        : Number of azumuthal angle points for angular averaging
        norm        : The density of the nanoparticles in Molar (Moles/Liter)
        sbkg        : Constant incoherent background for SAXS-term
        cbkg        : Constant incoherent background for cross-term
        abkg        : Constant incoherent background for Resonant-term
        flux        : Total X-ray flux to calculate the errorbar to simulate the errorbar for the fitted data
        D           : Hard Sphere Diameter
        phi         : Volume fraction of particles
        U           : The sticky-sphere interaction energy
        SF          : Type of structure factor. Default: 'None'
        term        : 'SAXS-term' or 'Cross-term' or 'Resonant-term'
        mpar        : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('Sol_Density' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Thickness ('H' in Angs)
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.norm=norm
        self.sbkg=sbkg
        self.cbkg=cbkg
        self.abkg=abkg
        self.dist=dist
        self.Hsig=Hsig
        self.Np=Np
        self.R=R
        self.RvvgtH=RvvgtH
        self.Nalf=Nalf
        self.Energy=Energy
        self.relement=relement
        self.NrDep=NrDep
        #self.rhosol=rhosol
        self.flux=flux
        self.D=D
        self.phi=phi
        self.U=U
        self.term=term
        self.__mpar__=mpar #If there is any multivalued parameter
        self.SF=SF
        self.choices={'RvvgtH':['True', 'False'],
                      'dist':['Gaussian','LogNormal'],'NrDep':['True','False'],
                      'SF':['None','Hard-Sphere', 'Sticky-Sphere'],
                      'term': ['SAXS-term', 'Cross-term', 'Resonant-term',
                               'Total']
                      } #If there are choices available for any fixed parameters
        self.__cf__=Chemical_Formula()
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
        self.params.add('D', value=self.D, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi', value=self.phi, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('R', value=self.R, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sbkg',value=self.sbkg,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('cbkg', value=self.cbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('abkg', value=self.abkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('U', value=self.U, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Hsig', value=self.Hsig, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mpar__.keys():
            for key in self.__mpar__[mkey].keys():
                if key != 'Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0.0,
                                        max=np.inf, expr=None, brute_step=0.1)
    @lru_cache(maxsize=10)
    def calc_Hdist(self, H, Hsig, dist, N):
        totalH = H[0]
        if Hsig > 0.001:
            fdist = eval(dist + '.' + dist + '(x=0.001, pos=totalH, wid=Hsig)')
            if dist=='Gaussian':
                hmin, hmax = max(0.001, totalH - 5 * Hsig), totalH + 5 * Hsig
            else:
                hmin, hmax = max(0.001, np.exp(np.log(totalH) - 5*Hsig)), np.exp(np.log(totalH) + 5*Hsig)
            dH = np.linspace(hmin, hmax, N)
            fdist.x = dH
            hdist = fdist.y()
            sumdist = np.sum(hdist)
            rdist = hdist / sumdist
            return dH, hdist, totalH
        else:
            return [totalH], [1.0], totalH

    @lru_cache(maxsize=10)
    def disk(self, q, R, RvvgtH, H, Hsig, rho, eirho, adensity, dist='Gaussian', Np=10, Nalf=1000):
        q = np.array(q)
        dH, hdist, totalH = self.calc_Hdist(H, Hsig, dist, Np)
        form = np.zeros_like(q)
        eiform = np.zeros_like(q)
        aform = np.zeros_like(q)
        cform = np.zeros_like(q)
        pfac = (2.818e-5 * 1.0e-8) ** 2
        ht=np.array(H)
        for i in range(len(dH)):
            ht[0] = dH[i]
            fft, ffs, ffc, ffr = disk_ml_asaxs(q, ht, R, RvvgtH, rho, eirho, adensity, Nalf)
            form = form + hdist[i] * fft
            eiform = eiform + hdist[i] * ffs
            aform = aform + hdist[i] * ffr
            cform = cform + hdist[i] * ffc
        return pfac * form, pfac * eiform, pfac * aform, np.abs(pfac * cform)  # in cm^2

    @lru_cache(maxsize=10)
    def disk_dict(self, q, R, RvvgtH, H, Hsig, rho, eirho, adensity, dist='Gaussian', Np=10, Nalf=1000):
        form, eiform, aform, cform = self.disk(q, R, RvvgtH, H, Hsig, rho, eirho, adensity, dist=dist, Np=Np,
                                                    Nalf=Nalf)
        sqf = {'Total': form, 'SAXS-term': eiform, 'Resonant-term': aform, 'Cross-term': cform}
        return sqf

    def update_params(self):
        mkey = self.__mkeys__[0]
        key = 'Density'
        Nmpar = len(self.__mpar__[mkey][key])
        self.__density__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'SolDensity'
        self.__solDensity__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Rmoles'
        self.__Rmoles__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'H'
        self.__H__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Material'
        self.__material__ = [self.__mpar__[mkey][key][i] for i in range(Nmpar)]

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        scale = 1e27 / 6.022e23
        svol = 1.5*0.0172**2/370**2  # scattering volume in cm^3
        self.update_params()
        if self.RvvgtH:
            RvvgtH=1.0
        else:
            RvvgtH=0.0
        rho, eirho, adensity, rhor, eirhor, adensityr = calc_rho(R=tuple(self.__H__), material=tuple(self.__material__),
                                                                 relement=self.relement,
                                                                 density=tuple(self.__density__),
                                                                 sol_density=tuple(self.__solDensity__),
                                                                 Energy=self.Energy, Rmoles=tuple(self.__Rmoles__),
                                                                 NrDep=self.NrDep)
        if type(self.x) == dict:
            sqf = {}
            key='SAXS-term'
            sqft=self.disk_dict(tuple(self.x[key]), self.R, RvvgtH, tuple(self.__H__), self.Rsig,
                                tuple(rho), tuple(eirho), tuple(adensity),
                                dist=self.dist, Np=self.Np, Nalf=self.Nalf)
            if self.SF is None:
                struct = np.ones_like(self.x[key])  # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x[key], D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x[key], D=self.D, phi=self.phi, U=self.U, delta=0.01)
            for key in self.x.keys():
                if key == 'SAXS-term':
                    sqf[key] = self.norm * 6.022e20 *sqft[key] * struct + self.sbkg # in cm^-1
                if key == 'Cross-term':
                    sqf[key] = self.norm * 6.022e20 *sqft[key] * struct + self.cbkg # in cm^-1
                if key == 'Resonant-term':
                    sqf[key] = self.norm * 6.022e20 *sqft[key] * struct + self.abkg # in cm^-1
            key1='Total'
            total= self.norm * 6.022e20 *sqft[key1] * struct + self.sbkg
            if not self.__fit__:
                dH, hdist, totalH = self.calc_Hdist(tuple(self.__H__), self.Hsig, self.dist, self.Np)
                self.output_params['Distribution'] = {'x': dH, 'y': hdist}
                self.output_params['Total'] = {'x': self.x[key], 'y':total}
                for key in self.x.keys():
                    self.output_params[key] = {'x': self.x[key], 'y': sqf[key]}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],
                                               'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],
                                                 'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1] * scale,
                                                    'names': ['r (Angs)', 'Density (Molar)']}
                self.output_params['Structure_Factor'] = {'x': self.x[key], 'y': struct}
                xtmp,ytmp=create_steps(x=self.__H__[:-1],y=self.__Rmoles__[:-1])
                self.output_params['Rmoles_normal']={'x':xtmp,'y':ytmp}
                xtmp, ytmp = create_steps(x=self.__H__[:-1], y=self.__density__[:-1])
                self.output_params['Density_normal'] = {'x': xtmp, 'y': ytmp}
        else:
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)

            tsqf, eisqf, asqf, csqf = self.disk(tuple(self.x), self.R, RvvgtH, tuple(self.__H__), self.Hsig,
                                                     tuple(rho), tuple(eirho),
                                                      tuple(adensity), dist=self.dist, Np=self.Np, Nalf=self.Nalf)
            sqf = self.norm * np.array(tsqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
            if not self.__fit__: #Generate all the quantities below while not fitting
                asqf = self.norm * np.array(asqf) * 6.022e20 * struct + self.abkg  # in cm^-1
                eisqf = self.norm * np.array(eisqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
                csqf = self.norm * np.array(csqf) * 6.022e20 * struct + self.cbkg  # in cm^-1
                sqerr = np.sqrt(self.norm*6.022e20*self.flux * tsqf * svol*struct+self.sbkg)
                sqwerr = (self.norm*6.022e20*tsqf * svol * struct*self.flux+self.sbkg + 2 * (0.5 - np.random.rand(len(tsqf))) * sqerr)
                self.output_params['simulated_total_w_err'] = {'x': self.x, 'y': sqwerr, 'yerr': sqerr}
                self.output_params['Total'] = {'x': self.x, 'y': sqf}
                self.output_params['Resonant-term'] = {'x': self.x, 'y': asqf}
                self.output_params['SAXS-term'] = {'x': self.x, 'y': eisqf}
                self.output_params['Cross-term'] = {'x': self.x, 'y': csqf}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],
                                               'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],
                                                 'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1] * scale,
                                                    'names': ['r (Angs)', 'Density (Molar)']}
                self.output_params['Structure_Factor'] = {'x': self.x, 'y': struct}
                xtmp, ytmp = create_steps(x=self.__H__[:-1], y=self.__Rmoles__[:-1])
                self.output_params['Rmoles_normal'] = {'x':xtmp , 'y': ytmp}
                sqf = self.output_params[self.term]['y']
                xtmp, ytmp = create_steps(x=self.__H__[:-1], y=self.__density__[:-1])
                self.output_params['Density_normal'] = {'x': xtmp, 'y': ytmp}
                dr, rdist, totalR = self.calc_Hdist(tuple(self.__H__), self.Hsig, self.dist, self.Np)
                self.output_params['Distribution'] = {'x': dr, 'y': rdist}
        return sqf



if __name__=='__main__':
    x=np.logspace(-3,0.0,100)
    fun=Disk_Uniform(x=x)
    print(fun.y())

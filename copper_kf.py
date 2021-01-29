#!/usr/bin/env python3

import importlib

import numpy as np
import sympy
import sym2num.model
from scipy import integrate
from scipy import interpolate
from scipy import signal

from ceacoest import kalman
from ceacoest.kalman import base, extended, unscented
from ceacoest.modelling import symsde, symstats


rbf_guess = np.load('rbf_guess.npz')
extrap = rbf_guess['extrap']
centers = rbf_guess['centers']
scale = rbf_guess['scale']
weights = rbf_guess['weights']


class SymbolicRBFCu(sym2num.model.Base):
    """Symbolic continuous-time RBF copper model."""
    
    generate_functions = ['f']
    
    def __init__(self):
        super().__init__()
        
        v = self.variables
        v['self']['g3'] = 'g3'
        v['self']['x1_meas_std'] = 'x1_meas_std'
        
        v['x'] = ['x1', 'x2', 'x3']
        v['y'] = ['x1_meas']
        v['t'] = 't'
        
        self.set_default_members()
    
    @sym2num.model.collect_symbols
    def f(self, t, x, *, s):
        """Drift function."""
        f1 = s.x2
        f2 = s.x3

        nc = len(weights)
        r_sq = np.sum((centers - x) ** 2, 1)
        rbf_basis = [sympy.exp(-scale * r_sqi) for r_sqi in r_sq]
        f3 = weights @ rbf_basis - extrap @ x
        return [f1, f2, f3]
    
    @sym2num.model.collect_symbols
    def g(self, t, x, *, s):
        """Diffusion matrix."""
        return [[0], [0], [s.g3]]
    
    @sym2num.model.collect_symbols
    def h(self, t, x, *, s):
        """Measurement function."""
        return [s.x1]
    
    @sym2num.model.collect_symbols
    def R(self, *, s):
        """Measurement covariance."""
        return [[s.x1_meas_std ** 2]]


#class SymbolicDiscretizedRBFCu(symsde.ItoTaylorAS15DiscretizedModel):
class SymbolicDiscretizedRBFCu(symsde.EulerDiscretizedSDEModel):
    
    ContinuousTimeModel = SymbolicRBFCu

    #def __init__(self):
    #    super().__init__()
    #    
    #    self.add_derivative('h', 'x', 'dh_dx')
    #    self.add_derivative('f', 'x', 'df_dx')
    #
    #@property
    #def generate_functions(self):
    #    return ['dh_dx', 'df_dx', *super().generate_functions]


if __name__ == '__main__':
    y_bitlen = 0.1 * 0.02442

    data = np.load('data.npz')
    start = 120
    stop = 1000
    y = data['y'][start:stop, None]
    x = data['x'][start:stop]
    t = data['t'][start:stop]
    N = len(y)
    
    sym_disc_mdl = SymbolicDiscretizedRBFCu()
    model = sym_disc_mdl.compile_class()()
    
    params = dict(
        g3=0.05, x1_meas_std=y_bitlen, dt=data['Ts'],
    )
    for k,v in params.items():
        setattr(model, k, v)

    x0 = x[0]
    Px0 = np.diag([8e-5, 2e-3, 2e-2])
    ukf = kalman.DTUnscentedFilter(model, x0, Px0)
    [xuf, Pxuf] = ukf.filter(y)
    

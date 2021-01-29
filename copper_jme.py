#!/usr/bin/env python3

import argparse
import importlib

import numpy as np
import sympy
import sym2num.model
from numpy import ma
from scipy import integrate, interpolate, stats, signal

from ceacoest import jme
from ceacoest.modelling import symjme, symstats


class SymbolicRBFCuJME(symjme.Model):
    """Symbolic continuous-time RBF copper JME estimation model."""
    
    collocation_order = 3
    
    def __init__(self, centers, scale, extrap):
        self.centers = centers
        """RBF centers."""
        
        self.scale = scale
        """RBF scale."""

        self.extrap = extrap
        """Envelope extrapolation weights."""
        
        v = self.Variables(
            x=['x1', 'x2', 'x3'],
            y=['x1_meas'],
            u=[],
            p=[*(f'weight{i}' for i in range(len(centers))), 'x1_meas_std'],
            G=[['g1'], ['g2'], ['g3']],
        )
        super().__init__(v, use_penalty=False)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        r_sq = np.sum((self.centers - x) ** 2, 1)
        rbf_basis = [sympy.exp(-self.scale * r_sqi) for r_sqi in r_sq]
        weights = p[:-1]
        
        f1 = s.x2
        f2 = s.x3
        f3 = weights @ rbf_basis - self.extrap @ x
        return [f1, f2, f3[()]]
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        return symstats.normal_logpdf1(s.x1_meas, s.x1, s.x1_meas_std)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_code', action='store_true')
    parser.add_argument('--import_model', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    rbf_guess = np.load('rbf_guess.npz')
    extrap = rbf_guess['extrap']
    centers = rbf_guess['centers']
    scale = rbf_guess['scale']
    weights = rbf_guess['weights']

    # Instantiate model class
    if args.import_model:
        import copper_jme_generated
        importlib.reload(copper_jme_generated)
        model = copper_jme_generated.GeneratedSymbolicRBFCuJME()
    else:
        sym_mdl = SymbolicRBFCuJME(centers, scale, extrap)
        code = sym_mdl.print_code()
        if args.save_model_code:
            with open('copper_jme_generated.py', 'w') as model_file:
                print(code, file=model_file)
        env = {}
        exec(compile(code, '<string>', 'exec'), env)
        model = env['GeneratedSymbolicRBFCuJME']()
    
    data = np.load('data.npz')
    y_bitlen = 0.1 * 0.02442
    start = 120
    stop = 2000
    y = data['y'][start:stop, None]
    x = data['x'][start:stop]
    t = data['t'][start:stop] - data['t'][start]
    N = len(y)
    
    model.G = np.array([[0], [0], [0.04]])
    #model.penweight = np.array([1.0, 1.0, 1.0])
    ufun = lambda t: np.empty((np.size(t), 0))
    problem = jme.Problem(model, t, y, ufun)
    tc = problem.tc
    
    # Define initial guess
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['x'][:] = interpolate.interp1d(t, x, axis=0)(tc)
    var0['p'][-1] = 2 * y_bitlen
    var0['p'][:-1] = weights
    
    # Define bounds on variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['p'][-1] = 2 * y_bitlen# / 3
    var_U['p'][-1] = 2 * y_bitlen# * 10
    #var_L['p'][:-1] = weights
    #var_U['p'][:-1] = weights
    
    # Define bounds on constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    # Define problem scaling
    dec_scale = np.ones(problem.ndec)
    constr_scale = np.ones(problem.ncons)
    obj_scale = -1.0
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 10.0)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    popt = opt['p']

import argparse

import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import signal


def rbf_basis(centers, scale, xev):
    xev = np.asarray(xev)
    if xev.ndim == 2:
        diff = xev[:, None] - centers
    else:
        diff = xev - centers
    r_sq = np.sum(diff ** 2, -1)
    return np.exp(-scale * r_sq)


def sel_centers(x, thresh):
    centers = []
    thresh_sq = thresh ** 2
    while x.size:
        i = np.random.randint(len(x))
        centers.append(x[i])
        dist = np.sum((x - x[i]) ** 2, -1)
        x = x[dist > thresh_sq]
    return np.array(centers)


def cdiff(x, T=1):
    xd = np.empty_like(x)
    xd[1:-1] = 0.5 / T * (x[2:] - x[:-2])
    xd[0] = xd[1]
    xd[-1] = xd[-2]
    return xd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_rbf', action='store_true')
    parser.add_argument('--save_data', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    np.random.seed(0)
    
    bitlen = 0.02442
    Ts_orig = 1 / 1500
    data = np.loadtxt('Tcopp.txt')    
    N = len(data)
    
    y = 0.1 * (data - 45)
    Ts = 0.1
    t = Ts * np.arange(N)
        
    lpf = signal.butter(3, 0.3, output='sos')
    yf0 = signal.sosfiltfilt(lpf, y)
    yf1 =  cdiff(yf0, Ts)
    yf2 =  cdiff(yf1, Ts)
    yf3 =  cdiff(yf2, Ts)
    
    x = np.c_[yf0, yf1, yf2]
    thresh = 0.25
    scale = 1 / (2.85 * thresh) ** 2
    centers = sel_centers(x, thresh)
    A = rbf_basis(centers, scale, x)

    if args.save_data:
        np.savez('data', x=x, y=y, t=t, Ts=Ts)
    
    # Add linear stabilizing component for extrapolation
    extrap = 1 * np.asarray([3,3,1], float)
    fregul = yf3 + x @ extrap
    weights = np.linalg.lstsq(A, fregul, rcond=None)[0]

    if args.save_rbf:
        save_vars = dict(
            extrap=extrap, centers=centers, scale=scale, weights=weights
        )
        np.savez('rbf_guess', **save_vars)

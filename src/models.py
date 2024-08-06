import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
from sklearn.utils.extmath import randomized_svd

class SoftImpute:
    def __init__(self, lambda_thres=None, J=None, maxit=100, tol=1e-5, verbose=True):
        self.lambda_thres = lambda_thres
        self.J = J
        self.maxit = maxit
        self.tol = tol
        self.verbose = verbose
        
    def fit_transform(self, X):
        m, n = X.shape
        obs_idx = ~np.isnan(X)
        
        if self.lambda_thres is None:
            s = self._max_singular_value(X, obs_idx)
            self.lambda_thres = s/50
        if self.J is None:
            self.J = min(m,n)-1

        Zold = np.zeros_like(X)
        cnt = 0
        error = float('inf')
        while cnt < self.maxit and error > self.tol:
            Z = Zold.copy()
            Z[obs_idx] = X[obs_idx]
            svd = np.linalg.svd(Z, full_matrices=False)
            sin_val = np.clip(svd[1]-self.lambda_thres, 0, float('inf'))
            rank = (sin_val > 0).sum()
            rank = min(rank, self.J)
            Z = (svd[0][:,:rank]*sin_val[:rank]).dot(svd[2][:rank,:])
            error = np.sum((Z - Zold)**2) / np.sum(Zold**2) if not np.allclose(Zold,0) else float('inf')
            cnt += 1
            Zold = Z
            if self.verbose:
                print(f'Iterations: {cnt}/{self.maxit}, error: {error:.6f}')
        if cnt == self.maxit:
            print(f'Reached maximal iterations instead of convergence with error {error:.9f}')
        else:
            print(f'Error {error:.9f} in {cnt}/{self.maxit} is smaller that tolerance {self.tol:.9f}.')

        return Z
    
    def _max_singular_value(self, X, obs_idx):
        Z = X.copy()
        Z[~obs_idx] = [0]
        _, s, _ = randomized_svd(Z,1,random_state=42)
        return s[0]
    
class pairwiseModel(torch.nn.Module):
    def __init__(self, Y_obs):
        super().__init__()
        m, n = Y_obs.shape
        self.mu = torch.nn.Linear(n, m, bias=False)
        self.idx = torch.tensor(~np.isnan(Y_obs))
        self.Y = torch.tensor(Y_obs[self.idx])

    def forward(self, x):
        idx_i = x[0]
        idx_j = x[1]
        y_diff = x[2]
        
        mu_diff = self.mu.weight[idx_i[:,0],idx_i[:,1]] - self.mu.weight[idx_j[:,0],idx_j[:,1]]
        
        loss = torch.mean(torch.log(1+torch.exp(-y_diff*mu_diff)))
        return loss
    

class MaxNorm:
    def __init__(self, R=10, alpha=10, rho=0.1, maxit=100, tol=1e-6, verbose=True):
        self.R = R
        self.alpha = alpha
        self.rho = rho
        self.maxit = maxit
        self.tol = tol
        self.verbose = verbose

    def fit_transform(self, Ynan):
        
        d1, d2 = Ynan.shape
        Y = torch.tensor(Ynan, dtype=torch.float32)
        idx = torch.isnan(Y)
        Y[idx] = 0
        
        Xupper = torch.cat((torch.eye(d1), Y), axis=1)
        Xlower = torch.cat((Y.T, torch.eye(d2)), axis=1)
        Xinit = torch.cat((Xupper, Xlower), axis=0)
        Winit = Xinit.clone()
        Zinit = torch.eye(d1+d2)
        X, W, Z = Xinit, Winit, Zinit
        
        prev_diff = float('inf')
        cnt = 0
        error = float('inf')
        R, alpha, rho = self.R, self.alpha, self.rho
        while cnt < self.maxit and error > self.tol:
            X = W - 1/rho*Z
            L, V = torch.linalg.eig(X)
            real = L.real
            real[abs(L.imag) > 1e-6] = 0
            L = real
            L = torch.clip(L, 0)
            X = torch.matmul(V.real * L, V.real.T)
            
            W = X + 1/rho*Z
            
            W1 = W[:d1, :d1]
            W2 = W[:d1, d1:]
            
            W4 = W[d1:, d1:]

            W1 = torch.clip(W1, -R, R)
            ind = np.diag_indices(W1.shape[0])
            W1[ind[0], ind[1]] = torch.clip(torch.diag(W1), 0, R)

            W4 = torch.clip(W4, -R, R)
            ind = np.diag_indices(W4.shape[0])
            W4[ind[0], ind[1]] = torch.clip(torch.diag(W4), 0, R)


            W2[idx] = torch.clip(W2[idx], -alpha, alpha)
            W2[~idx] = torch.clip((Y[~idx] + rho*W2[~idx])/(1+rho), -alpha, alpha)

            W = torch.cat((torch.cat((W1, W2), axis=1), torch.cat((W2.T, W4), axis=1)), axis=0)
            
            Z = Z + 1.618*rho * (X-W)

            cur_diff = torch.sqrt(torch.mean((X-W)**2))
            error = abs(cur_diff-prev_diff)
            prev_diff = cur_diff
            cnt += 1
            
            if self.verbose:
                print(f'Iterations: {cnt}/{self.maxit}, error: {error:.6f}.')
        if cnt == self.maxit:
            print(f'Reached maximal iterations instead of convergence with error {error:.7f}')
        else:
            print(f'Error {error:.7f} at {cnt}/{self.maxit} is smaller that tolerance {self.tol:.9f}.')
            
        return W[:d1, d1:]

class ModelFreeWeighting:
    def __init__(self, kappa=1e-8, mu=30, beta=10, rho=0.1, maxit=300, tol=1e-5, verbose=True):
        self.kappa = kappa
        self.mu = mu
        self.beta = beta
        self.rho = rho
        self.tol = tol
        self.maxit = maxit
        self.tol = tol
        self.W = None
        self.verbose = verbose
    def fit_transform(self, X): 
        
        
        Y = X.copy()
        D = (~np.isnan(Y)).astype(float)
        if self.W is None:
            self.D = D
            x0 = np.zeros_like(D[D==1])
            bounds = [(0,float("inf")) for i in D[D==1]]
            res = scipy.optimize.minimize(fun=self._cal_weight_fun, x0=x0, method='L-BFGS-B', jac=True, bounds=bounds)
            print(f'Objective funtion val for weights: {res.fun:.6f}')
            W = np.ones_like(D)
            W[D==1] = res.x + 1
            W = W.real
            self.W = W
        else:
            W = self.W

        Xres = self._admm(Y, W, D)
        self.Xres = Xres

        return Xres

    def _cal_weight_fun(self, x):
        D, kappa = self.D, self.kappa
        W = np.zeros_like(D)
        W[D==1] = x+1
        J = np.ones_like(D)
        X = D*W - J
        u, d, v = randomized_svd(X,1,random_state=42)
        loss = d[0] + kappa*np.sum((D*W)**2)
        
        grad = np.matmul(u[:,0].reshape(-1,1), v[0,:].reshape(1,-1))
        grad = grad+2*kappa*D
        grad = grad[D==1]
        return (loss, grad)

    def _admm(self, Y, W, D):
        n1, n2 = Y.shape
        Y[D==0] = 0
        Zupper = np.hstack((np.diag(np.ones(n1)), Y))
        Zlower = np.hstack((Y.T, np.diag(np.ones(n2))))
        Zini = np.vstack((Zupper, Zlower))
        
        Xini = Zini.copy()
        Vini = np.diag(np.ones(n1+n2))
        I = np.diag(np.ones(n1+n2))

        Z = Zini
        X = Xini
        V = Vini

        old_obj = float('inf')
        cnt = 0 
        error = float('inf')
        rho, mu, beta = self.rho, self.mu, self.beta
        
        while cnt < self.maxit and error > self.tol:
            X = Z - 1/rho * (V+mu*I)

            eig = np.linalg.eig(X)
            eig_val = eig[0]
            real = eig_val.real
            real[abs(eig_val.imag)>1e-6] = 0
            eig_val = real
            eig_val = np.clip(eig_val,0,float('inf'))
            # rank = (eig_val>0).sum()
            X = (eig[1].real*eig_val).dot(eig[1].T.real)

            cand = X + 1/rho * V

            cand[:n1,:n1] = np.clip(cand[:n1,:n1], -beta, beta)
            cand[n1:,n1:] = np.clip(cand[n1:,n1:], -beta, beta)

            np.fill_diagonal(cand, np.clip(np.diag(cand), 0, beta))

            proxy_Y = cand[:n1, n1:]
            proxy_Y[D==1] = np.clip((Y[D==1]*W[D==1] + rho*proxy_Y[D==1])/(W[D==1]+rho),-beta,beta)
            proxy_Y[D==0] = np.clip(proxy_Y[D==0],-beta,beta)
            cand[:n1,n1:] = proxy_Y
            cand[n1:,:n1] = proxy_Y.T
            
            Z = cand
            V = V+1.618*rho*(X-Z)
            new_obj = self._cal_obj(Y, W, D, X, Z)
            error = abs(old_obj - new_obj)
            old_obj = new_obj
            cnt += 1
            if self.verbose:
                print(f'Iterations: {cnt}/{self.maxit}, error: {error:.6f}.')
            
        if cnt == self.maxit:
            print(f'Reached maximal iterations instead of convergence with error {error:.7f}')
        else:
            print(f'Error {error:.7f} at {cnt}/{self.maxit} is smaller that tolerance {self.tol:.9f}.')
        return Z[:n1,n1:]
        
    def _cal_obj(self, Y, W, D, X, Z):
        return np.sqrt(np.mean((X-Z)**2))
    

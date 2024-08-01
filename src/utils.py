import numpy as np
import torch
import sklearn.linear_model
import matplotlib.pyplot as plt
import seaborn as sns

class GaussianLogistic:
    def __init__(self, m=50, n=50, r=3, seed=42,
                 scale=3, shift=0, std=1.):
        np.random.seed(seed)
        self.m, self.n, self.r = m, n, r
        L = np.random.normal(size=(m,r))
        R = np.random.normal(size=(r,n))
        A = np.matmul(L, R)
        A = A - A.mean()
        
        Y = np.random.normal(loc=A, scale=std)
        P = 1/(1+np.exp(scale*(Y-shift)))
        D = np.random.binomial(1, P)
        
        D_rawtest = 1-D
        
        self.A = A
        self.D = D
        self.D_rawtest = D_rawtest
        
        x_loc, y_loc = np.where(D_rawtest > 0)
        length = D_rawtest.sum()

        shuffle = list(range(length))
        np.random.shuffle(shuffle)

        Dval = np.zeros_like(D)
        Dtest = np.zeros_like(D)
        
        for idx in range(length//2):
            Dval[x_loc[shuffle[idx]], y_loc[shuffle[idx]]] = 1
            
        for idx in range(length//2, length):
            Dtest[x_loc[shuffle[idx]], y_loc[shuffle[idx]]] = 1
        
        self.Dval = Dval
        self.Dtest = Dtest
        
        self.Y = Y
        Ynan = Y.copy()
        Ynan[D<1e-3] = np.nan
        self.Ynan = Ynan
        
        train_length = D.sum()
        val_length = length // 2
        test_length = length - length // 2
        print(f'train: {train_length}, val: {val_length}, test: {test_length}')
        
def hist_plot(res_dict, methods, key):
    sim = res_dict['sim']
    fig, axes = plt.subplots(2,4)
    fig.set_size_inches(16, 8)
    A = np.concatenate((sim.A[sim.D==1], sim.Atest[sim.Dtest==1]))
    Atest = sim.Atest
    
    for i in range(2):
        for j in range(4):
            ax = axes[i][j]
            if i==1 and j==2:
                method = methods[i*4+j-1]
                sns.histplot(res_dict[method]['X'][sim.Dtest==1], bins=100, ax=ax, label=method)
                sns.histplot(A, bins=100, ax=ax, label='all', color='lightblue')
            else:
                sns.histplot(A, bins=100, ax=ax, label='all', color='lightblue')
                if i==0 and j==0:
                    sns.histplot(sim.A[sim.D == 1], bins=100, ax=ax, label='observed')
                else:
                    method = methods[i*4+j-1]
                    sns.histplot(res_dict[method]['X'][sim.Dtest==1], bins=100, ax=ax, label=method)
            ax.legend()
    _ = fig.suptitle(key.title() + " recoverd entries on missing")
    fig.show()

def get_diff(Y):
    obs_idx = np.argwhere(~np.isnan(Y))
    n_obs = obs_idx.shape[0]
    Y_diff = []
    diff_idx = []
    for i in range(n_obs-1):
        for j in range(i+1, n_obs):
            Y1 = Y[obs_idx[i,0], obs_idx[i,1]]
            Y2 = Y[obs_idx[j,0], obs_idx[j,1]]
            Y_diff.append(Y1 - Y2)
            diff_idx.append((i,j))
    x_loc = torch.tensor(obs_idx[[x[0] for x in diff_idx], :], dtype=torch.int64)
    y_loc = torch.tensor(obs_idx[[x[1] for x in diff_idx], :], dtype=torch.int64)
    Y_diff = torch.Tensor(Y_diff)
    
    return x_loc, y_loc, Y_diff

def get_error(X, sim):
    mu = X
    mu_val = mu[sim.Dval > 0.5]
    data_val = sim.A[sim.Dval > 0.5]
    mu_test = mu[sim.Dtest > 0.5]
    data_test = sim.A[sim.Dtest > 0.5]

    trans_mu_val = mu_val
    trans_mu_test = mu_test
    val_rmse = np.sqrt(((trans_mu_val - data_val)**2).mean())
    val_mae = np.abs(trans_mu_val - data_val).mean()
    rmse = np.sqrt(((trans_mu_test - data_test)**2).mean())
    mae = np.abs(trans_mu_test - data_test).mean()
    
    # u,d,v = np.linalg.svd(mu)
    # rank = (d>1e-6).sum()
    # print(f'RMSE: {rmse:.6f}, MAE: {mae:.6f}, rank: {rank}')
    return val_rmse, val_mae, rmse, mae

def get_error2(X, sim):
    mu = X
    mu_val = mu[sim.Dval > 0.5]
    data_val = sim.A[sim.Dval > 0.5]
    mu_test = mu[sim.Dtest > 0.5]
    data_test = sim.A[sim.Dtest > 0.5]
    
    lr_reg = sklearn.linear_model.LinearRegression()
    lr_reg.fit(mu_val.reshape(-1,1), data_val)
    trans_mu_val = lr_reg.predict(mu_val.reshape(-1,1))
    val_rmse = np.sqrt(((trans_mu_val - data_val)**2).mean())
    val_mae = np.abs(trans_mu_val - data_val).mean()
    
    trans_mu_test = lr_reg.predict(mu_test.reshape(-1,1))
    rmse = np.sqrt(((trans_mu_test - data_test)**2).mean())
    mae = np.abs(trans_mu_test - data_test).mean()
    
    # u,d,v = np.linalg.svd(mu)
    # rank = (d>1e-6).sum()
    # print(f'RMSE: {rmse:.6f}, MAE: {mae:.6f}, rank: {rank}')
    return val_rmse, val_mae, rmse, mae

class Dataset:
    def __init__(self, val_size = 0.5, seed=42):
        self.val_size = val_size
        self.seed = seed
        
    def _get_val_data(self):
        self.Dtrain = self.D.copy()

        print(f'{self.name}. Observing ratio: {int(self.D.sum())}/{self.D.shape[0]*self.D.shape[1]}')
        np.random.seed(self.seed)
        self.test_len = self.Atest.shape[0]
        Dtest = self.Dtest.copy()
        n = (Dtest==1).sum()
        val_size = int(n*self.val_size)
        ind = np.concatenate((np.ones(n-val_size), np.zeros(val_size)))
        np.random.shuffle(ind)
        Dtest[Dtest == 1] = ind
        self.Dval = self.Dtest - Dtest
        self.Dtest = Dtest
        print(f'{self.name}. Observing ratio in train : {int(self.Dtrain.sum())}/{self.Dtrain.shape[0]* self.Dtrain.shape[1]}')
        print(f'{self.name}. Observing ratio in val: {int(self.Dval.sum())}/{self.Dval.shape[0]*self.Dval.shape[1]}')
        print(f'{self.name}. Observing ratio in test: {int(self.Dtest.sum())}/{self.Dtest.shape[0]*self.Dtest.shape[1]}')
        Dnan = self.Dtrain.copy().astype(float)
        Dnan[Dnan == 0] = np.nan
        self.Ynan = self.A * Dnan
        self.Dnan = Dnan
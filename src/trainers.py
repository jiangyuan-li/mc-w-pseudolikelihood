import numpy as np
import torch
from .models import *

def si_tune(pool, sim):
    workers = []
    for thres in np.exp(np.linspace(1e-3, 3, 10))-1:
        workers.append(pool.apply_async(si_solve, [sim, thres]))
    return workers


def maxnorm_tune(pool, sim):
    workers = []
    for R in [1, 5]:
        for alpha in [1, 5]:
            for rho in [0.1, 0.01]:
                workers.append(pool.apply_async(
                    maxnorm_solve, [sim, R, alpha, rho]))
    return workers


def mfw_tune(pool, sim):
    workers = []
    for kappa in [1e-9]:
        for mu in [10, 5]:
            for beta in [10, 5]:
                for rho in [0.1, 0.01]:
                    workers.append(pool.apply_async(
                        mfw_solve, [sim, kappa, mu, beta, rho]))
    return workers


def pseudo_tune(pool, sim):
    workers = []
    for thres in np.exp(np.linspace(1e-3, 2.5, 10))-1:
        workers.append(pool.apply_async(pseudo_solve, [sim, 0, thres]))
    return workers


def snn_tune(pool, sim):
    workers = []
    for n_neighbor in [1]:
        workers.append(pool.apply_async(snn_solve, [sim, n_neighbor]))
    return workers


def si_solve(sim, thres):
    si = SoftImpute(thres, verbose=False)
    X_si = si.fit_transform(sim.Ynan)
    tmp_res = get_error(X_si, sim)
    return tmp_res, X_si


def maxnorm_solve(sim, R, alpha, rho):
    maxnorm = MaxNorm(R=R, alpha=alpha, rho=rho, verbose=False)
    X_max = maxnorm.fit_transform(sim.Ynan)
    # print(f'R:{R},alpha:{alpha},rho:{rho}')
    tmp_res = get_error(X_max, sim)
    return tmp_res, X_max


def mfw_solve(sim, kappa, mu, beta, rho):
    mfw = ModelFreeWeighting(
        kappa=kappa, mu=mu, beta=beta, rho=rho, verbose=False)
    X_mfw = mfw.fit_transform(sim.Ynan)
    # print(f'kappa:{kappa},mu:{mu},beta:{beta},rho:{rho}')
    tmp_res = get_error(X_mfw, sim)
    return tmp_res, X_mfw


def snn_solve(sim, n_neighbor):
    snn = SyntheticNearestNeighbors(n_neighbors=n_neighbor, verbose=False)
    Xsnn = snn.fit_transform(sim.Ynan)
    Xsnn[np.isnan(Xsnn)] = 0
    tmp_res = get_error(Xsnn, sim)
    return tmp_res, Xsnn


def pocs(X, a=10, epochs=1000, eps=1e-6):
    Xprev = X.clone()
    for _ in range(epochs):
        X = Xprev - Xprev.mean()
        X = torch.clip(X, -a, a)
        if torch.norm(X-Xprev) < eps:
            break
        Xprev = X
    return X


def pseudo_solve(sim, idx, thres):
    es = EarlyStopping()
    si = SoftImpute(thres, verbose=False)
    X_si = si.fit_transform(sim.Ynan)
    model = pairwiseModel(sim.Ynan)
    # torch.nn.init.normal_(model.mu.weight)
    model.mu.weight.data = torch.tensor(X_si)  # model.mu.weight.data.mean()
    lr = 1.
    thres = thres
    epochs = 100
    x = get_diff(sim.Ynan)

    # reload(Trainer)
    # Trainer(model, x, sim, lr, thres, epochs, name)

    for epoch in range(epochs):
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        grad = model.mu.weight.grad.data
        # mean_grad = grad.mean()
        model.mu.weight.data -= lr * grad

        u, d, v = torch.svd(model.mu.weight.data)
        d = torch.clip(d-thres, 0)
        nuclear_norm = d.sum().item()
        obj = loss.item()+thres*nuclear_norm
        # print(f'objective: {obj:.6f}')
        rank = (d > 0).sum()
        model.mu.weight.data = torch.matmul(
            torch.matmul(u[:, :rank], torch.diag_embed(d[:rank])),
            v[:, :rank].transpose(-2, -1))

        model.mu.weight.data = pocs(model.mu.weight.data)
        # loss = model(x)

        # u, d, v = torch.svd(model.mu.weight.data)
        # rank = (d>1e-6).sum()
        # print(f'epoch: {epoch}, loss: {loss.item():.6f}, nuclear_norm: {nuclear_norm:.6f}, rank: {rank}')

        # obj = loss.item()+thres*d.sum().item()
        # print(f'objective: {obj:.6f}')
        mu = model.mu.weight.data.detach().numpy()
        Xpseudo = mu

        # Xpseudo = np.exp(Xpseudo)
        tmp_res = get_error2(Xpseudo, sim)
        es(-tmp_res[idx])
        # if es.early_stop:
        #   return tmp_res
    return tmp_res, Xpseudo

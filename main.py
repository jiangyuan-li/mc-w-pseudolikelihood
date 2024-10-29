import os, argparse
import pandas as pd
from src import *
from multiprocessing import Pool


def main():
    parser = argparse.ArgumentParser(
        description="Experiment with simulated data from Gaussian distributions")
    parser.add_argument("--name", default="gaussian-sim",
                        type=str, help="name of task")
    parser.add_argument("--m", default=50, type=int, help="number of rows")
    parser.add_argument("--n", default=50, type=int, help="number of columns")
    parser.add_argument("--scale", default=3, type=int, help="Gaussian mean")
    parser.add_argument("--std", default=1, type=int, help="Gaussian variance")
    parser.add_argument("--shift", default=2, type=int,
                        help="center shift in mssing")
    parser.add_argument("--reps", default=9, type=int, help="number of repititions for each run")

    args = parser.parse_args()
    print("Running with following command line arguments: {}".
          format(args))

    name = args.name
    m = args.m
    n = args.n
    scale = args.scale
    std = args.std
    shift = args.shift
    reps = args.reps
    sim = GaussianLogistic(scale=scale, shift=shift, std=std, m=m, n=n)

    # Distribution plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    sns.histplot(sim.A[sim.D_rawtest > 0],
                 label='unobserved', color='lightblue', ax=ax)
    sns.histplot(sim.A[sim.D > 0], label='observed', ax=ax)
    ax.set_title('Observed/Unobserved entries')
    ax.legend()
    fig.tight_layout()
    fig.savefig('figs/'+name+'_observation.pdf')

    # Generate calculation pools
    pool = Pool()
    tunes = {'si': si_tune, 'mfw': mfw_tune, 'maxnorm': maxnorm_tune,
             'snn': snn_tune, 'pseudo': pseudo_tune}
    keys = ['si', 'maxnorm', 'mfw', 'snn', 'pseudo']

    params = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    reps = reps

    # Put tasks into pools
    workers = {}
    for k in keys:
        workers[k] = {}
        for std in params:
            workers[k][std] = {}
    for std in params:
        for rep in range(reps):
            sim = GaussianLogistic(
                scale=-3., shift=2, m=100, n=100, std=np.sqrt(std), seed=42+rep*10)
            for k in keys:
                workers[k][std][rep] = tunes[k](pool, sim)
    pool.close()
    pool.join()

    # Fetch all results
    res = {}
    for k in keys:
        res[k] = {}
        for std in params:
            res[k][std] = {}
            for rep in range(reps):
                res[k][std][rep] = []
                for worker in workers[k][std][rep]:
                    res[k][std][rep].append(worker.get())

    # Fetch results after tunning
    fres = {}
    tres = {}
    idx = 0
    for k in keys:
        fres[k] = {}
        tres[k] = {}
        for std in params:
            fres[k][std] = []
            tres[k][std] = []
            for rep in range(len(res[k][std])):
                tmpres = res[k][std][rep]
                tres[k][std].extend([x[2] for x in tmpres])
                best = float('inf')
                tmp = []
                for i in range(len(tmpres)):
                    if tmpres[i][0][idx] < best:
                        best = tmpres[i][0][idx]
                        tmp = tmpres[i]
                fres[k][std].append(tmp)
    ts_res = {}
    for k in tres:
        ts_res[k] = {}
        for kk in tres[k]:
            tmp = np.array(tres[k][kk])
            ts_res[k][kk] = [float(tmp.mean()), float(tmp.std())]
    df_ts = pd.DataFrame(ts_res)
    df_ts.to_csv('figs/'+name+'_time.csv', sep='\t')
    
    plotXs = {}
    for k in keys:
        plotXs[k] = fres[k][1][0][1]

    # Distribution plot of recoverted entries
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsmath}'})
    plt.rcParams.update({'lines.linewidth': 10})
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'legend.frameon': False})

    fig, axes = plt.subplots(2, 3)
    fig.set_size_inches(16, 10)

    l = 6
    ax = axes[0][0]
    X = sim.A
    sns.histplot(X.reshape(-1), ax=ax)
    ax.set_xlim(-l, l)
    ax.set_title('True entries')

    X = np.array(plotXs['si'])
    ax = axes[0][1]
    sns.histplot(X.reshape(-1), ax=ax)
    ax.set_xlim(-l, l)
    ax.set_title('Soft Impute')

    X = plotXs['maxnorm']  # - Zmax.mean()
    ax = axes[0][2]
    sns.histplot(X.reshape(-1), ax=ax)
    ax.set_xlim(-l, l)
    ax.set_title('Max Norm')

    X = plotXs['mfw']
    ax = axes[1][0]
    sns.histplot(X.reshape(-1), ax=ax)
    ax.set_xlim(-l, l)
    ax.set_title('Model Free Weighting')

    X = plotXs['snn']
    ax = axes[1][1]
    sns.histplot(X.reshape(-1), ax=ax)
    ax.set_xlim(-l, l)
    ax.set_title('Synthetic NN')

    X = plotXs['pseudo']
    ax = axes[1][2]
    sns.histplot(X.reshape(-1), ax=ax)
    ax.set_title('Pseudolikelihood')

    fig.tight_layout()
    fig.savefig('figs/'+name+'_sim_dist.pdf')
    
    # Fetch metrics
    RMSE = {}
    MAE = {}
    Xs = {}
    RMSE_std = {}
    MAE_std = {}
    for k in keys:
        RMSE[k] = []
        RMSE_std[k] = []
        MAE[k] = []
        MAE_std[k] = []
        Xs[k] = {}
        for std in params:
            RMSE[k].append(np.mean([fres[k][std][i][0][2] for i in range(len(fres[k][std]))]))
            RMSE_std[k].append(np.std([fres[k][std][i][0][2] for i in range(len(fres[k][std]))]))
            MAE[k].append(np.mean([fres[k][std][i][0][3] for i in range(len(fres[k][std]))]))
            MAE_std[k].append(np.std([fres[k][std][i][0][3] for i in range(len(fres[k][std]))]))
            Xs[k][std] = [fres[k][std][i][1] for i in range(len(fres[k][std]))]

    # RMSE plot
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsmath}'})
    plt.rcParams.update({'lines.linewidth': 5})
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'legend.frameon': True})
    
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    cnt = -0.02
    figs_name = {'snn':'SNN','mfw':'MFW', 'si':'SoftImpute',
        'maxnorm':'CZ','pseudo':'Ours'}
    for k in ['snn','mfw','maxnorm','si','pseudo']:
        label = figs_name[k]
        ax.errorbar([x+cnt for x in params], [x for x in RMSE[k]], [x/np.sqrt(9) for x in RMSE_std[k]], 
                    alpha=.5, label = label, linewidth=3,
                    fmt='o:', capsize=5)
        cnt += 0.01
    ax.legend()
    ax.set_ylim(-0.1,2.8)
    ax.set_title('TRMSE for different variances', fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    fig.tight_layout()
    fig.savefig('figs/'+name+'_sim_rmse.pdf')
    
    # MAE plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    cnt = -0.02
    for k in ['snn','mfw','maxnorm','si','pseudo']:
        label = figs_name[k]
        ax.errorbar([x+cnt for x in params], [x for x in MAE[k]], [x/np.sqrt(9) for x in MAE_std[k]],
                    alpha=.5, label = label, linewidth=3,
                    fmt='o:', capsize=5)
        cnt += 0.01
    ax.legend()
    ax.set_ylim(-0.1,2.1)
    ax.set_title('TMAE for different variances', fontsize=30)
    ax.tick_params(axis='both', labelsize=25)
    fig.tight_layout()
    fig.savefig('figs/'+name+'_sim_mae.pdf')
    
    # Generate metric tables
    RMSE = {}
    MAE = {}
    Xs = {}
    RMSE_std = {}
    MAE_std = {}
    for k in keys:
        Xs[k] = {}
        for std in params:
            RMSE[k] = np.mean([fres[k][std][i][0][2] for i in range(len(fres[k][std]))])
            RMSE_std[k] = (np.std([fres[k][std][i][0][2] for i in range(len(fres[k][std]))]))
            MAE[k] = (np.mean([fres[k][std][i][0][3] for i in range(len(fres[k][std]))]))
            MAE_std[k] = (np.std([fres[k][std][i][0][3] for i in range(len(fres[k][std]))]))
            Xs[k][std] = [fres[k][std][i][1] for i in range(len(fres[k][std]))]
            
    df = pd.DataFrame([RMSE,RMSE_std,MAE, MAE_std], index=['RMSE', 'RMSE_std', 'MAE', 'MAE_std']).T
    df.to_csv('figs/'+name+'_sim_metrics.csv', sep='\t')
    
if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import multiprocessing as mp
from scipy.io import loadmat # type: ignore
from stress import calc_stress
from QA import quantum_annealing

# ==============================
# Load data
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(
    os.path.join(BASE_DIR, "Element_conditions.xlsx"),
    skiprows=2,
    header=None    
)

strike = df.iloc[372:, 5].to_numpy(dtype=float)
dip = df.iloc[372:, 6].to_numpy(dtype=float)
direction = df.iloc[372:, 21].to_numpy(dtype=float)
rake = np.zeros(strike.size)

mat = loadmat("DC3DE.mat")
DC = mat["DC3DE"]
Sxx = DC[372:, 8]
Syy = DC[372:, 9]
# Szz = DC[372:,10]
# Syz = DC[372:,11]
# Sxz = DC[372:,12]
Sxy = DC[372:,13]
n = Sxx.size

stress_eq = np.vstack([Sxx, Syy, np.zeros(n), np.zeros(n), np.zeros(n), Sxy])

lamb_list = np.logspace(np.log10(0.05), np.log10(3.0), 25)

# ==============================
# objective
# ==============================
def make_objective(strike, dip, rake, direction, stress_eq):

    def objective(sigma):
        ss = np.vstack([
            stress_eq[0] + sigma[0, 0],
            stress_eq[1] + sigma[1, 1],
            stress_eq[2],
            stress_eq[3], 
            stress_eq[4],
            stress_eq[5] + sigma[0, 1]
        ])

        shear, normal = calc_stress(strike, dip, rake, ss)

        value = np.zeros_like(shear)
        value[shear > 0] = 1
        value[shear < 0] = -1

        score = np.sum(value == direction)
        
        return score, shear, normal, value

    return objective


objective = make_objective(
    strike, dip, rake, direction, stress_eq
)

# ==============================
#   I2
# ==============================
def stress_I2(sxx,syy,sxy):
    # """
    # sigma: array-like, shape (2,2)
    # """
    # sxx = df["sigma11"].values
    # syy = df["sigma22"].values
    # sxy = df["sigma12"].values

    I2 = sxx * syy -  sxy**2

    return I2

# ==============================
#   J2
# ==============================
def stress_J2(sxx,syy,sxy):

    J2 = 0.25 * (sxx - syy)**2 + sxy**2

    return J2

# ==============================
# QA parameters
# ==============================
QA_PARAMS = dict(
    T0=50.0,
    alpha_T=0.999,
    Gamma0=1.5,
    alpha_G=0.999,
    max_iter=5000,
    P=8,
    boundary=150,
    verbose=False
)

N_REPEAT = 10
N_WORKERS = 6

# ==============================
# single QA （For paralle）
# ==============================
def run_single_QA(args):
    lamb, seed = args
    np.random.seed(seed)

    best_sigma, best_score, best_extra, history, count = quantum_annealing(
    objective,
    lamb=lamb,
    **QA_PARAMS
    )

    print(f"best score = {best_score}\n")
    print(f"best sigma = \n")
    print(best_sigma)

 
    last = history[-1]

    sxx = last["sigma11"] / 10 # convert to MPa
    syy = last["sigma22"] / 10
    sxy = last["sigma12"] / 10

    return {
        "lambda": lamb,
        "sigma11": sxx,
        "sigma22": syy,
        "sigma12": sxy,
        "score": last["score"],
        "best_sigma11": best_sigma[0,0] / 10,
        "best_sigma22": best_sigma[1,1] / 10,
        "best_sigma12": best_sigma[0,1] / 10,
        "best_score": best_score,
        "history": history
    }

# ==============================
# main
# ==============================
if __name__ == "__main__":

    ctx = mp.get_context("spawn")

    lamb_vals = []
    score_vals = []
    I2_vals = []
    J2_vals = []

    last_results = []
    avg_results = []
    best_results = []

    for lamb in lamb_list:
        print(f"\n=== Lambda = {lamb:.3f} ===")

        seeds = np.random.randint(0, 1_000_000, size=N_REPEAT)
        args = [(lamb, seed) for seed in seeds]

        with ctx.Pool(processes=N_WORKERS) as pool:
            results = pool.map(run_single_QA, args)

        # ========= average =========
        sxx_m = np.mean([r["sigma11"] for r in results])
        syy_m = np.mean([r["sigma22"] for r in results])
        sxy_m = np.mean([r["sigma12"] for r in results])
        score_m = np.mean([r["score"] for r in results])
        I2_m = stress_I2(sxx_m, syy_m, sxy_m)
        J2_m = stress_J2(sxx_m, syy_m, sxy_m)

        lamb_vals.append(lamb)
        score_vals.append(score_m)
        I2_vals.append(I2_m)
        J2_vals.append(J2_m)

        avg_results.append({
            "lambda": lamb,
            "sigma11": sxx_m,
            "sigma22": syy_m,
            "sigma12": sxy_m,
            "score": score_m,
            "I2": I2_m,
            "J2": J2_m
        })

        # ========= Save the final results of each lambda） =========
        for i, r in enumerate(results):
            last_results.append({
            "lambda": lamb,
            "repeat_id": i,
            "sigma11": r["sigma11"],
            "sigma22": r["sigma22"],
            "sigma12": r["sigma12"],
            "score": r["score"]
        })
        
        # ========= Save the best results of 10 independent runs for each lambda =========
        for i, r in enumerate(results):
            best_results.append({
            "lambda": lamb,
            "repeat_id": i,
            "best_sigma11": r["best_sigma11"],
            "best_sigma22": r["best_sigma22"],
            "best_sigma12": r["best_sigma12"],
            "best_score": r["best_score"]
        })

        # for r in results:
        #     all_history.append({
        #         "lambda": lamb,
        #         "history": r["history"]
        #     })

        print(f" Mean score = {score_m:.2f}, Mean I2 = {I2_m:.3f}, Mean J2 = {J2_m:.3f}")
        print(f"Mean sigma = {sxx_m},{sxy_m};{sxy_m},{syy_m}\n")

 
    # ==============================
    # Save result_QA.csv
    # ==============================
    avg_df = pd.DataFrame(avg_results)
    avg_df.to_csv("result_QA_parallel_mean10.csv", index=False)

    last_df = pd.DataFrame(last_results)
    last_df.to_csv("last_result_QA_parallel.csv", index=False)

    best_df = pd.DataFrame(best_results)
    best_df.to_csv("best_result_QA_parallel.csv", index=False)
    # np.save("QA_history_parallel", all_history, allow_pickle=True)

    # # ==============================
    # # lambd - score  trade-off
    # # ==============================
    plt.figure(figsize=(8, 6))
    plt.semilogx(lamb_vals, score_vals, "-o", lw=2)
    # plt.plot(lamb_vals, score_vals, "-o")
    # plt.xscale("log")

    plt.xlabel(r" Weight $\lambda$")
    plt.ylabel("Mean matched score  (last accepted)")
    plt.title("lambda and fitting trade-off")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)

    # # ==============================
    # # lambda - J/I trade-off
    # # ==============================
    plt.figure(figsize=(8, 6))
    plt.semilogx(lamb_vals, I2_vals, "-*", lw=2)

    plt.xlabel(r" Weight $\lambda$")
    plt.ylabel(r" Mean$I_2/MPa$ (last accepted)")
    plt.title("Lambda and Stress trade-off")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)

    plt.figure(figsize=(8, 6))
    plt.semilogx(lamb_vals, J2_vals, "-*", lw=2)

    plt.xlabel(r" Weight $\lambda$")
    plt.ylabel(r" Mean$J_2/MPa$ (last accepted)")
    plt.title("Lambda and Stress trade-off")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
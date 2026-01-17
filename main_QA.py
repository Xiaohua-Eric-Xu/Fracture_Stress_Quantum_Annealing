import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
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
# QA parameters
# ==============================
best_scores = []
best_sigmas = []
all_history = []

for lamb in lamb_list:
    print(f"\n=== Running with lambda = {lamb:.3f} ===")

    best_sigma, best_score, best_extra, history, count = quantum_annealing(
    objective,
    T0=50.0,
    alpha_T=0.999,
    Gamma0=1.5,
    alpha_G=0.999,
    max_iter=5000,
    P=8,
    boundary = 150,
    lamb=lamb,
    verbose=True
    )

    best_scores.append(best_score)
    best_sigmas.append(best_sigma)

    all_history.append({
        "lambda": lamb,
        "history": history
    })

    print(f"Best score = {best_score}")
    print("Best sigma:")
    print(best_sigma)

# best_sigma, best_score, best_extra, history = quantum_annealing(
#     objective,
#     T0=80.0,
#     alpha_T=0.9993,
#     Gamma0=1.5,
#     alpha_G=0.999,
#     max_iter=5000,
#     P=12,
#     boundary = 300,
#     verbose=True
# )

# print("Best sigma:")
# print(best_sigma)
# print("Best score:", best_score)


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

    I2 = (sxx * syy -  sxy**2) / 100

    return I2


# ==============================
# Save result_QA.csv
# ==============================
df_list = []
lamb_vals = []
score_vals = []
I2_vals = []

output_path = os.path.join(BASE_DIR, "result_QA.csv")
with open(output_path, "w") as f:
    f.write(
        "lambda,""sigma11,sigma22,sigma12,I2"
        "score\n"
    )
    f.write(
        "(--),(MPa),(MPa),(MPa),(MPa),(--)\n"
    )

    for item in all_history:
        lamb = item["lambda"]
        hist = item["history"]

        df = pd.DataFrame(hist) 
        df["lambda"] = lamb
        df_list.append(df)

        last = hist[-1] #last accepted resolution

        sxx = last["sigma11"] / 10
        syy = last["sigma22"] /10
        sxy = last["sigma12"] /10
        score_last = last["score"]

        lamb_vals.append(lamb) # trade-off数据
        score_vals.append(score_last) 
        I2_vals.append(stress_I2(sxx, syy, sxy))

        #存result_QA.csv
        f.write(
            f"{lamb:.3f},"
            f"{sxx:.6f},"
            f"{syy:.6f},"
            f"{sxy:.6f},"
            f"{stress_I2(sxx, syy, sxy):.6f},"
            f"{int(score_last)}\n"
        )

np.save('QA_history', all_history, allow_pickle=True)
all_history_df = pd.concat(df_list, ignore_index=True)
all_history_df.to_csv("QA_all_lambda_history.csv", index=False)


# ==============================
# 3D scatter plot (σ cloud)
# ==============================
X = np.array([h["sigma11"] for h in history])
Y = np.array([h["sigma22"] for h in history])
Z = np.array([h["sigma12"] for h in history])
C = np.array([h["score"]   for h in history])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    X, Y, Z,
    c=C,
    s=10,
    cmap=plt.get_cmap("jet", 16),
    vmin = best_score-16+0.5,
    vmax = best_score+0.5
)

cb = plt.colorbar(sc, ax=ax)
cb.set_label("Matching number")

ax.set_xlabel(r"$\sigma_{11}$")
ax.set_ylabel(r"$\sigma_{22}$")
ax.set_zlabel(r"$\sigma_{12}$")
ax.set_title("Quantum Annealing: stress tensor search")
ax.legend()

plt.tight_layout()
plt.show(block=False)

# ==============================
# Optimal point plot
# ==============================
score   = np.array([h["score"]   for h in history])
mask = score == best_score
print(f"Number of points with score = {best_score}: {np.sum(mask)}")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X[mask], Y[mask], Z[mask],
    c="red",
    s=12,
    marker="o",
    edgecolors="k",
    label=f"Score = {best_score}"
)

ax.set_xlabel(r"$\sigma_{11}$")
ax.set_ylabel(r"$\sigma_{22}$")
ax.set_zlabel(r"$\sigma_{12}$")

ax.set_title(f"Best solutions in stress space (score = {best_score})")
ax.legend()

plt.tight_layout()
plt.show(block=False)

# # ==============================
# # lambd - score  trade-off
# # ==============================
plt.figure(figsize=(8, 6))
plt.semilogx(lamb_vals, score_vals, "-o", lw=2)
# plt.plot(lamb_vals, score_vals, "-o")
# plt.xscale("log")

plt.xlabel(r" Weight $\lambda$")
plt.ylabel("matched score  (last accepted)")
plt.title("lambda and fitting trade-off")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# # ==============================
# # lambda - J trade-off
# # ==============================
plt.figure(figsize=(8, 6))
plt.semilogx(lamb_vals, I2_vals, "-*", lw=2)

plt.xlabel(r" Weight $\lambda$")
plt.ylabel(r"$I_2/MPa$ (last accepted)")
plt.title("Lambda-Stress trade-off")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
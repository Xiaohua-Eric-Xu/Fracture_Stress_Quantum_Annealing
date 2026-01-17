# QA.py
import numpy as np

def quantum_annealing(
    objective_func,
    *,
    T0,
    alpha_T,
    Gamma0,
    alpha_G,
    max_iter,
    P,
    boundary,
    lamb,
    verbose=True
):
    """
    Quantum Annealing (Simulated Quantum Annealing)

    Returns
    -------
    best_sigma : (2,2)
    best_score : int
    best_extra : dict
    history : list of dict
        Each element contains:
        sigma11, sigma22, sigma12, score
    """

    # ---------------------------------
    # Initialize replicas
    # ---------------------------------
    sigmas = np.zeros((P, 2, 2))
    for k in range(P):
        eig = 1.0 + 50.0 * np.random.rand(2)
        V, _ = np.linalg.qr(np.random.randn(2, 2))
        sigmas[k] = V @ np.diag(eig) @ V.T

    T = T0
    Gamma = Gamma0

    best_score = -np.inf
    best_sigma = np.random.randn(2, 2) * 0
    #best_sigma[0,0] = 20
    best_extra = None
    count = 0
    history = []

    # lamb_test = 8e-1
    sigma = best_sigma
    print(best_sigma)
    # ---------------------------------
    # Main loop
    # ---------------------------------
    for it in range(max_iter):

        for k in range(P):

            sigma = sigmas[k]

            scale = max(np.linalg.norm(sigma), 1.0)
            perturb = (T / T0) * scale * np.random.randn(2, 2)
            perturb = 0.5 * (perturb + perturb.T)
            #print(perturb)
            sigma_new = sigma + perturb
            #sigma_new[0, 0] = sigma[0, 0]

            w, V = np.linalg.eigh(sigma_new)
            w = np.clip(w, -1 * boundary, boundary)
            sigma_new = V @ np.diag(w) @ V.T

            # objective
            old_score, *_ = objective_func(sigma)
            new_score, shear, normal, value = objective_func(sigma_new)

            kp = (k + 1) % P
            km = (k - 1) % P

            E_q_old = Gamma * (
                np.linalg.norm(sigmas[k] - sigmas[kp])**2 +
                np.linalg.norm(sigmas[k] - sigmas[km])**2
            )

            E_q_new = Gamma * (
                np.linalg.norm(sigma_new - sigmas[kp])**2 +
                np.linalg.norm(sigma_new - sigmas[km])**2
            )     

            # Esigma_old = np.sum(sigma**2) / boundary**2
            # Esigma_new = np.sum(sigma_new**2) / boundary**2

            #dE = old_score - new_score + E_q_new - E_q_old
            # if dE < 0 or np.random.rand() < np.exp(-dE / T) :
            
            de_score = (old_score - new_score) 
            de_Eq = (E_q_new - E_q_old) 
            #de_Esigma = lamb * (Esigma_new - Esigma_old)

            d_norm =  (np.sum(np.abs(sigma_new)) - np.sum(np.abs(sigma)))
            dE = de_score + de_Eq + lamb * d_norm
            
            if (dE <= 0 or np.random.rand() < np.exp(-dE / T)) :
                sigmas[k] = sigma_new

                history.append(dict(
                    sigma11=sigma_new[0, 0],
                    sigma22=sigma_new[1, 1],
                    sigma12=sigma_new[0, 1],
                    score=new_score
                ))

                count = count + 1
                if new_score > best_score:
                    best_score = new_score
                    best_sigma = sigma_new.copy()
                    best_extra = dict(
                        shear=shear.copy(),
                        normal=normal.copy(),
                        value=value.copy()
                    )

        T *= alpha_T
        Gamma *= alpha_G

        if verbose and it % 200 == 0:
            print(f"Iter {it:4d} | T={T:.3f} | Γ={Gamma:.3f} | New={new_score} | Norm = {np.sum(np.abs(sigma_new))}| Count={count} | Score={new_score-lamb*np.sum(np.abs(sigma_new))}" )
            print(f"sigma:")
            print(sigma_new)
            # print(f"delta_Score={de_score} | delta_Esigma={de_Esigma} delta_E_q={de_Eq} ")

    return best_sigma, best_score, best_extra, history, count
"""
Stage 1: FAITHFUL translation of the MATLAB single-trap X65 McNabb-Foster TDS-FEM code.
No optimization. Structure, loops and variable names follow the MATLAB file.
Python indices are 0-based; MATLAB's are 1-based (see comments marked [IDX]).
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# numpy >= 2.0 calls it trapezoid; older versions call it trapz
trapz = getattr(np, "trapezoid", None) or np.trapz

# --- 0. Import Experimental Data (I/O only; optional) -------------------------
data_path = r"C:\Users\rozer\Desktop\mtpfiles\x65tempvsdesorptionratetext2.txt"
try:
    exp_data = np.loadtxt(data_path)          # whitespace-separated 2 columns
    exp_temp_C = exp_data[:, 0]
    exp_rate_ppm = exp_data[:, 1]
except Exception as err:
    print("Experimental file not loaded (%s). Continuing without it." % err)
    exp_temp_C = None

# --- 1. Test & Material Parameters (X65 Steel) --------------------------------
phi = 100 / 3600
L = 0.0063
L_domain = L / 2
t_rest = 2700
T_min = 293.15
T_max = 873.15
R = 8.314
D_0 = 7.23e-8
E_L = 5690
C_L0 = 0.06
nu = 1e13

N_A = 6.022e23
N_L_raw = 5.1e29
N_L = N_L_raw / N_A

dH_traps = np.array([-80000.0])
NT_traps_raw = np.array([2.0e23])
NT_traps = NT_traps_raw / N_A
num_traps = len(dH_traps)

k_pen = 8e5
E_bc = 17100

# --- 2. Spatial & Time Discretization -----------------------------------------
nel = 25
nnode = nel + 1
h = L_domain / nel
dt = 0.5
x_nodes = np.linspace(0, L_domain, nnode)

# --- 3. Precompute Constant Matrices ------------------------------------------
M_e = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
K_unit_e = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])

M_global = sp.lil_matrix((nnode, nnode))
K_unit_global = sp.lil_matrix((nnode, nnode))
for e in range(nel):                       # [IDX] e = 0 ... nel-1
    nodes = slice(e, e + 2)                # [IDX] MATLAB [e, e+1] -> Python e:e+2 (end excluded)
    M_global[nodes, nodes] = M_global[nodes, nodes].toarray() + M_e
    K_unit_global[nodes, nodes] = K_unit_global[nodes, nodes].toarray() + K_unit_e
M_global = M_global.tocsr()
K_unit_global = K_unit_global.tocsr()

# --- 4. Initial Conditions (Oriani equilibrium) -------------------------------
C_L = np.ones(nnode) * C_L0
C_T = np.zeros((nnode, num_traps))
for i in range(num_traps):
    K_T = np.exp(-dH_traps[i] / (R * T_min))
    theta_L = C_L / N_L
    theta_T = (theta_L * K_T) / (1 + (K_T - 1) * theta_L)
    C_T[:, i] = theta_T * NT_traps[i]

# --- READ-ONLY DIAGNOSTIC (added from the 4-trap file; not part of solver) ----
def print_inventory(label, t_val, T_val, C_L_now, C_T_now):
    avg_L = trapz(C_L_now, x_nodes) / L_domain
    avg_T = [trapz(C_T_now[:, i], x_nodes) / L_domain for i in range(num_traps)]
    print("\n--- Hydrogen inventory: %s (t = %.1f s, T = %.2f K / %.2f C) ---"
          % (label, t_val, T_val, T_val - 273.15))
    print("lattice H : %.6e mol/m^3" % avg_L)
    for i in range(num_traps):
        print("trap %d H  : %.6e mol/m^3" % (i + 1, avg_T[i]))
    print("Total     : %.6e mol/m^3" % (avg_L + sum(avg_T)))
    return avg_L + sum(avg_T)

rest_report_done = False
print_inventory("Start of simulation", 0, T_min, C_L, C_T)

# --- 5. Time Integration Loop -------------------------------------------------
t = 0
t_final = t_rest + (T_max - T_min) / phi
T_out = []
flux_out = []
print("Running X65 Custom FEM Simulation (1-Trap System)...")
tic = time.perf_counter()

while t < t_final:
    t = t + dt
    if t <= t_rest:
        T = T_min
    else:
        T = T_min + phi * (t - t_rest)

    D_L = D_0 * np.exp(-E_L / (R * T))
    K_diff_global = D_L * K_unit_global

    C_L_old = C_L.copy()          # .copy() is ESSENTIAL: see teaching note 3
    C_T_old = C_T.copy()

    # Newton-Raphson
    for it in range(20):
        R_L = M_global @ (C_L - C_L_old) / dt + K_diff_global @ C_L
        R_T = np.zeros((nnode, num_traps))

        J_LL = (M_global / dt + K_diff_global).tolil()
        J_TT = [None] * num_traps
        J_LT = [None] * num_traps
        J_TL = [None] * num_traps

        for i in range(num_traps):
            R_T[:, i] = M_global @ (C_T[:, i] - C_T_old[:, i]) / dt
            J_TT[i] = (M_global / dt).tolil()
            J_LT[i] = sp.lil_matrix((nnode, nnode))
            J_TL[i] = sp.lil_matrix((nnode, nnode))

        for e in range(nel):
            nodes = slice(e, e + 2)
            cl_local = C_L[nodes]

            for i in range(num_traps):
                ct_local = C_T[nodes, i]
                p_i = nu * np.exp(-(E_L - dH_traps[i]) / (R * T))
                k_i = nu * np.exp(-E_L / (R * T))

                jtrap = (k_i * NT_traps[i] / N_L) * cl_local - p_i * ct_local \
                        - ((k_i - p_i) / N_L) * cl_local * ct_local
                d1 = (k_i * NT_traps[i] / N_L) - ((k_i - p_i) / N_L) * ct_local
                d2 = -p_i - ((k_i - p_i) / N_L) * cl_local

                F_trap_e = (h / 2.0) * jtrap
                dF_dCL = np.diag((h / 2.0) * d1)
                dF_dCT = np.diag((h / 2.0) * d2)

                R_L[nodes] = R_L[nodes] + F_trap_e
                R_T[nodes, i] = R_T[nodes, i] - F_trap_e

                J_LL[nodes, nodes] = J_LL[nodes, nodes].toarray() + dF_dCL
                J_LT[i][nodes, nodes] = J_LT[i][nodes, nodes].toarray() + dF_dCT
                J_TL[i][nodes, nodes] = J_TL[i][nodes, nodes].toarray() - dF_dCL
                J_TT[i][nodes, nodes] = J_TT[i][nodes, nodes].toarray() - dF_dCT

        bc_factor = (k_pen / N_L) * np.exp(-E_bc / (R * T))
        R_L[-1] = R_L[-1] + bc_factor * C_L[-1]        # [IDX] MATLAB end -> -1
        J_LL[nnode - 1, nnode - 1] = J_LL[nnode - 1, nnode - 1] + bc_factor

        # Jacobian and residual assembly for 1 trap (hard-coded, as in MATLAB)
        J_global = sp.bmat([[J_LL,    J_LT[0]],
                            [J_TL[0], J_TT[0]]], format="csc")
        R_global = np.concatenate([R_L, R_T[:, 0]])

        delta = -spsolve(J_global, R_global)

        C_L = C_L + delta[0:nnode]
        for i in range(num_traps):
            # [IDX] MATLAB (i*nnode+1 : (i+1)*nnode) with i=1..  ->  Python 0-based i:
            C_T[:, i] = C_T[:, i] + delta[(i + 1) * nnode:(i + 2) * nnode]

        if np.linalg.norm(delta) / (np.linalg.norm(np.concatenate([C_L, C_T.flatten(order="F")])) + 1e-12) < 1e-6:
            break

    if t % 2 < dt:
        current_flux = bc_factor * C_L[-1]
        T_out.append(T)
        flux_out.append(current_flux)

    if (not rest_report_done) and t >= t_rest:      # read-only diagnostic
        print_inventory("End of rest period / start of heating ramp", t, T, C_L, C_T)
        rest_report_done = True

runtime = time.perf_counter() - tic
print("Simulation Complete.  Runtime = %.1f s" % runtime)
H_final = print_inventory("End of heating ramp (final state)", t, T, C_L, C_T)

T_out = np.array(T_out)
flux_out = np.array(flux_out)

# --- 6. Unit Conversion (mol/m^2/s -> ppm/s) ----------------------------------
flux_ppm_s = (2.016 * flux_out) / (L * 7.847)

# --- 7. Plotting --------------------------------------------------------------
heating_idx = T_out > T_min
T_plot_C = T_out[heating_idx] - 273.15
sim_flux_plot = flux_ppm_s[heating_idx]

plt.figure(figsize=(9, 6))
plt.plot(T_plot_C, sim_flux_plot, "--b", linewidth=2.5,
         label="McNabb-Foster Simulation (1 Trap)")
if exp_temp_C is not None:
    plt.xlim(20, exp_temp_C.max() + 20)
plt.xlabel("Temperature [°C]", fontsize=12, fontweight="bold")
plt.ylabel("ΔC [ppm/s]", fontsize=12, fontweight="bold")
plt.title("TDS Spectrum: X65 Pipeline Steel (Single Trap Model)", fontsize=14)
plt.legend(loc="upper right")
plt.grid(True)

# --- READ-ONLY VERIFICATION QUANTITIES (from the 4-trap file) -----------------
T_heat = T_out[heating_idx]
flux_heat = flux_ppm_s[heating_idx]
idx_peak = np.argmax(flux_heat)
J_max = flux_heat[idx_peak]
T_peak_C = T_heat[idx_peak] - 273.15
H_desorbed = trapz(flux_heat, T_heat) / phi

print("\n========================================")
print("NUMERICAL RESULTS")
print("========================================")
print("nel              = %d" % nel)
print("dt               = %.5f s" % dt)
print("J_max            = %.8e ppm/s" % J_max)
print("T_peak           = %.4f degC" % T_peak_C)
print("H_desorbed       = %.8e ppm" % H_desorbed)
print("Final H inventory= %.8e mol/m^3" % H_final)
print("========================================")

np.savetxt("python_tds_stage1.csv", np.column_stack([T_out, flux_ppm_s]),
           delimiter=",", header="T_K,flux_ppm_per_s", comments="")
np.savetxt("python_final_conc_stage1.csv", np.column_stack([C_L, C_T]),
           delimiter=",", header="C_L,C_T1", comments="")
plt.show()
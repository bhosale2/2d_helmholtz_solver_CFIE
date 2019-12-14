from quad import QuadratureInfo
import matplotlib.pyplot as plt
import numpy as np
from plotset import plotset
import numpy.linalg as la
from kernels import eval_sp_dp_QBX, eval_target, bvp, sommerfeld
from kernels import Images_Integral
from scipy import stats

plotset()
xs = -2
ys = 2
k = 10.2
alpha = k  # CFIE parameter
beta = 2.04
interval = 15
m = int(np.floor(np.log(k / ys) / np.log(2)))
(sp, dp, qbx_exp_slp, qbx_exp_dlp, img_sp, img_dp, _, _) = eval_sp_dp_QBX(4, k)
som_sp, som_dp = sommerfeld(k, beta, interval, "far")
imgs = Images_Integral(m, beta, img_sp, img_dp)
n_panels = np.arange(10, 50, 5)
convg_err = []
test_targets = np.array([[2, 4], [0, 5], [-2, 4]]).T
for n in n_panels:
    domain = QuadratureInfo(n, show_bound=0)
    # finding the source response and solving bvp
    inc = sp(domain.curve_nodes[0], domain.curve_nodes[1], xs, ys)
    som_rhs = som_sp(domain.curve_nodes[0], domain.curve_nodes[1], xs, ys)
    imag_sp_rhs, _ = imgs.eval_integral(domain.curve_nodes, (xs, ys), 0.0, 0.0)
    rhs = -(inc + som_rhs + imag_sp_rhs)
    soln1 = bvp(
        n,
        k,
        domain,
        alpha,
        qbx_exp_slp,
        qbx_exp_dlp,
        rhs,
        som_sp=som_sp,
        som_dp=som_dp,
        imgs=imgs,
    )

    # evaluating at target locations
    num_test = eval_target(
        test_targets,
        domain.curve_nodes,
        domain.curve_weights,
        domain.normals,
        soln1,
        sp,
        dp,
        alpha,
        som_sp=som_sp,
        som_dp=som_dp,
        imgs=imgs,
    )
    convg_err.append(num_test)
convg = np.array((convg_err))
num_fine = convg[-1]
convg = convg[:-1] - num_fine
convg_norm = []
for x in convg:
    convg_norm.append(la.norm(x) / la.norm(num_fine))
plt.figure(1)
plt.loglog(n_panels[:-1], np.array(convg_norm), "-o", label="QBX4")
plt.loglog(
    n_panels[:-1], 5e4 * np.power(n_panels[:-1].astype(float), -5), label="$N^{-5}$",
)
plt.loglog(
    n_panels[:-1], 2e3 * np.power(n_panels[:-1].astype(float), -4), label="$N^{-4}$"
)
title = "Half space scattering: target evaluation convergence"
plt.xlabel("Number of panels", size=25)
plt.ylabel("$|u - u_e| / |u_e|$", size=25)
plt.title(title, size=25)
plt.legend(loc="upper right", ncol=1, frameon=True)
plt.show()
slope, _, _, _, _ = stats.linregress(np.log(n_panels[:-1]), np.log(np.array(convg_norm)))
print(slope)

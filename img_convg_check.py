import matplotlib.pyplot as plt
import numpy as np
from plotset import plotset
import numpy.linalg as la
from kernels import eval_sp_dp_QBX
from kernels import Images_Integral

plotset()
xs = -2
ys = 0.1
k = 10
alpha = k  # CFIE parameter
beta = 2
(_, _, _, _, img_sp, img_dp, _, _) = eval_sp_dp_QBX(4, k)
convg_err = []
test_targets = np.array([[-2, 0.01], [0, 5], [-2, 4]]).T

C = 1
m = int(np.floor(np.log(k / ys * C) / np.log(2)))
for x in range(1, m + 1):
    imgs = Images_Integral(x, beta, img_sp, img_dp)
    num_test, _ = imgs.eval_integral(test_targets, (xs, ys), 0.0, 0.0)
    convg_err.append(num_test)
    print(x)
    print(la.norm(num_test))

convg = np.array((convg_err))
num_fine = convg[-1]
convg = convg[:-1] - num_fine
convg_norm = []
for x in convg:
    convg_norm.append(la.norm(x) / la.norm(num_fine))
plt.figure(1)
plt.loglog(np.arange(1, m), np.array(convg_norm), "-o", label="ys")
title = "Half space scattering: image evaluation convergence"
plt.xlabel("Number of intervals", size=25)
plt.ylabel("$|u - u_e| / |u_e|$", size=25)
plt.title(title, size=25)
plt.legend(loc="upper right", ncol=1, frameon=True)
plt.show()

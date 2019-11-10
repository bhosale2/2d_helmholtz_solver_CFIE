import matplotlib.pyplot as plt
import numpy as np
# import numpy.linalg as la
from kernels import eval_sp_dp_QBX, sommerfeld, Images_Integral

plt.gca().set_aspect("equal")


k = 10
alpha = k  # CFIE parameter
beta = 2
interval = 20
xs = 0
ys = 5
m = int(np.ceil(np.log(k / ys) / np.log(2)))
(sp, dp, qbx_exp_slp, qbx_exp_dlp, img_sp,
 img_dp, _, _) = eval_sp_dp_QBX(4, k)
som_sp, som_dp = sommerfeld(k, beta, interval, "far")
imgs = Images_Integral(m, beta, img_sp, img_dp)

tg_size = 101
x = np.linspace(-5, 5, tg_size)
y = np.linspace(0, 10, tg_size)
X, Y = np.meshgrid(x, y)
test_targets = np.array((X.reshape(-1), Y.reshape(-1)))
exact_test = sp(test_targets[0], test_targets[1], xs, ys).reshape(tg_size, -1)
som_test = som_sp(test_targets[0], test_targets[1], xs,
                  ys).reshape(tg_size, -1)
imag_sp, imag_dp = imgs.eval_integral(test_targets, (xs, ys), 0.0, 0.0)
imag_sp = imag_sp.reshape(tg_size, -1)
v = np.linspace(-.15, 0.15, 100, endpoint=True)
plt.figure(1)
plt.contourf(X, Y, np.real(exact_test), v,
             cmap='twilight')
plt.colorbar()
plt.figure(2)
plt.contourf(X, Y, np.real(exact_test + som_test + imag_sp), v,
             cmap='twilight')
plt.colorbar()
plt.show()

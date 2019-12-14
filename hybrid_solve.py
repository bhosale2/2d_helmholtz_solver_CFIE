from quad import QuadratureInfo
import matplotlib.pyplot as plt
import numpy as np

# import numpy.linalg as la
from kernels import eval_sp_dp_QBX, eval_target, bvp, sommerfeld
from kernels import Images_Integral

plt.gca().set_aspect("equal")


xs = -2
ys = 2
k = 10.2
alpha = k  # CFIE parameter
beta = 2.04
interval = 15
C = 1
m = int(np.floor(np.log(k / ys * C) / np.log(2)))
(sp, dp, qbx_exp_slp, qbx_exp_dlp, img_sp, img_dp, _, _) = eval_sp_dp_QBX(4, k)
npanels = 30
domain = QuadratureInfo(npanels, show_bound=1)
som_sp, som_dp = sommerfeld(k, beta, interval, "far")
imgs = Images_Integral(m, beta, img_sp, img_dp)

# finding the source response and solving bvp
inc = sp(domain.curve_nodes[0], domain.curve_nodes[1], xs, ys)
som_rhs = som_sp(domain.curve_nodes[0], domain.curve_nodes[1], xs, ys)
imag_sp_rhs, _ = imgs.eval_integral(domain.curve_nodes, (xs, ys), 0.0, 0.0)
rhs = -(inc + som_rhs + imag_sp_rhs)
soln1 = bvp(
    npanels,
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
tg_size = 51
x = np.linspace(-4, 4, tg_size)
y = np.linspace(0, 8, tg_size)
X, Y = np.meshgrid(x, y)
test_targets = np.array((X.reshape(-1), Y.reshape(-1)))
exact_test = (sp(test_targets[0], test_targets[1], xs, ys)).reshape(tg_size, -1)
som_test = som_sp(test_targets[0], test_targets[1], xs, ys).reshape(tg_size, -1)
imag_sp_test, _ = imgs.eval_integral(test_targets, (xs, ys), 0.0, 0.0)
exact_test += som_test + imag_sp_test.reshape(tg_size, -1)
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
).reshape(tg_size, -1)
v = np.linspace(-0.15, 0.15, 200, endpoint=True)
show_targets = np.array([[2, 4], [0, 5], [-2, 4]]).T
plt.figure(1)
plt.contourf(X, Y, np.real(exact_test), v, cmap="twilight")
plt.xlabel("X", size=10)
plt.ylabel("Y", size=10)
plt.colorbar()
plt.figure(2)
plt.contourf(X, Y, np.real(num_test), v, cmap="twilight")
plt.xlabel("X", size=10)
plt.ylabel("Y", size=10)
plt.fill(domain.curve_nodes[0], domain.curve_nodes[1], "w")
plt.colorbar()
plt.figure(3)
plt.contourf(X, Y, np.real(exact_test + num_test), v, cmap="twilight")
plt.xlabel("X", size=10)
plt.ylabel("Y", size=10)
plt.fill(domain.curve_nodes[0], domain.curve_nodes[1], "w")
plt.plot(show_targets[0], show_targets[1], "go")
plt.colorbar()
plt.show()

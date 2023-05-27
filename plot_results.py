import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


EXPORT_DIR_PATH = 'data/'

tn = np.loadtxt(f'{EXPORT_DIR_PATH}tn.csv', delimiter=',')
xyzn = np.loadtxt(f'{EXPORT_DIR_PATH}xyzn.csv', delimiter=';')
# yn = np.loadtxt(f'{EXPORT_DIR_PATH}yn.csv', delimiter=',')
# zn = np.loadtxt(f'{EXPORT_DIR_PATH}zn.csv', delimiter=',')
xyzn = xyzn.T

# mpl.use('module://backend_interagg')
# print(mpl.get_backend())
mpl.rcParams['legend.fontsize'] = 10

ax = plt.figure().add_subplot(projection='3d')
ax.plot(xyzn[0], xyzn[1], xyzn[2], lw=0.5, label='parametric curve')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Vallis Attractor")
ax.legend()

plt.savefig(f'{EXPORT_DIR_PATH}fig1.png')
plt.show()

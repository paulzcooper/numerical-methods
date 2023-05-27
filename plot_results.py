import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


EXPORT_DIR_PATH = 'data/'

tn = np.loadtxt(f'{EXPORT_DIR_PATH}tn.csv', delimiter=',')
xn = np.loadtxt(f'{EXPORT_DIR_PATH}xn.csv', delimiter=',')
yn = np.loadtxt(f'{EXPORT_DIR_PATH}yn.csv', delimiter=',')
zn = np.loadtxt(f'{EXPORT_DIR_PATH}zn.csv', delimiter=',')

# mpl.use('module://backend_interagg')
# print(mpl.get_backend())
mpl.rcParams['legend.fontsize'] = 10

ax = plt.figure().add_subplot(projection='3d')
ax.plot(xn, yn, zn, lw=0.5, label='parametric curve')
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Vallis Attractor")
ax.legend()

plt.savefig(f'{EXPORT_DIR_PATH}fig2.png')
plt.show()

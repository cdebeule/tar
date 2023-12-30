import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

# use LaTeX in plots
mpl.rcParams['text.usetex'] = True
plt.rc('text.latex')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

path_figs = f'figures'
path_data = f'data'

## Make plots for Bernal bilayer graphene ##

u = 0.2
Delta, phi = 0.004, 1.0
Lx, Ly, w = 100, 99, 2
# armchair
filename = f'{path_data}/bilayer_armchair_Lx_{Lx}_Ly_{Ly}_w_{w}_u_{u}_Delta_{Delta}_phi_{phi}_.hdf5'
print('armchair bilayer graphene junction')

with h5py.File(filename, 'r') as f:
    keys = ['fermi_energy', 'bias', 'transmission']
    mulis, xdata0, ydata = [list(f[key]) for key in keys]

ne, npar = len(xdata0), len(mulis)
# create list with both negative and positive bias
xdata1 = [-x for x in xdata0[:0:-1]] + xdata0

print('-'*80, '\nmake plots')

g11 = []
for i in range(npar):
    g11.append([2 * item[1] + item[4] + item[5] for item in ydata[i][:0:-1]] + [2 * item[0] + item[2] + item[3] for item in ydata[i]])
    plt.plot(xdata1, g11[i], label = f'$\\mu/t = {mulis[i]}$')

plt.title(f'$\\Delta_0/t = {Delta} \\quad (L_x,L_y) = ({Lx},{Ly})$')
plt.xlabel('$eV_1/\\Delta_0$')
plt.ylabel('$G_{11}~[e^2/h]$')
plt.legend()
# plt.savefig(f'{path_figs}/example_Delta_{Delta}_phi_{phi}_.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

dg11 = []
for i in range(npar):
    dg11.append([2 * ( item[0] - item[1] ) + item[2] - item[4] + item[3] - item[5] for item in ydata[i]])  
    plt.plot(xdata0, dg11[i], label = f'$\\mu/t = {mulis[i]}$')

plt.title(f'$\\Delta_0/t = {Delta} \\quad (L_x,L_y) = ({Lx},{Ly})$')
plt.xlabel('$eV_1/\\Delta_0$', fontsize=16)
plt.ylabel('$\\delta G_{11}(V_1)$', fontsize=16)
plt.legend()
plt.show()
plt.close()

g21 = []
for i in range(npar):
    g21.append([item[4] - item[5] for item in ydata[i][:0:-1]] + [item[2] - item[3] for item in ydata[i]])
    plt.plot(xdata1, g21[i], label = f'$\\mu/t = {mulis[i]}$')

plt.title(f'$\\Delta_0/t = {Delta} \\quad (L_x,L_y) = ({Lx},{Ly})$')
plt.xlabel('$eV_1/\\Delta_0$')
plt.ylabel('$G_{21}~[e^2/h]$')
plt.legend()
plt.show()
plt.close()

dg21= []
for i in range(npar):
    dg21.append([item[2] - item[4] + item[5] - item[3] for item in ydata[i]])  
    plt.plot(xdata0, dg21[i], label = f'$\\mu/t = {mulis[i]}$')

plt.title(f'$\\Delta_0/t = {Delta} \\quad (L_x,L_y) = ({Lx},{Ly})$')
plt.xlabel('$eV_1/\\Delta_0$')
plt.ylabel('$\\delta G_{21}(V_1)$')
plt.legend()
plt.show()
plt.close()

print('-'*80, '\nfinished')
print('-'*80)
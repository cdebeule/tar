import kwant
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends
import matplotlib as mpl
import h5py
import time
import os

### Overview ###

# File : toy_pointcontact.py
# Authors : Christophe De Beule and Pok Man Tam
# Contact : christophe.debeule@gmail.com

# Ancillary file for arXiv:2302.14050 "Topological Andreev Rectification" [URL: https://arxiv.org/abs/2302.14050]
# Published version: 
# Pok Man Tam, Christophe De Beule, and Charles L. Kane, Topological Andreev Rectification, PRB ... (2023) [URL: ]

# This code pertains to the Andreev point contact for the toy model on the square lattice described
# in Section III of the paper, and reproduces Figure 11.

### Code ###

# create directories for figures and data
if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('data'):
    os.makedirs('data')
# paths
path_figs = f'figures'
path_data = f'data'

# define unit matrix and Pauli matrices
tau_0 = np.array([[1, 0], [0, 1]])
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])
# define constants
pi = np.pi

# create system
def make_system(**kwargs):
    keys = ['Nx','Ny', 'mu', 'tx', 'ty', 'Delta', 'phi', 'w', 'l']
    Nx, Ny, mu, tx, ty, Delta, phi, w, l = [kwargs.get(key) for key in keys]

    tau_phi = np.cos(pi * phi) * tau_x + np.sin(pi * phi) * tau_y
    e0, Lx, Ly = 2 * ( tx + ty ) - mu, 2 * Nx, 2 * Ny

    # define superconducting region
    def f(x):
        if x < -l / 2:
            return w / 2 + ( Ly - w ) * ( - 2 * x - l ) / ( 2 * ( Lx - l ) )
        elif x > l / 2:
            return w / 2 + ( Ly - w ) * ( 2 * x - l ) / ( 2 * ( Lx - l ) )
        else:
            return w / 2

    # function for onsite terms
    def onsite(site):
        x, y = site.pos
        if y >=  f(x):
            return e0 * tau_z + Delta * tau_x
        elif y <= -f(x):
            return e0 * tau_z + Delta * tau_phi
        else:
            return e0 * tau_z

    # square lattice with two orbitals
    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()

    # onsite
    syst[(lat(x, y) for x in range(-Nx, Nx+1) for y in range(-Ny, Ny+1))] = onsite
    # nearest-neighbor hopping
    syst[kwant.builder.HoppingKind((1, 0), lat)] = -tx * tau_z
    syst[kwant.builder.HoppingKind((0, 1), lat)] = -ty * tau_z
 
    # leads
    sym_left = kwant.TranslationalSymmetry((-1, 0))
    sym_right = kwant.TranslationalSymmetry((1, 0))
    sym_up = kwant.TranslationalSymmetry((0, 1))
    sym_down = kwant.TranslationalSymmetry((0, -1))
    # specify the conservation law used to treat electrons and holes separately
    lead0 = kwant.Builder(sym_left, conservation_law=-tau_z)
    lead1 = kwant.Builder(sym_right, conservation_law=-tau_z)
    lead2 = kwant.Builder(sym_up)
    lead3 = kwant.Builder(sym_down)

    # left lead
    lead0[(lat(0, y) for y in range(-Ny, Ny+1))] = e0 * tau_z
    lead0[kwant.builder.HoppingKind((1, 0), lat)] = -tx * tau_z
    lead0[kwant.builder.HoppingKind((0, 1), lat)] = -ty * tau_z
   
    # right lead
    lead1[(lat(0, y) for y in range(-Ny, Ny+1))] = e0 * tau_z
    lead1[kwant.builder.HoppingKind((1, 0), lat)] = -tx * tau_z
    lead1[kwant.builder.HoppingKind((0, 1), lat)] = -ty * tau_z
    
    # top lead
    lead2[(lat(x, 0) for x in range(-Nx, Nx+1))] = e0 * tau_z + Delta * tau_x
    lead2[kwant.builder.HoppingKind((1, 0), lat)] = -tx * tau_z
    lead2[kwant.builder.HoppingKind((0, 1), lat)] = -ty * tau_z

    # bottom lead
    lead3[(lat(x, 0) for x in range(-Nx, Nx+1))] = e0 * tau_z + Delta * tau_phi
    lead3[kwant.builder.HoppingKind((1, 0), lat)] = -tx * tau_z
    lead3[kwant.builder.HoppingKind((0, 1), lat)] = -ty * tau_z
   
    return syst, [lead0, lead1, lead2, lead3]

# visualize pair potential
def site_color(i):
    foo = fsyst.hamiltonian(i, i)[1, 0]
    if foo.real > 0:
        return 'darkorange'
    elif foo.imag > 0:
        return 'limegreen'
    else:
        return 'black'

# compute transmission functions
def transmission(syst, energies):
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
        Rhe = smatrix.transmission((0, 1), (0, 0))
        Reh = smatrix.transmission((0, 0), (0, 1))
        Tee = smatrix.transmission((1, 0), (0, 0))
        The = smatrix.transmission((1, 1), (0, 0))
        Thh = smatrix.transmission((1, 1), (0, 1))
        Teh = smatrix.transmission((1, 0), (0, 1))
        data.append([Rhe, Reh, Tee, The, Thh, Teh])
    return data

## input parameters ##

# 2*Nx + 1 : number of sites along x
# 2*Ny + 1 : number of sites along y
# Lx = 2*Nx : length of the scattering region
# Ly = 2*Ny : width of the scattering region
# w : width of the Andreev channel
# l : length of the Andreev channel
# tx : nearest-neighbor hopping in the x direction
# ty : nearest-neighbor hopping in the y direction
# mu : Fermi energy EF of the metal
# Delta : magnitude of the superconducting pair potential
# phi : phase difference between superconductors

# energy unit : tx
# length unit : lattice constant a

Nx, Ny = 50, 50                             
tx, ty = 1.0, 0.5
Delta, phi = 0.1, 1.0
mulis = [0.55, 1.05, 2.55, 3.05, 4.55, 5.05] # list of EF values

# figure 11
w, l = 2, 1

print(f'number of sites: {2 * Nx + 1} x {2 * Ny + 1}')
print(f'junction width: {w}')
print(f'junction length: {l}')
filename = f'toy_pointcontact_Nx_{Nx}_Ny_{Ny}_w_{w}_l_{l}_tx_{tx}_ty_{ty}_Delta_{Delta}_phi_{phi}_.hdf5'

## plot the system ##

print('-'*80, '\nplotting system - close window to start computation')

params = {'Nx': Nx, 'Ny': Ny, 'mu': 1.0, 'tx': tx, 'ty': ty, 'Delta': Delta, 'phi': phi, 'w': w, 'l': l}
# add leads and finalize system
syst, leads = make_system(**params)
[syst.attach_lead(leads[n]) for n in range(4)]
fsyst = syst.finalized()                        
fig, ax = plt.subplots(1, 1)
kwant.plot(fsyst, ax=ax, site_size=0.21, site_color=site_color, hop_lw=0.08)
ax.set_aspect('equal')
# ax.axis('off')
plt.xlabel(r'$x/a$')
plt.ylabel(r'$y/a$')
# plt.savefig(f'{path_figs}/toy_pointcontact_Nx_{Nx}_Ny_{Ny}_w_{w}_.png', dpi=1000, bbox_inches='tight', pad_inches=0.1)
plt.show()

## calculate the transmission functions ##

# the calculation is done only for positive bias
# results for negative bias are obtained from electron-hole symmetry

ne = 101                                # number of subgap bias values
xdata = np.linspace(0.0, 1.0, ne)[:-1]  # bias in units of Delta
ydata = []
energies = [i * Delta for i in xdata]   # bias in units of tx
print('-'*80, '\nstarting computation')

npar = len(mulis)
initial_time = time.time()
# loop over different mu
for i in range(npar):
    start_time = time.time()
    params = {'Nx': Nx, 'Ny': Ny, 'mu': mulis[i], 'tx': tx, 'ty': ty, 'Delta': Delta, 'phi': phi, 'w': w, 'l': l}
    # add leads and finalize system
    syst, leads = make_system(**params)
    [syst.attach_lead(leads[n]) for n in range(4)]
    fsyst = syst.finalized()
    foo = transmission(fsyst, energies)
    # save intermediate results to temporary files
    with h5py.File(f'{path_data}/temp_EF_{mulis[i]}_' + filename, 'w') as f:
        f.create_dataset('bias', data=xdata)
        f.create_dataset('transmission', data=foo)
    ydata.append(foo)
    print(f'\nfinished {i+1} out of {npar} in {round(time.time() - start_time, 1)} seconds')

print(f'\ncomputation finished. total computation time : {round(time.time() - initial_time, 1)} seconds')
print('-'*80, '\nwrite results to files')

## save data ##

with h5py.File(f'{path_data}/' + filename, 'w') as f:
    f.create_dataset('fermi_energy', data=mulis)
    f.create_dataset('bias', data=xdata)
    f.create_dataset('transmission', data=ydata)
# delete temporary files
for mu in mulis:
    os.remove(f'{path_data}/temp_EF_{mu}_' + filename)

print('-'*80, '\nfinished')
print('-'*80)
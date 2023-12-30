import kwant
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends
import matplotlib as mpl
import h5py
import time
import os

### Overview ###

# File : InAs_junction.py
# Authors : Christophe De Beule and Pok Man Tam
# Contact : christophe.debeule@gmail.com

# Ancillary file for arXiv:2302.14050 "Topological Andreev Rectification" [URL: https://arxiv.org/abs/2302.14050]
# Published version: 
# Pok Man Tam, Christophe De Beule, and Charles L. Kane, Topological Andreev Rectification, PRB ... (2023) [URL: ]

# This code pertains to the Andreev junction for the InAs quantum well in Section IV.A of the paper.

### Code ###

# create directories for figures and data
if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('data'):
    os.makedirs('data')
# paths
path_figs = f'figures'
path_data = f'data'

# define Nambu and spin matrices
t3s0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
t3s1 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]])
t3s2 = np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]])
t1s0 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
t2s0 = np.array([[0, 0, -1j, 0], [0, 0, 0, -1j], [1j, 0, 0, 0], [0, 1j, 0, 0]])
# t2s2 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
# define constants
pi = np.pi

# create system
def make_system(**kwargs):
    keys = ['Nx','Ny', 'muN', 'muS', 't', 'tR', 'Delta', 'phi', 'w']
    Nx, Ny, muN, muS, t, tR, Delta, phi, w = [kwargs.get(key) for key in keys]

    tau_phi = np.cos(pi * phi) * t1s0 + np.sin(pi * phi) * t2s0
    eN, eS = 4 * t - muN, 4 * t - muS

    # function for onsite terms
    def onsite(site):
        y = site.pos[1]
        if y >=  w / 2:
            return eS * t3s0 + Delta * t1s0
        elif y <= -w / 2:
            return eS * t3s0 + Delta * tau_phi
        else:
            return eN * t3s0
        
    # square lattice with four orbitals
    lat = kwant.lattice.square(norbs=4)
    syst = kwant.Builder()

    # onsite
    syst[(lat(x, y) for x in range(-Nx, Nx+1) for y in range(-Ny, Ny+1))] = onsite
    # nearest-neighbor hopping
    syst[kwant.builder.HoppingKind((1, 0), lat)] = - t * t3s0 - 1j * tR * t3s2
    syst[kwant.builder.HoppingKind((0, 1), lat)] = - t * t3s0 + 1j * tR * t3s1
 
    # leads
    sym_left = kwant.TranslationalSymmetry((-1, 0))
    sym_right = kwant.TranslationalSymmetry((1, 0))
    sym_up = kwant.TranslationalSymmetry((0, 1))
    sym_down = kwant.TranslationalSymmetry((0, -1))
    # specify the conservation law used to treat electrons and holes separately
    lead0 = kwant.Builder(sym_left, conservation_law=-t3s0)
    lead1 = kwant.Builder(sym_right, conservation_law=-t3s0)
    lead2 = kwant.Builder(sym_up)
    lead3 = kwant.Builder(sym_down)

    # left lead
    lead0[(lat(0, y) for y in range(-Ny, Ny+1))] = eN * t3s0
    lead0[kwant.builder.HoppingKind((1, 0), lat)] = - t * t3s0 - 1j * tR * t3s2
    lead0[kwant.builder.HoppingKind((0, 1), lat)] = - t * t3s0 + 1j * tR * t3s1
   
    # right lead
    lead1[(lat(0, y) for y in range(-Ny, Ny+1))] = eN * t3s0
    lead1[kwant.builder.HoppingKind((1, 0), lat)] = - t * t3s0 - 1j * tR * t3s2
    lead1[kwant.builder.HoppingKind((0, 1), lat)] = - t * t3s0 + 1j * tR * t3s1
    
    # top lead
    lead2[(lat(x, Ny+2) for x in range(-Nx, Nx+1))] = eS * t3s0 + Delta * t1s0
    lead2[kwant.builder.HoppingKind((1, 0), lat)] = - t * t3s0 - 1j * tR * t3s2
    lead2[kwant.builder.HoppingKind((0, 1), lat)] = - t * t3s0 + 1j * tR * t3s1

    # bottom lead
    lead3[(lat(x, -Ny-1) for x in range(-Nx, Nx+1))] = eS * t3s0 + Delta * tau_phi
    lead3[kwant.builder.HoppingKind((1, 0), lat)] = - t * t3s0 - 1j * tR * t3s2
    lead3[kwant.builder.HoppingKind((0, 1), lat)] = - t * t3s0 + 1j * tR * t3s1

    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    syst.attach_lead(lead2)
    syst.attach_lead(lead3)

    return syst

# visualize pair potential
def site_color(i):
    foo = fsyst.hamiltonian(i, i)[2, 0]
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
# Lx = 2*Nx : length of the scattering region/junction
# Ly = 2*Ny : width of the scattering region
# w : width of the Andreev channel
# a : lattice constant (10 nm)
# t : nearest-neighbor hopping
# tR : Rashba parameter \lambda_R
# muN : Fermi energy EF in normal region
# muS : Fermi energy EF in proximitized regions
# Delta : magnitude of the superconducting pair potential
# phi : phase difference between superconductors

# energy unit : meV
# length unit : lattice constant a

Nx = 100    # Lx = 2000 nm
# Nx = 150    # Lx = 3000 nm
# Nx = 200    # Lx = 4000 nm
# Nx = 300    # Lx = 6000 nm
Ny, w = 50, 10
t, tR = 15.0, 0.75
Delta, phi = 0.15, 1.0
muNlis = [10.0, 10.0]
muSlis = [10.0, 10.3]

print(f'number of sites: {2 * Nx + 1} x {2 * Ny + 1}')
a = 10 # nm
print(f'(Lx, Ly) = ({2 * Nx * a} nm, {2 * Ny * a} nm)', flush=True)
print(f'junction width: {w * a} nm', flush=True)
filename = f'InAs_junction_Nx_{Nx}_Ny_{Ny}_w_{w}_t_{t}_tR_{tR}_Delta_{Delta}_phi_{phi}_.hdf5'

## plot the system ##

print('-'*80, '\nplotting system - close window to start computation')

params = {'Nx': Nx, 'Ny': Ny, 'muN': 1.0, 'muS': 1.0, 't': t, 'tR': tR, 'Delta': Delta, 'phi': phi, 'w': w}
fsyst = make_system(**params).finalized()
fig, ax = plt.subplots(1, 1)
kwant.plot(fsyst, ax=ax, site_color=site_color, hop_lw=0.05)
ax.axis('equal')
# ax.axis('off')
plt.xlabel(r'$x/a$')
plt.ylabel(r'$y/a$')
# plt.savefig(f'{path_figs}/InAs_junction_Nx_{Nx}_Ny_{Ny}_w_{w}_.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
plt.show()

## calculate the transmission functions ##

# the calculation is done only for positive bias
# results for negative bias are obtained from electron-hole symmetry

ne = 11                                 # number of subgap bias values
xdata = np.linspace(0.0, 1.0, ne)[:-1]  # bias in units of Delta
ydata = []
energies = [i * Delta for i in xdata]   # bias in units of tx
print('-'*80, '\nstarting computation')

npar = len(muNlis)
initial_time = time.time()
# loop over different mu
for i in range(npar):
    start_time = time.time()
    params = {'Nx': Nx, 'Ny': Ny, 'muN': muNlis[i], 'muS': muSlis[i], 't': t, 'tR': tR, 'Delta': Delta, 'phi': phi, 'w': w}
    fsyst = make_system(**params).finalized()
    foo = transmission(fsyst, energies)
    # save intermediate results to temporary file
    with h5py.File(f'{path_data}/temp_i_{i}_' + filename, 'w') as f:
        f.create_dataset('bias', data=xdata)
        f.create_dataset('transmission', data=foo)
    ydata.append(foo)
    print(f'\nfinished {i+1} out of {npar} in {round(time.time() - start_time, 1)} seconds', flush=True)

print(f'\ncomputation finished. total computation time: {round(time.time() - initial_time, 1)} seconds')
print('-'*80, '\nwrite results to files')

## save data ##

with h5py.File(f'{path_data}/' + filename, 'w') as f:
    f.create_dataset('fermi_energy', data=[muNlis,muSlis])
    f.create_dataset('bias', data=xdata)
    f.create_dataset('transmission', data=ydata)
# delete temporary files
for i in range(npar):
    os.remove(f'{path_data}/temp_i_{i}_' + filename)

# print data
# print([muNlis,muSlis])
# for i in range(npar):
#     print([[item[2] - item[3] for item in ydata[i]], [item[4] - item[5] for item in ydata[i]]])

print('-'*80, '\nfinished')
print('-'*80)
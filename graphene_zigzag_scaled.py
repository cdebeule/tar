import kwant
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends
import matplotlib as mpl
import h5py
import time
import os

### Overview ###

# File : graphene_zigzag_scaled.py
# Authors : Christophe De Beule and Pok Man Tam
# Contact : christophe.debeule@gmail.com

# Ancillary file for arXiv:2302.14050 "Topological Andreev Rectification" [URL: https://arxiv.org/abs/2302.14050]
# Published version: 
# Pok Man Tam, Christophe De Beule, and Charles L. Kane, Topological Andreev Rectification, PRB ... (2023) [URL: ]

# This code pertains to the Andreev junction for monolayer graphene in Section IV.B.1 of the paper
# with the junction oriented along the zigzag direction.

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
    keys = ['Lx','Ly', 'mu', 't', 'Delta', 'phi', 'w']
    Lx, Ly, mu, t, Delta, phi, w = [kwargs.get(key) for key in keys]
    tau_phi = np.cos(pi * phi) * tau_x + np.sin(pi * phi) * tau_y

    # function for onsite terms
    def onsite(site):
        y = site.pos[1]
        if y >=  w / 2:
            return -mu * tau_z + Delta * tau_x
        elif y <= -w / 2:
            return -mu * tau_z + Delta * tau_phi
        else:
            return -mu * tau_z

    # rectangular scattering region
    def rectangle(pos):
        x, y = pos
        return -Lx / 2 <= x <= Lx / 2 and -Ly / 2 <= y <= Ly / 2
    
    # define honeycomb lattice
    # x axis : zigzag
    # y axis : armchair
    lat = kwant.lattice.general([(1, 0), (1 / 2, np.sqrt(3) / 2)], [(1 / 2, 1 / (2 * np.sqrt(3))), (1 / 2, -1 / (2 * np.sqrt(3)))], norbs=2)
    a, b = lat.sublattices
    syst = kwant.Builder()

    # onsite
    syst[lat.shape(rectangle, (0, 0))] = onsite
    # nearest-neighbor hopping
    syst[lat.neighbors()] = -t * tau_z
 
    # leads
    
    # In order to add leads with a larger translation period we follow the discussion found here:
    # https://kwant-discuss.kwant-project.narkive.com/sOepG1sx/kwant-symmetry-and-leads

    # First we specify a unit cell for the translational symmetry of the leads: 
    sym_left = kwant.TranslationalSymmetry((-1, 0))
    sym_right = kwant.TranslationalSymmetry((1, 0))
    sym_up = kwant.TranslationalSymmetry(lat.vec((-1, 2)))
    sym_down = kwant.TranslationalSymmetry(lat.vec((1, -2)))
    # Then we add a second (linear independent) lattice vector with other_vectors that
    # complements the translational vector above. Both vectors then form a basis.
    # Documentation: https://kwant-project.org/doc/1/reference/generated/kwant.lattice.TranslationalSymmetry
    sym_left.add_site_family(a, other_vectors=[(-1, 2)])
    sym_left.add_site_family(b, other_vectors=[(-1, 2)])
    sym_right.add_site_family(a, other_vectors=[(-1, 2)])
    sym_right.add_site_family(b, other_vectors=[(-1, 2)])

    # specify the conservation law used to treat electrons and holes separately 
    lead0 = kwant.Builder(sym_left, conservation_law=-tau_z)
    lead1 = kwant.Builder(sym_right, conservation_law=-tau_z)
    lead2 = kwant.Builder(sym_up)
    lead3 = kwant.Builder(sym_down)

    # shape of vertical leads
    def lead_shape_v(pos):
        y = pos[1]
        return -Ly / 2 <= y <= Ly / 2
    # shape of horizontal leads
    def lead_shape_h(pos):
        x = pos[0]
        return -Lx / 2 <= x <= Lx / 2

    # left lead
    lead0[lat.shape(lead_shape_v, (0, 0))] = -mu * tau_z
    lead0[lat.neighbors()] = -t * tau_z
   
    # right lead
    lead1[lat.shape(lead_shape_v, (0, 0))] = -mu * tau_z
    lead1[lat.neighbors()] = -t * tau_z
    
    # top lead
    lead2[lat.shape(lead_shape_h, (0, 0))] = -mu * tau_z + Delta * tau_x
    lead2[lat.neighbors()] = -t * tau_z

    # bottom lead
    lead3[lat.shape(lead_shape_h, (0, 0))] = -mu * tau_z + Delta * tau_phi
    lead3[lat.neighbors()] = -t * tau_z
   
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    syst.attach_lead(lead2)
    syst.attach_lead(lead3)

    return syst

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

# Lx : length of the scattering region/junction
# Ly : width of the scattering region
# w : width of the Andreev channel
# t : scaled nearest-neighbor hopping
# mu : Fermi energy EF
# Delta : magnitude of the superconducting pair potential
# phi : phase difference between superconductors

# energy unit : meV
# length unit : s*a
# s : scaling factor

# Delta = 0.2 for Al
# Delta = 1.0 for Nb or MoRe

s = 12 
t = 2800.0 / s
Delta, phi = 2.0, 1.0
mulis = [-10.0, 20.0]   
print(f'xi/sa = {np.sqrt(3) * t / ( 2 * pi * Delta)}', flush=True)

# small system (in the paper we take Lx = Ly = 1000 and w = 100)
Lx, Ly, w = 300, 300, 30
filename = f'graphene_zigzag_s_{s}_Lx_{Lx}_Ly_{Ly}_w_{w}_Delta_{Delta}_phi_{phi}_.hdf5'

# force horizontal zigzag edge
Ny = round((np.sqrt(3) * Ly + 1) / 3)
Ly = ( 3 * Ny - 1 ) / np.sqrt(3)
# force horizontal bearded edge
# Ny = round((np.sqrt(3) * Ly + 2) / 3)
# Ly = ( 3 * Ny - 2 ) / np.sqrt(3)
print(f'(Lx, Ly) = ({Lx}, {Ly})', flush=True)

## plot the system ##

print('-'*80, '\nplotting system - close window to start computation')

params = {'Lx': Lx, 'Ly': Ly, 'mu': 0.0, 't': t, 'Delta': Delta, 'phi': phi, 'w': w}
fsyst = make_system(**params).finalized()
fig, ax = plt.subplots(1, 1)
kwant.plot(fsyst, ax=ax, site_color=site_color, hop_lw=0.05)
ax.axis('equal')
# ax.axis('off')
plt.xlabel(r'$x/a$')
plt.ylabel(r'$y/a$')
# plt.savefig(f'{path_figs}/graphene_zigzag_s_{s}_Lx_{Lx}_Ly_{round(Ly)}_w_{w}_.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
plt.show()

## calculate the transmission functions ##

# the calculation is done only for positive bias
# results for negative bias are obtained from electron-hole symmetry

ne = 11                                 # number of subgap bias values
xdata = np.linspace(0.0, 1.0, ne)[:-1]  # bias in units of Delta
ydata = []
energies = [i * Delta for i in xdata]   # bias in meV
print('-'*80, '\nstarting computation', flush=True)

npar = len(mulis)
initial_time = time.time()
for i in range(npar):
    start_time = time.time()
    params = {'Lx': Lx, 'Ly': Ly, 'mu': mulis[i], 't': t, 'Delta': Delta, 'phi': phi, 'w': w}
    fsyst = make_system(**params).finalized()
    foo = transmission(fsyst, energies)
    # save intermediate results to temporary file
    with h5py.File(f'{path_data}/temp_EF_{mulis[i]}_' + filename, 'w') as f:
        f.create_dataset('bias', data=xdata)
        f.create_dataset('transmission', data=foo)
    ydata.append(foo)
    print(f'\nfinished {i+1} out of {npar} in {round(time.time() - start_time, 1)} seconds', flush=True)

print(f'\ncomputation finished. total computation time: {round(time.time() - initial_time, 1)} seconds')
print('-'*80, '\nwrite results to files')

## save data ##

with h5py.File(f'{path_data}/' + filename, 'w') as f:
    f.create_dataset('fermi_energy', data=mulis)
    f.create_dataset('bias', data=xdata)
    f.create_dataset('transmission', data=ydata)
# delete temporary files
for mu in mulis:
    os.remove(f'{path_data}/temp_EF_{mu}_' + filename)

# print data
# print(mulis)
# for i in range(npar):
#     print([[item[2] - item[3] for item in ydata[i]], [item[4] - item[5] for item in ydata[i]]])

print('-'*80, '\nfinished')
print('-'*80)
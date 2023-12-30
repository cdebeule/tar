import kwant
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends
import matplotlib as mpl
import h5py
import time
import os

### Overview ###

# File : bilayer_armchair.py
# Authors : Christophe De Beule and Pok Man Tam
# Contact : christophe.debeule@gmail.com

# Ancillary file for arXiv:2302.14050 "Topological Andreev Rectification" [URL: https://arxiv.org/abs/2302.14050]
# Published version: 
# Pok Man Tam, Christophe De Beule, and Charles L. Kane, Topological Andreev Rectification, PRB ... (2023) [URL: ]

# This code pertains to the Andreev junction for Bernal bilayer graphene in Section IV.B.2 of the paper
# with the junction oriented along the armchair direction.

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

# define bilayer graphene lattice
# x axis : armchair
# y axis : zigzag
lat = kwant.lattice.general([(0, 1), (np.sqrt(3) / 2, 1 / 2)],
                            [(0, 0), (-1 / np.sqrt(3), 0), (1 / np.sqrt(3), 0), (0, 0)], norbs=2)
a1, b1, a2, b2 = lat.sublattices

# create system
def make_system(**kwargs):
    keys = ['Lx','Ly', 'mu', 't', 'g1', 'g3', 'u', 'Delta', 'phi', 'w']
    Lx, Ly, mu, t, g1, g3, u, Delta, phi, w = [kwargs.get(key) for key in keys]
    tau_phi = np.cos(pi * phi) * tau_x + np.sin(pi * phi) * tau_y

    # function for onsite terms
    def onsite(site):
        y = site.pos[1]
        fam = site.family
        e0 = (u / 2) if fam == a1 or fam == b1 else (-u / 2)
        if y >= w / 2:
            return ( e0 - mu ) * tau_z + Delta * tau_x
        elif y <= -w / 2:
            return ( e0 - mu ) * tau_z + Delta * tau_phi
        else:
            return ( e0 - mu ) * tau_z

    # rectangular scattering region
    def rectangle(pos):
        x, y = pos
        return -Lx / 2 <= x <= Lx / 2 and -Ly / 2 <= y <= Ly / 2
    
    syst = kwant.Builder()

    # onsite
    syst[lat.shape(rectangle, (0, 0))] = onsite

    # hoppings

    # (n, (a, b)) : hopping between sublattices a and b with n = n_a - n_b where n_a and n_b are cell indices

    # intralayer nearest-neighbor hopping
    intra = (((0, 0), a1, b1), ((0, -1), a1, b1), ((1, -1), a1, b1), ((0, 0), a2, b2), ((0, -1), a2, b2), ((1, -1), a2, b2))    
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in intra]] = -t * tau_z

    # interlayer hopping
    syst[kwant.builder.HoppingKind((0, 0), a1, b2)] = -g1 * tau_z
    inter3 = (((-1, 2), b1, a2), ((0, 1), b1, a2), ((-1, 1), b1, a2))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in inter3]] = -g3 * tau_z
 
    # leads
    
    # In order to add leads with a larger translation period we follow the discussion found here:
    # https://kwant-discuss.kwant-project.narkive.com/sOepG1sx/kwant-symmetry-and-leads

    # First we specify a unit cell for the translational symmetry of the leads:
    sym_left = kwant.TranslationalSymmetry(lat.vec((1, -2)))
    sym_right = kwant.TranslationalSymmetry(lat.vec((-1, 2)))
    sym_up = kwant.TranslationalSymmetry((0, 1))
    sym_down = kwant.TranslationalSymmetry((0, -1))
    # Then we add a second (linear independent) lattice vector with other_vectors that
    # complements the translational vector above. Both vectors then form a basis.
    # Documentation: https://kwant-project.org/doc/1/reference/generated/kwant.lattice.TranslationalSymmetry
    sym_up.add_site_family(a1, other_vectors=[(-1, 2)])
    sym_up.add_site_family(b1, other_vectors=[(-1, 2)])
    sym_up.add_site_family(a2, other_vectors=[(-1, 2)])
    sym_up.add_site_family(b2, other_vectors=[(-1, 2)])

    sym_down.add_site_family(a1, other_vectors=[(-1, 2)])
    sym_down.add_site_family(b1, other_vectors=[(-1, 2)])
    sym_down.add_site_family(a2, other_vectors=[(-1, 2)])
    sym_down.add_site_family(b2, other_vectors=[(-1, 2)])

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

    # functions for onsite terms in the leads
    def onsite_lead(site):
        fam = site.family
        e0 = (u / 2) if fam == a1 or fam == b1 else (-u / 2)
        return ( e0 - mu ) * tau_z
    
    def onsite_lead_up(site):
        fam = site.family
        e0 = (u / 2) if fam == a1 or fam == b1 else (-u / 2)
        return ( e0 - mu ) * tau_z + Delta * tau_x

    def onsite_lead_down(site):
        fam = site.family
        e0 = (u / 2) if fam == a1 or fam == b1 else (-u / 2)
        return ( e0 - mu ) * tau_z + Delta * tau_phi

    # left lead
    lead0[lat.shape(lead_shape_v, (0, 0))] = onsite_lead
    lead0[[kwant.builder.HoppingKind(*hopping) for hopping in intra]] = -t * tau_z
    lead0[kwant.builder.HoppingKind((0, 0), a1, b2)]  = -g1 * tau_z
    lead0[[kwant.builder.HoppingKind(*hopping) for hopping in inter3]] = -g3 * tau_z

    # right lead
    lead1[lat.shape(lead_shape_v, (0, 0))] = onsite_lead
    lead1[[kwant.builder.HoppingKind(*hopping) for hopping in intra]] = -t * tau_z
    lead1[kwant.builder.HoppingKind((0, 0), a1, b2)]  = -g1 * tau_z
    lead1[[kwant.builder.HoppingKind(*hopping) for hopping in inter3]] = -g3 * tau_z
    
    # top lead
    lead2[lat.shape(lead_shape_h, (0, 0))] = onsite_lead_up
    lead2[[kwant.builder.HoppingKind(*hopping) for hopping in intra]] = -t * tau_z
    lead2[kwant.builder.HoppingKind((0, 0), a1, b2)]  = -g1 * tau_z
    lead2[[kwant.builder.HoppingKind(*hopping) for hopping in inter3]] = -g3 * tau_z

    # bottom lead
    lead3[lat.shape(lead_shape_h, (0, 0))] = onsite_lead_down
    lead3[[kwant.builder.HoppingKind(*hopping) for hopping in intra]] = -t * tau_z
    lead3[kwant.builder.HoppingKind((0, 0), a1, b2)]  = -g1 * tau_z
    lead3[[kwant.builder.HoppingKind(*hopping) for hopping in inter3]] = -g3 * tau_z

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

# show only one of the two graphene layers (for debugging hoppings)
# def family_color(site):
#     fam = site.family
#     return 'blue' if fam == a1 or fam == b1 else 'None'
#     # return 'None' if fam == a1 or fam == b1 else 'blue'

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
# t : intralayer nearest-neighbor hopping
# g1 : interlayer hopping \gamma_1 (t_\perp) [see Fig. 17(a)]
# g3 : interlayer hopping \gamma_3 [see Fig. 17(a)]
# u : interlayer bias
# mu : Fermi energy EF
# Delta : magnitude of the superconducting pair potential
# phi : phase difference between superconductors

# energy unit : t
# length unit : a
# in the paper we take Delta = 0.0025 and Lx = 600, Ly = 601 and w = 4

g1, g3, u = 0.1, 0.1, 0.2               # values for Fig. 18(b)
Delta, phi = 0.004, 1.0
mulis = [0.045, 0.072, 0.121, 0.18]     # values for Fig. 18(b)
print(f'xi/a = {np.sqrt(3) / (2 * pi * Delta)}', flush=True)

# to obtain good corners, Ly should be odd
Lx, Ly, w = 100, 99, 2
filename = f'bilayer_armchair_Lx_{Lx}_Ly_{Ly}_w_{w}_u_{u}_Delta_{Delta}_phi_{phi}_.hdf5'

# avoid extra column of atoms at the left and right edge
Nx = round((Lx / np.sqrt(3) + 1) / 2)
Lx = np.sqrt(3) * (2 * Nx - 1)
print(f'(Lx, Ly) = ({Lx}, {Ly})', flush=True)

## plot the system ##

print('-'*80, '\nplotting system - close window to start computation')

params = {'Lx': Lx, 'Ly': Ly, 'mu': 0.0, 't': 1.0, 'g1': g1, 'g3': g3, 'u': u, 'Delta': Delta, 'phi': phi, 'w': w}
fsyst = make_system(**params).finalized()
fig, ax = plt.subplots(1, 1)
kwant.plot(fsyst, ax=ax, site_size=0.2, site_color=site_color, hop_lw=0.05)
ax.axis('equal')
# ax.axis('off')
plt.xlabel(r'$x/a$')
plt.ylabel(r'$y/a$')
# plt.savefig(f'bilayer_armchair_Lx_{round(Lx)}_Ly_{Ly}_w_{w}_.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
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
    params = {'Lx': Lx, 'Ly': Ly, 'mu': mulis[i], 't': 1.0, 'g1': g1, 'g3': g3, 'u': u, 'Delta': Delta, 'phi': phi, 'w': w}
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
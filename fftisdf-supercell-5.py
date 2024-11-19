import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum
import scipy.linalg

import pyscf
from pyscf.pbc import tools
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

c   = pyscf.pbc.gto.Cell()
c.a = numpy.ones((3, 3)) * 3.5668 - numpy.eye(3) * 3.5668
# c.a = numpy.eye(3) * 3.5668
c.atom = '''C     0.0000  0.0000  0.0000
            C     0.8917  0.8917  0.8917 '''
            # C     1.7834  1.7834  0.0000
            # C     2.6751  2.6751  0.8917
            # C     1.7834  0.0000  1.7834
            # C     2.6751  0.8917  2.6751
            # C     0.0000  1.7834  1.7834
            # C     0.8917  2.6751  2.6751'''
c.basis  = 'gth-szv'
c.pseudo = 'gth-pade'
c.verbose = 0
c.unit = 'aa'
c.max_memory = 100
c.mesh = [15, 15, 15]
c.build()

nao = c.nao_nr()
print(f"{nao = }")

from pyscf.pbc.dft.gen_grid import gen_uniform_grids, gen_becke_grids
coord = gen_uniform_grids(c, mesh=[11, 11, 11], wrap_around=False)
weigh = c.vol / coord.shape[0]
print(f"{coord.shape = }, {weigh.shape = }")

phi  = numpy.sqrt(weigh) * c.pbc_eval_gto("GTOval_sph", coord)
phi  = phi
ng, nao = phi.shape
print(f"{ng = }, {nao = }")

from pyscf.lib.scipy_helper import pivoted_cholesky
zeta = einsum("gm,gn,hm,hn->gh", phi, phi, phi, phi)
chol, perm, rank = pivoted_cholesky(zeta, tol=1e-30)
print(f"{rank = }")
mask = perm[:400]
nip = mask.size

# build the supercell
from pyscf.pbc.tools import k2gamma

kmesh = [2, ] * 3
nimg = nkpt = numpy.prod(kmesh)

tv = k2gamma.translation_vectors_for_kmesh(c, kmesh, wrap_around=False)
kpts = c.make_kpts(kmesh)
sc, phase = k2gamma.get_phase(c, kpts, kmesh=kmesh, wrap_around=False)

ng = coord.shape[0]
g0 = coord
gx = coord[None, :, :] + tv[:, None, :]
assert gx.shape == (nimg, ng, 3)
gx = gx.reshape(-1, 3)
gw = weigh

# Notations: *_full stands for the full supercell quantities, has the shape of (nimg, *, nimg, *)
#            *_k   stands for the k-point quantities, has the shape of (nkpt, *, *)
#            *_s   stands for the stripe quantities, has the shape of (nimg, *, *)

phi_full = numpy.sqrt(gw) * sc.pbc_eval_gto("GTOval_sph", gx)
phi_full = phi_full.reshape(nimg, ng, nimg, nao)
# rho_full_ref = einsum("RgSm,RgLn->RgSmLn", phi_full, phi_full)

phi_s = phi_full[0]
phi_s = phi_s.reshape(ng, nimg, nao)

# phi_s_1 = phi_full[0].reshape(ng, nimg, nao)
# phi_s_2 = phi_full[:, :, 0].reshape(nimg, ng, nao)
# phi_s_2 = phi_s_2.transpose(1, 0, 2)

# err = abs(phi_s_1 - phi_s_2).max()
# print(f"{err = }")
# assert err < 1e-10
# assert 1 == 2
rho_s_ref = einsum("gMm,gNn->gMmNn", phi_s, phi_s)

x2_k = []
for k1, vk1 in enumerate(kpts):
    x_k = c.pbc_eval_gto("GTOval_sph", g0, kpt=vk1)
    x_k *= numpy.sqrt(weigh)
    assert x_k.shape == (ng, nao)

    x2_k.append(x_k @ x_k.T.conj())

x2_k = numpy.asarray(x2_k)
x2_k = x2_k.reshape(nkpt, ng, ng)

x2_s = einsum("kgh,Rk->Rgh", x2_k, phase)
x2_s = x2_s.reshape(nimg, ng, ng)
assert abs(x2_s.imag).max() < 1e-10
x2_s = x2_s.real
assert 1 == 2

x4_s = x2_s * x2_s
x4_k = einsum("Rgh,Rk->kgh", x4_s, phase.conj())
x4_k = x4_k.reshape(nkpt, ng, ng) # / numpy.sqrt(nimg)

z_k = []
for k1, vk1 in enumerate(kpts):
    y = x4_k[k1][mask]
    a = x4_k[k1][mask][:, mask]

    assert a.shape == (nip, nip)
    assert y.shape == (nip, ng)

    z = scipy.linalg.lstsq(a, y)[0]
    err = abs(y - a @ z).max()
    print(f"{k1 = }, {err = :6.4e}")
    assert err < 1e-10

    z_k.append(z.reshape(nip, ng))

z_k = numpy.asarray(z_k)
z_s = einsum("kIg,Rk->RIg", z_k, phase)
z_s = z_s.reshape(nimg, nip, ng)
# assert abs(z_s.imag).max() < 1e-10
z_s = z_s.real

# x_s = phi_s_1[mask].reshape(nip, nimg, nao)
rho_s_ref = rho_s_ref.reshape(ng, nimg * nimg * nao * nao)

x_full = phi_full[:, mask].reshape(nimg, nip, nimg, nao)
rho_s_sol = einsum("RIg,RIMm,RINn->gMmNn", z_s, x_full, x_full)
rho_s_sol = rho_s_sol.reshape(ng, nimg * nimg * nao * nao) / numpy.sqrt(nimg)
err = abs(rho_s_sol - rho_s_ref).max()
print(f"{err = :6.4e}")
print("rho_s_sol = ")
numpy.savetxt(sys.stdout, rho_s_sol[:10, :10], fmt="% 8.4e", delimiter=", ")

print("rho_s_ref = ")
numpy.savetxt(sys.stdout, rho_s_ref[:10, :10], fmt="% 8.4e", delimiter=", ")

assert err < 1e-6

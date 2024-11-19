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

from pyscf.pbc.gto.cell import Cell

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

x4_s = x2_s * x2_s
x4_k = einsum("Rgh,Rk->kgh", x4_s, phase.conj())
x4_k = x4_k.reshape(nkpt, ng, ng) # / numpy.sqrt(nimg)

# x2_full_sol = einsum("kgh,Gk,Hk->GgHh", x2_k, phase, phase.conj())
# assert abs(x2_full_sol.imag).max() < 1e-10
# x2_full_sol = x2_full_sol.real

# x2_full_ref = einsum("GgMm,HhMm->GgHh", phi_full, phi_full)
# err = abs(x2_full_sol - x2_full_ref).max()
# print(f"{err = }")
# assert err < 1e-10

# x4_full_sol = einsum("kgh,Gk,Hk->GgHh", x4_k, phase, phase.conj()) / numpy.sqrt(nimg)
# assert abs(x4_full_sol.imag).max() < 1e-10
# x4_full_sol = x4_full_sol.real

# x4_full_ref = x2_full_sol * x2_full_sol
# err = abs(x4_full_sol - x4_full_ref).max()
# print(f"{err = }", f"{x4_full_sol[0, 0, 0, 0] / x4_full_ref[0, 0, 0, 0] = :6.4f}")
# assert err < 1e-10
# assert 1 == 2

z_k = []
for k1, vk1 in enumerate(kpts):
    y = x4_k[k1][mask]
    a = x4_k[k1][mask][:, mask]

    assert a.shape == (nip, nip)
    assert y.shape == (nip, ng)

    z = scipy.linalg.lstsq(a, y)[0]
    err = abs(y - a @ z).max()
    print(f"{k1 = }, {err = }")
    assert err < 1e-10

    z_k.append(z.reshape(nip, ng))

z_k = numpy.asarray(z_k)
# z_s = einsum("kgh,Rk->Rgh", z_k, phase)
# z_s = z_s.reshape(nimg, ng, ng) / numpy.sqrt(nimg)

z_full = einsum("kgh,Gk,Hk->GgHh", z_k, phase, phase.conj())
# assert abs(z_full.imag).max() < 1e-10
z_full = z_full.real
assert z_full.shape == (nimg, nip, nimg, ng)

x_full = phi_full[:, mask]
assert x_full.shape == (nimg, nip, nimg, nao)

# rho_full_sol = einsum("RIGg,RIMm,RINn->GgMmNn", z_full, x_full, x_full)
# err = abs(rho_full_sol - rho_full_ref).max()
# print(f"{err = }")
# print(f"{rho_full_sol.shape = }")
# rho_full_sol = rho_full_sol.reshape(nimg * ng, nimg * nao * nimg * nao)
# numpy.savetxt(c.stdout, rho_full_sol[:10, :10], fmt="% 8.4e", delimiter=", ")

# print(f"{rho_full_ref.shape = }")
# rho_full_ref = rho_full_ref.reshape(nimg * ng, nimg * nao * nimg * nao)
# numpy.savetxt(c.stdout, rho_full_ref[:10, :10], fmt="% 8.4e", delimiter=", ")

# assert err < 1e-5

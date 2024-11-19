import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum

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

# from pyscf.pbc.df.fft import FFTDF
# df = FFTDF(c)
# df.max_memory = 4
# gmesh = c.mesh

from pyscf.pbc.dft.gen_grid import gen_uniform_grids, gen_becke_grids
coord = gen_uniform_grids(c, mesh=[7, 7, 7], wrap_around=False)
weigh = c.vol / coord.shape[0]
print(f"{coord.shape = }, {weigh.shape = }")

# # coord, weigh = gen_becke_grids(c, level=1)
# # print(f"{coord.shape = }, {weigh.shape = }")

phi  = numpy.sqrt(weigh) * c.pbc_eval_gto("GTOval_sph", coord)
phi  = phi
ng, nao = phi.shape
print(f"{ng = }, {nao = }")

# ovlp_sol = einsum("mg,ng->mn", phi, phi)
# ovlp_ref = c.pbc_intor('int1e_ovlp', hermi=0)
# print(f"{ovlp_sol.shape = }")
# numpy.savetxt(c.stdout, ovlp_sol, fmt="% 6.4f", delimiter=", ")
# print(f"{ovlp_ref.shape = }")
# numpy.savetxt(c.stdout, ovlp_ref, fmt="% 6.4f", delimiter=", ")
# err = abs(ovlp_ref - ovlp_sol).max()
# print(f"{err = }")
# assert 1 == 2

from pyscf.lib.scipy_helper import pivoted_cholesky
zeta = einsum("gm,gn,hm,hn->gh", phi, phi, phi, phi)
chol, perm, rank = pivoted_cholesky(zeta, tol=1e-30)
print(f"{rank = }")
mask = perm[:40]
nip = mask.size

z = zeta[mask][:, mask]
rhs = zeta[mask, :]
assert z.shape == (nip, nip)
assert rhs.shape == (nip, ng)

xi = scipy.linalg.lstsq(z, rhs)[0]
assert xi.shape == (nip, ng)
res = z @ xi - rhs
err = abs(res).max()
print(f"{err = :6.4e}")
assert err < 1e-10

rho_ref = einsum("gm,gn->gmn", phi, phi)
rho_sol = einsum("Ig,Im,In->gmn", xi, phi[mask], phi[mask])
err = abs(rho_ref - rho_sol).max()
print(f"{err = :6.4e}")
assert err < 1e-10

# build the supercell
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import super_cell
from pyscf.pbc.tools.pbc import _build_supcell_

kmesh = [3, 3, 3]
nimg = numpy.prod(kmesh)
tv = k2gamma.translation_vectors_for_kmesh(c, kmesh, wrap_around=False)
kpts = c.make_kpts(kmesh)
sc, phase = k2gamma.get_phase(c, kpts, kmesh=kmesh, wrap_around=False)

# x0 = c.atom_coords()
# x1 = sc.atom_coords().reshape(-1, 2, 3)

# df_sc = FFTDF(sc)
# df_sc.max_memory = 2000

ng = coord.shape[0]
g0 = coord
gx = coord[None, :, :] + tv[:, None, :]
assert gx.shape == (nimg, ng, 3)
gx = gx.reshape(-1, 3)
gw = weigh

# phi1 = numpy.sqrt(gw) * sc.pbc_eval_gto("GTOval_sph", gx)
# phi1 = phi1.T
# assert phi1.shape == (nao * numpy.prod(nimg), ng * numpy.prod(nimg))

# print(f"{phi1.shape = }")
# # numpy.savetxt(c.stdout, phi1, fmt="% 6.4f", delimiter=", ")

# # coord = coord[mask]
# nip = coord.shape[0]
# coord = coord[None, :, :] + tv[:, None, :]
# coord = coord.reshape(-1, 3)
# phi2 = numpy.sqrt(weigh) * sc.pbc_eval_gto("GTOval_sph", coord)
# phi2 = phi2.T

# print(f"{phi2.shape = }")

# assert phi2.shape == (nao * numpy.prod(nimg), nip * numpy.prod(nimg))

# err = abs(phi1 - phi2).max()
# print(f"{err = :6.4e}")
# assert err < 1e-10

# print(f"{phi1.shape = }, {phi2.shape = }")
# rhs = einsum("mg,ng,mI,nI->gI", phi1, phi1, phi2, phi2)
# assert rhs.shape == (ng * numpy.prod(nimg), nip * numpy.prod(nimg))
# print(f"{rhs.shape = }, {ng = }, {nip = }")
# numpy.savetxt(c.stdout, rhs, fmt="% 8.2f", delimiter=", ")

phi = numpy.sqrt(gw) * sc.pbc_eval_gto("GTOval_sph", gx)
phi_s = phi.reshape(nimg, ng, nimg, nao)
phi_k = einsum("RgSm,Rk,Sl->kglm", phi_s, phase.conj(), phase)
for k1 in range(nimg):
    for k2 in range(nimg):
        if k1 != k2:
            err = abs(phi_k[k1, :, k2, :]).max()
            assert err < 1e-10

        else:
            f_sol = phi_k[k1, :, k1, :]

            f_ref = c.pbc_eval_gto("GTOval_sph", g0, kpt=kpts[k1])
            f_ref *= numpy.sqrt(weigh)
            err = abs(f_sol - f_ref).max()
            assert err < 1e-10

# rhs = einsum("gm,gn,hm,hn->gh", phi, phi, phi, phi)
tmp = einsum("gm,hm->gh", phi, phi)
rhs = tmp * tmp

tmp_s = einsum("RgSm,LhSm->RgLh", phi_s, phi_s)
tmp_s_ref = tmp_s.reshape(nimg, ng, nimg, ng)
err = abs(tmp.reshape(nimg, ng, nimg, ng) - tmp_s).max()
print(f"{err = }")
assert err < 1e-10

tmp_k = einsum("RgLh,Rk,Ll->kglh", tmp_s, phase.conj(), phase)

phi_kk = [phi_k[k, :, k, :] for k in range(nimg)]
phi_kk = numpy.array(phi_kk).reshape(nimg, ng, nao)
tmp_k_sol = einsum("kgm,khm->kgh", phi_kk, phi_kk.conj())
tmp_k_sol = tmp_k_sol.reshape(nimg, ng, ng)

for k1 in range(nimg):
    for k2 in range(nimg):
        if k1 != k2:
            err = abs(tmp_k[k1, :, k2, :]).max()
            assert err < 1e-10

        else:
            f = phi_k[k1, :, k1, :]
            tmp_ref = einsum("gm,hm->gh", f, f.conj())
            err = abs(tmp_ref - tmp_k[k1, :, k1, :]).max()
            assert err < 1e-10
tmp_k_ref = numpy.array([tmp_k[k, :, k, :] for k in range(nimg)])
tmp_k_ref = tmp_k_ref.reshape(nimg, ng, ng)
err = abs(tmp_k_ref - tmp_k_sol).max()
print(f"{err = :6.4e}")
assert err < 1e-10

tmp_k = tmp_k_ref.reshape(nimg, ng, ng)
tmp_x_sol = einsum("kgh,Rk->Rgh", tmp_k, phase)
tmp_x_sol = tmp_x_sol.reshape(nimg, ng, ng) / numpy.sqrt(nimg)
assert abs(tmp_x_sol.imag).max() < 1e-10
tmp_x_sol = tmp_x_sol.real

tmp_x_ref = tmp_s[0].transpose(1, 2, 0)
tmp_x_ref = tmp_x_ref.reshape(nimg, ng, ng)
err = abs(tmp_x_sol - tmp_x_ref).max()
assert err < 1e-10
tmp_x = tmp_x_sol

rhs_s = einsum("RgSm,RgKn,LhSm,LhKn->RgLh", phi_s, phi_s, phi_s, phi_s)
rhs_s_sol = rhs_s.reshape(nimg * ng, nimg * ng)
# tmp_s = tmp_s.reshape(nimg * ng, nimg * ng)
rhs_s_ref = (tmp_s * tmp_s).reshape(nimg * ng, nimg * ng)
err = abs(rhs_s_sol - rhs_s_ref).max()
assert err < 1e-10

# rhs_k = einsum("RgLh,Rk,Ll,RgLh,Rk,Ll->kglh", tmp_s, phase.conj(), phase, tmp_s, phase, phase.conj())
# they are identical, but wrong!
# rhs_k_sol = tmp_k * tmp_k.conj()
# rhs_k_ref = einsum("RgLh,Rk,Ll,PgSh,Pk,Sl->kglh", tmp_s, phase.conj(), phase, tmp_s, phase, phase.conj())
# err = abs(rhs_k_sol - rhs_k_ref).max()
# assert err < 1e-10

# this is the only correct way of doing this
rhs_k_ref = einsum("RgLh,Rk,Ll->kglh", rhs_s, phase, phase.conj())
rhs_k_sol = einsum("RgLh,RgLh,Rk,Ll->kglh", tmp_s, tmp_s, phase, phase.conj())

err = abs(rhs_k_ref - rhs_k_sol).max()
assert err < 1e-10

rhs_x_sol = tmp_x * tmp_x
assert rhs_x_sol.shape == (nimg, ng, ng)

rhs_x_ref = rhs_s[0].transpose(1, 2, 0)
rhs_x_ref = rhs_x_ref.reshape(nimg, ng, ng)

err = abs(rhs_x_sol - rhs_x_ref).max()
print(f"{err = :6.4e}")
assert err < 1e-10

rhs_k_sol = einsum("Rgh,Rk->kgh", rhs_x_sol, phase) * numpy.sqrt(nimg)
rhs_k_ref = einsum("RgLh,Rk,Lk->kgh", rhs_s, phase, phase.conj())

err = abs(rhs_k_sol - rhs_k_ref).max()
print(f"{err = :6.4e}")
assert err < 1e-10

rhs_s_sol = einsum("kgh,Rk,Sk->RgSh", rhs_k_sol, phase.conj(), phase)
assert abs(rhs_s_sol.imag).max() < 1e-10
rhs_s_sol = rhs_s_sol.real
rhs_s_sol = rhs_s_sol.reshape(nimg * ng, nimg * ng)

err = abs(rhs_s_sol - rhs_s_ref).max()
print(f"{err = :6.4e}")
assert err < 1e-10

rhs_k = rhs_k_sol
zeta_k = rhs_k

print(f"{ng = }, {zeta_k.shape = }")

xi_k = []
for k in range(nimg):
    z = zeta_k[k][mask][:, mask]
    rhs = zeta_k[k][mask, :]
    assert z.shape == (nip, nip)
    assert rhs.shape == (nip, ng)

    xi = scipy.linalg.lstsq(z, rhs)[0]
    assert xi.shape == (nip, ng)
    res = z @ xi - rhs
    err = abs(res).max()
    print(f"{err = :6.4e}")
    assert err < 1e-10

    xi_k.append(xi)

xi_k = numpy.array(xi_k)
assert xi_k.shape == (nimg, nip, ng)

xi_s = einsum("kgh,Rk,Sk->RgSh", xi_k, phase, phase.conj())
assert xi_s.shape == (nimg, nip, nimg, ng)

rho_s_sol = einsum("SIRg,SILn,SILm->RgSmLn", xi_s, phi_s[:, mask], phi_s[:, mask])
rho_s_ref = einsum("RgSm,RgLn->RgSmLn", phi_s, phi_s)

err = abs(rho_s_sol - rho_s_ref).max()
print(f"{err = :6.4e}")
assert err < 1e-10
assert 1 == 2
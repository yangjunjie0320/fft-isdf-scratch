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
assert 1 == 2

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
err = abs(tmp.reshape(nimg, ng, nimg, ng) - tmp_s).max()
print(f"{err = }")
assert err < 1e-10

tmp_k = einsum("RgLh,Rk,Ll->kglh", tmp_s, phase.conj(), phase)
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

rhs_s = einsum("RgSm,RgKn,LhSm,LhKn->RgLh", phi_s, phi_s, phi_s, phi_s)
rhs_s_sol = rhs_s.reshape(nimg * ng, nimg * ng)
# tmp_s = tmp_s.reshape(nimg * ng, nimg * ng)
rhs_s_ref = (tmp_s * tmp_s).reshape(nimg * ng, nimg * ng)
err = abs(rhs_s_sol - rhs_s_ref).max()
print(f"{err = :6.4e}")
assert err < 1e-10

# rhs_k = einsum("RgLh,Rk,Ll,RgLh,Rk,Ll->kglh", tmp_s, phase.conj(), phase, tmp_s, phase, phase.conj())
# they are identical, but wrong!
rhs_k_sol = tmp_k * tmp_k.conj()
rhs_k_ref = einsum("RgLh,Rk,Ll,PgSh,Pk,Sl->kglh", tmp_s, phase.conj(), phase, tmp_s, phase, phase.conj())
err = abs(rhs_k_sol - rhs_k_ref).max()
print(f"{err = :6.4e}")
assert err < 1e-10

# this is the only correct way of doing this
rhs_k_ref = einsum("RgLh,Rk,Ll->kglh", rhs_s, phase, phase.conj())
rhs_k_sol = einsum("RgLh,RgLh,Rk,Ll->kglh", tmp_s, tmp_s, phase, phase.conj())

err = abs(rhs_k_ref - rhs_k_sol).max()
print(f"{err = :6.4e}")
assert err < 1e-10

# id_sol = einsum("Rk,Rl->kl", phase.conj(), phase)
# id_ref = numpy.eye(nimg)
# err = abs(id_sol - id_ref).max()
# print(f"{err = }")
# assert err < 1e-10

# for k1 in range(nimg):
#     for k2 in range(nimg):
#         if k1 != k2:
#             err = abs(rhs_k[k1, :, k2, :]).max()
#             print(f"{err = }")
#             assert err < 1e-10

#         else:
#             f = phi_k[k1, :, k1, :]
#             assert abs(f.imag).max() < 1e-10
#             f = f.real

#             tmp = einsum("gm,hm->gh", f, f)
#             rhs_ref = tmp * tmp

#             print("\nrhs_ref real")
#             numpy.savetxt(c.stdout, rhs_ref[:10, :10].real, fmt="% 6.4e", delimiter=", ")
#             print("rhs_ref imag")
#             numpy.savetxt(c.stdout, rhs_ref[:10, :10].imag, fmt="% 6.4e", delimiter=", ")

#             print("\nrhs_k real")
#             numpy.savetxt(c.stdout, rhs_k[k1, :, k1, :][:10, :10].real, fmt="% 6.4e", delimiter=", ")
#             print("rhs_k imag")
#             numpy.savetxt(c.stdout, rhs_k[k1, :, k1, :][:10, :10].imag, fmt="% 6.4e", delimiter=", ")

#             err_real = abs(rhs_ref.real - rhs_k[k1, :, k1, :].real).max()
#             err_imag = abs(rhs_ref.imag - rhs_k[k1, :, k1, :].imag).max()
#             err = err_real + err_imag
#             print(f"{err_real = }, {err_imag = }, {err = }")
#             assert err < 1e-10

# err = abs(rhs.reshape(nimg, ng, nimg, ng) - rhs_s).max()
# print(f"{err = }")
# assert err < 1e-10

# assert 1 == 2

# print(f"{ovlp.shape = }")
# numpy.savetxt(c.stdout, ovlp[:8, :8], fmt="% 8.2f", delimiter=", ")
# numpy.savetxt(c.stdout, ovlp[8:16, 8:16], fmt="% 8.2f", delimiter=", ")

# for i in range(numpy.prod(nimg)):
#     for j in range(numpy.prod(nimg)):
#         ig = slice(i * ng, (i+1) * ng)
#         jp = slice(j * nip, (j+1) * nip)

#         imj = (i - j) % numpy.prod(nimg)
#         jmi = (j - i) % numpy.prod(nimg)

#         g0 = slice(0, ng)
#         p0 = slice(imj * nip, (imj+1) * nip)
#         err = abs(rhs[g0, p0] - rhs[ig, jp]).max()
#         print(f"{i = }, {j = }, {err = }")

# rhs_s = rhs.reshape(numpy.prod(nimg), ng, numpy.prod(nimg), nip)
# rhs_k = einsum("RgSi,Rk,Sl->kgli", rhs_s, phase.conj(), phase)

# for k1 in range(numpy.prod(nimg)):
#     for k2 in range(numpy.prod(nimg)):
#         print(f"{k1 = }, {k2 = }")
#         if k1 != k2:
#             err = abs(rhs_k[k1, k2]).max()
#             print(f"{err = }")

# print(f"{rhs.shape = }, {rhs.max() = }, {rhs.min() = }")
# print(f"{rhs[0:ng, 0:nip].shape = }, {rhs[0:ng, 0:nip].max() = }, {rhs[0:ng, 0:nip].min() = }")
# numpy.savetxt(c.stdout, rhs, fmt="% 8.2f", delimiter=", ")

# print(f"{gx.shape = }, {gx[:10] = }")
# print(f"{coord.shape = }, {coord[:10] = }")
# assert 1 == 2



# phi = numpy.sqrt(weigh) * sc.pbc_eval_gto("GTOval_sph", coord)
# phi = phi.T

# print(f"{df.grids.coords.shape = }, {df.grids.coords[:10] = }")

# print(f"{gx.shape = }, {gx[:10] = }")
# print(f"{df_sc.grids.coords.shape = }, {df_sc.grids.coords[:10] = }")

# assert abs(gx - df_sc.grids.coords).max() < 1e-10


# gw = df.grids.weights
# print(f"{gx.shape = }, {gw.shape = }")
# for ao, p0, p1 in df_sc.aoR_loop(grids=grids, deriv=0):
#     print(f"{p0 = }, {p1 = }")
#     weigh = ao[3]
#     coord = ao[4]   

#     print(coord[:10])
#     assert 1 == 2
#     ng, nao = ao[0][0].shape

#     rhs = einsum("gm,gn,mI,nI->gI", ao[0][0], ao[0][0], phi, phi)
#     assert rhs.shape == (ng, rank * numpy.prod(nimg))

#     # check for the translation invariance
#     for i in range(numpy.prod(nimg)):
#         err = abs(rhs[i:(i+ng), :] - rhs[:ng, :]).max()

#     # res = scipy.linalg.lstsq(zeta[mask][:, mask], rhs.T)
#     # z = res[0].T
#     # resid = res[1]
#     # rank = res[2]
#     # print(f"{z.shape = }, {rank = }, {resid = }")

#     # resid = zeta[mask][:, mask] @ z.T - rhs.T
#     # print(f"{resid.shape = }, {resid.max() = }")

#     # rho_ref = einsum("gm,gn->gmn", ao[0][0], ao[0][0])
#     # print(f"{rho_ref.shape = }, {rho_ref.max() = }")

#     # rho_sol = einsum("gI,mI,nI->gmn", z, phi[:, mask], phi[:, mask])
#     # print(f"{rho_sol.shape = }, {rho_sol.max() = }")

#     # err = abs(rho_ref - rho_sol).max()
#     # print(f"{err = }")

#     assert 1 == 2


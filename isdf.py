import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum

import pyscf
from pyscf.pbc import tools
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

c = pbcgto.Cell()
c.atom = """
He 1.000000 1.000000 2.000000
He 1.000000 1.000000 4.000000
"""
c.a = numpy.diag([2.0, 2.0, 6.0])
c.basis = "sto3g"
c.verbose = 0
# c.mesh = [10] * 3   
c.mesh = [10] * 3
c.build()

from pyscf.pbc.lib.kpts_helper import get_kconserv
vk = c.make_kpts([4, 4, 4], wrap_around=False)
kconserv3 = get_kconserv(c, vk)
kconserv2 = kconserv3[:, :, 0].T

nk = vk.shape[0]
nq = len(numpy.unique(kconserv2))
assert nk == nq

df = pyscf.pbc.df.FFTDF(c)
gmesh = c.mesh
coord = c.gen_uniform_grids(mesh=gmesh, wrap_around=False)
phik  = pbcdft.numint.eval_ao_kpts(c, coord, kpts=vk, deriv=0)
phik  = numpy.asarray(phik)
ng, nao = phik.shape[1:]

zeta = einsum("kgm,khm,lgn,lhn->gh", phik.conj(), phik, phik.conj(), phik) / nk / nk
assert abs(zeta.imag).max() < 1e-10
zeta = zeta.real

from pyscf.lib.scipy_helper import pivoted_cholesky
chol, perm, rank = pivoted_cholesky(zeta, tol=1e-20)
mask = perm[:rank]

res = scipy.linalg.lstsq(zeta[mask][:, mask], zeta[mask, :])
z = res[0].T
nip = z.shape[1]
assert z.shape == (ng, nip)
print(f"{z.shape = }, {nip = }")

# for (k1, k2) in itertools.product(range(nk), repeat=2):
#     vk1, vk2 = vk[k1], vk[k2]

#     rho_k1_k2_ref = einsum("gm,gn->gmn", phik[k1].conj(), phik[k2])
#     rho_k1_k2_sol = einsum("gI,Im,In->gmn", z, phik[k1][mask].conj(), phik[k2][mask])

#     err = abs(rho_k1_k2_ref - rho_k1_k2_sol).max()
#     print(f"k1 = {k1:4d}, k2 = {k2:4d}, err = {err:6.2e}")
#     assert err < 1e-5

for (k1, k2, k3) in itertools.product(range(nk), repeat=3):
    vk1, vk2, vk3 = vk[k1], vk[k2], vk[k3]

    q = kconserv2[k1, k2]
    vq = vk[q]
    k4 = kconserv3[k1, k2, k3]
    vk4 = vk[k4]

    # if not (k1, k2, k3, k4) == (1, 0, 0, 1):
    #     continue

    from pyscf.pbc.df.df_ao2mo import _iskconserv
    from pyscf.pbc.df.df_ao2mo import _format_kpts
    vk1234 = numpy.asarray([vk1, vk2, vk3, vk4])
    assert _iskconserv(c, vk1234)

    rho_k1_k2_ref = einsum("gm,gn->gmn", phik[k1].conj(), phik[k2])
    rho_k3_k4_ref = einsum("gm,gn->gmn", phik[k3].conj(), phik[k4])
    rho_12 = rho_k1_k2_ref
    rho_34 = rho_k3_k4_ref

    rho_k1_k2_sol = einsum("gI,Im,In->gmn", z, phik[k1][mask].conj(), phik[k2][mask])
    rho_k1_k2 = rho_k1_k2_sol
    err = numpy.linalg.norm(rho_k1_k2_ref - rho_k1_k2_sol)
    print(f"k1 = {k1:4d}, k2 = {k2:4d}, err = {err:6.2e}")

    from pyscf.pbc.tools import fft, ifft
    t12 = numpy.dot(coord, vq) # 12)
    f12 = numpy.exp(1j * t12)
    z12_g = fft(einsum("gI,g->Ig", z, f12).reshape(-1, ng), gmesh)
    assert z12_g.shape == (nip, ng)
    # v12_g  = z12_g * tools.get_coulG(c, k=vq, mesh=gmesh) * (c.vol / ng)
    # assert v12_g.shape == (nip, ng)
    ao_pair_g_sol = einsum("Ig,Im,In->gmn", z12_g, phik[k1][mask], phik[k2][mask].conj())
    ao_pair_g_sol = ao_pair_g_sol.reshape(ng, -1)
    # z12_g = fft(einsum("gmn,g->mng", rho_12, f12).reshape(-1, ng), gmesh)
    ao_pair_g_ref = einsum("gmn,g->mng", rho_k1_k2, f12).reshape(-1, ng)
    ao_pair_g_ref = fft(ao_pair_g_ref, gmesh).transpose(1, 0)

    err = abs(ao_pair_g_ref - ao_pair_g_sol).max()
    assert err < 1e-10, f"{err = :6.2e}"

    from pyscf.pbc.df.fft_ao2mo import get_ao_pairs_G
    ao_pair_g_ref = get_ao_pairs_G(df, [vk1, vk2], vq, compact=False)
    err = numpy.linalg.norm(ao_pair_g_ref - ao_pair_g_sol)
    print(f"{ao_pair_g_ref.shape = }")
    numpy.savetxt(sys.stdout, ao_pair_g_ref[:10], fmt="% 12.6f", delimiter=", ")

    print(f"{ao_pair_g_sol.shape = }")
    numpy.savetxt(sys.stdout, ao_pair_g_sol[:10], fmt="% 12.6f", delimiter=", ")

    assert err < 1e-10, f"{err = :6.2e}"

    ao_pair_ginv = get_ao_pairs_G(df, [-vk3, -vk4], vq, compact=False).conj()

    ao_pair_g = ao_pair_g_ref
    coul_g = tools.get_coulG(c, k=vq, mesh=gmesh) * (c.vol / ng / ng)
    print(f"{ao_pair_g.shape = }, {ao_pair_ginv.shape = }")
    # assert 1 == 2
    # v12_g = z12_g * tools.get_coulG(c, k=vq, mesh=gmesh) * (c.vol / ng)

    # t34 = numpy.dot(coord, vq)
    # f34 = numpy.exp(1j * t34)
    # z34_g = ifft(einsum("gI,g->Ig", z, f34).reshape(-1, ng), gmesh)
    # assert z34_g.shape == (nip, ng)
    # z34_g = ifft(einsum("gmn,g->mng", rho_34, f34).reshape(-1, ng), gmesh)
    # coul_q = einsum("Ig,Jg->IJ", v12_g, z34_g)

    # coul_g_vq   = tools.get_coulG(c, k=vk[q], mesh=gmesh)
    # print(f"\n{coul_g_vq.shape = }, {coul_g_vq.dtype = }")
    # numpy.savetxt(sys.stdout, coul_g_vq[:10], fmt="% 12.6f", delimiter=", ")

    # coul_g_vq12 = tools.get_coulG(c, k=vq12, mesh=gmesh)
    # print(f"\n{coul_g_vq12.shape = }, {coul_g_vq12.dtype = }")
    # numpy.savetxt(sys.stdout, coul_g_vq12[:10], fmt="% 12.6f", delimiter=", ")

    # coul_g_vq34 = tools.get_coulG(c, k=vq34, mesh=gmesh)
    # print(f"\n{coul_g_vq34.shape = }, {coul_g_vq34.dtype = }")
    # numpy.savetxt(sys.stdout, coul_g_vq34[:10], fmt="% 12.6f", delimiter=", ")
    # assert 1 == 2

    # assert not (k1, k2, k3, k4) == (1, 0, 0, 1)

    # assert _iskconserv(c, vk1234), f"{vk1 = }, {vk2 = }, {vk3 = }, {vk4 = }"
    # x1, x2, x3, x4 = pbcdft.numint.eval_ao_kpts(c, coord[mask], kpts=vk1234, deriv=0)

    # eri_sol = einsum("IJ,Im,In,Jk,Jl->mnkl",
    #         coul_q, x1.conj(), x2, x3.conj(), x4,
    #         optimize=True
    #         )
    eri_sol = einsum("gx,gy,g->xy", ao_pair_g, ao_pair_ginv, coul_g, optimize=True)
    eri_sol = eri_sol.reshape(nao * nao, nao * nao)

    eri_ref = df.get_eri(vk1234, compact=False)
    eri_ref = eri_ref.reshape(nao * nao, nao * nao)

    err1 = numpy.abs(eri_sol.real - eri_ref.real).max()
    err2 = numpy.abs(eri_sol.imag - eri_ref.imag).max()

    print(f"k1 = {k1:4d}, k2 = {k2:4d}, k3 = {k3:4d}, k4 = {k4:4d}, err = {err1 + err2:6.2e}")
    if not err1 + err2 < 1e-10:
        print("vk = ")
        numpy.savetxt(sys.stdout, numpy.asarray(vk1234), fmt="% 8.4f", delimiter=", ")

        print("vq = ")
        numpy.savetxt(sys.stdout, vq, fmt="% 8.4f", delimiter=", ")
        
        print(f"err1 = {err1:6.2e}, err2 = {err2:6.2e}")

        print("eri_sol (real) = ")
        numpy.savetxt(sys.stdout, eri_sol.real, fmt="% 12.6f", delimiter=", ")

        print("eri_sol (imag) = ")
        numpy.savetxt(sys.stdout, eri_sol.imag, fmt="% 12.6f", delimiter=", ")

        print("eri_ref = ")
        numpy.savetxt(sys.stdout, eri_ref.real, fmt="% 12.6f", delimiter=", ")

        print("eri_ref = ")
        numpy.savetxt(sys.stdout, eri_ref.imag, fmt="% 12.6f", delimiter=", ")

        raise RuntimeError("eri_err too large")

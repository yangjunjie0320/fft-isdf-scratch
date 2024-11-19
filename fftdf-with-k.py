import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum
import scipy.linalg

import pyscf
from pyscf.lib import logger, current_memory

from pyscf.pbc import tools as pbctools
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

def gen_interpolating_points(c, gmesh=None, cisdf=10.0):
    if gmesh is None:
        gmesh = [11, 11, 11]

    from pyscf.pbc.dft.gen_grid import gen_uniform_grids
    gx = gen_uniform_grids(c, mesh=gmesh, wrap_around=False)
    gw = c.vol / gx.shape[0]

    phi = numpy.sqrt(gw) * c.pbc_eval_gto("GTOval_sph", gx)
    ng, nao = phi.shape

    x4 = einsum("gm,gn,hm,hn->ghmn", phi, phi, phi, phi)

    from pyscf.lib.scipy_helper import pivoted_cholesky
    chol, perm, rank = pivoted_cholesky(x4, tol=1e-30)
    nip = min(rank, cisdf * nao)

    mask = perm[:nip]
    return gx[mask], gw

def get_coul(c, ke=None, kmesh=None, cisdf=10.0):
    if ke is None:
        ke = c.ke_cutoff

    if kmesh is None:
        kmesh = [1, 1, 1]

    from pyscf.pbc.tools.k2gamma import get_phase
    vk = c.get_kpts(kmesh)
    sc, phase = get_phase(c, vk, kmesh=kmesh, wrap_around=False)

    log = logger.new_logger(c, c.verbose)

    from pyscf.pbc.gto.cell import estimate_ke_cutoff
    k0 = estimate_ke_cutoff(c)
    if ke is None:
        ke = k0

    if ke < k0:
        info  = "ke_cutoff = %8.2f is smaller" % ke
        info += " than the recommended value %8.2f." % k0
        log.warn(info)  

    l = c.lattice_vectors()

    k1 = max(0.01 * ke, 8)
    gmesh = pbctools.cutoff_to_mesh(l, k1)
    ng = gmesh.prod()

    max_memory = c.max_memory - current_memory()[0]
    required_memory = ng * ng * 8 / 1e6
    log.info("Max memory = %d MB, required memory = %d MB.", max_memory, required_memory)

    if max_memory < required_memory:
        info  = "max_memory = %d MB is not enough.\n" % max_memory
        info += "Required memory = %d MB." % required_memory
        raise RuntimeError(info)

    nao = c.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    rank = (lambda x: x * (x + 1) // 2)(nao * nimg)
    if ng < rank:
        info  = "ng = %d is smaller" % ng
        info += " than the rank of the density pair %d." % rank
        log.warn(info)

    log.info("total rank = %d, selecting rank = %d, rate = %4.2f, recommended cisdf = %4.2f.", rank, int(nao * cisdf) * nimg, int(nao * cisdf) * nimg / rank, rank / nimg / nao)

    gx = c.gen_uniform_grids(gmesh)
    gw = c.vol / ng

    # select the interpolating points in the unit cell
    x0 = c.pbc_eval_gto("GTOval_sph", gx)
    x0 = numpy.sqrt(gw) * numpy.asarray(x0)
    assert x0.shape == (ng, nao)
    x4 = (numpy.dot(x0, x0.T)) ** 2

    from pyscf.lib.scipy_helper import pivoted_cholesky
    chol, perm, rank = pivoted_cholesky(x4, tol=1e-30)
    # nip = min(rank, cisdf * nao)
    nip = int(cisdf * nao)

    print(f"{nip = }, {rank = }")

    mask = perm[:nip]
    x_k = c.pbc_eval_gto("GTOval_sph", gx[mask], kpts=vk)
    x_k = numpy.sqrt(gw) * numpy.array(x_k)
    assert x_k.shape == (nkpt, nip, nao)

    x_s = einsum("Rk,kIm->RIm", phase, x_k)
    assert abs(x_s.imag).max() < 1e-10

    x_f = einsum("kIm,Rk,Sk->RISm", x_k, phase, phase.conj())
    assert x_f.shape == (nimg, nip, nimg, nao)
    assert abs(x_f.imag).max() < 1e-10
    x_f = x_f.reshape(nimg * nip, nimg * nao)

    # x2_f_ref = einsum("RISm,TJSm->RITJ", x_f, x_f)
    # x2_f_ref = x2_f_ref.reshape(nimg * nip, nimg * nip)

    x2_k = einsum("kIm,kJm->kIJ", x_k.conj(), x_k)
    assert x2_k.shape == (nkpt, nip, nip)

    # x2_f = einsum("kIJ,Rk,Sk->RISJ", x2_k, phase, phase.conj())
    # x2_f_sol = x2_f.reshape(nimg * nip, nimg * nip)

    # assert abs(x2_f_sol - x2_f_ref).max() < 1e-10, f"x2_f_sol is not equal to x2_f_ref"

    # assert abs(x2_f_sol.imag).max() < 1e-10, f"imaginary part of x2_f_sol is not zero, {abs(x2_f_sol).max() = }"

    x2_s = einsum("Rk,kIJ->RIJ", phase, x2_k)
    assert abs(x2_s.imag).max() < 1e-10

    # x2_s = einsum("Rk,kIJ->RIJ", phase, x2_k)
    # assert abs(x2_s).max() < 1e-10, f"imaginary part of x2_s is not zero, {abs(x2_s).max() = }"

    x4_s = x2_s * x2_s
    x4_k = einsum("Rk,RIJ->kIJ", phase.conj(), x4_s)
    assert x4_k.shape == (nkpt, nip, nip)

    # x4inv_k = [scipy.linalg.inv(x4_k[k]) for k in range(nkpt)]

    from pyscf.pbc.df.fft import FFTDF
    df_obj = FFTDF(c)
    df_obj.mesh = [21, 21, 21]
    grids = df_obj.grids
    coord = grids.coords
    weigh = grids.weights
    ngrid = coord.shape[0]

    mesh = df_obj.mesh
    assert ngrid == numpy.prod(mesh)

    # the size of the dataset is nkpt * ngrid * nip * 16 bytes
    log.info("Required disk space = %d MB", nkpt * ngrid * nip * 16 / 1e6)

    from pyscf.lib import H5TmpFile
    fswp = H5TmpFile()
    fswp.create_dataset("z", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
    
    # for ao_ks_etc, p0, p1 in df_obj.aoR_loop(grids, vk):
    ni = df_obj._numint
    nao = c.nao_nr()
    p0, p1 = 0, 0
    for ao_k_etc in ni.block_loop(c, grids, nao, deriv=0, kpts=vk, blksize=2000):
        f_k = numpy.asarray(ao_k_etc[0])
        p0, p1 = p1, p1 + f_k.shape[1]
        assert f_k.shape == (nkpt, p1 - p0, nao)

        print(f"\n{p0 = }, {p1 = }, {f_k.shape = }")

        # f_f = einsum("kgm,Rk,Sk->RgSm", f_k, phase, phase.conj())
        # assert abs(f_f.imag).max() < 1e-10
        # assert f_f.shape == (nimg, p1 - p0, nimg, nao)
        # f_f = f_f.reshape(nimg * (p1 - p0), nimg * nao).real

        assert f_k.shape == (nkpt, p1 - p0, nao)

        fx_k = einsum("kgm,kIm->kgI", f_k.conj(), x_k)
        assert fx_k.shape == (nkpt, p1 - p0, nip)

        fx_s = einsum("Rk,kgI->RgI", phase, fx_k)
        assert abs(fx_s.imag).max() < 1e-10

        y_s = fx_s * fx_s
        y_k = einsum("RgI,Rk->kgI", y_s, phase)
        assert y_k.shape == (nkpt, p1 - p0, nip)

        z_k = []
        for y, x4 in zip(y_k, x4_k):
            z_k.append(scipy.linalg.lstsq(x4, y.T)[0].T)
        z_k = numpy.asarray(z_k).reshape(nkpt, p1 - p0, nip)

        # z_f = einsum("kgI,Rk,Sk->RgSI", z_k, phase, phase.conj())
        # assert z_f.shape == (nimg, p1 - p0, nimg, nip)
        # z_f = z_f.reshape(nimg * (p1 - p0), nimg * nip)

        # rho_full_ref = einsum("gm,gn->gmn", f_f, f_f).real
        # rho_full_sol = einsum("gI,Im,In->gmn", z_f, x_f, x_f).real

        # rho_full_sol = rho_full_sol.reshape((p1 - p0) * nimg, -1)
        # rho_full_ref = rho_full_ref.reshape((p1 - p0) * nimg, -1)
        # print(f"{abs(rho_full_sol - rho_full_ref).max() = :6.2e}")

        # print(f"{rho_full_sol.shape = }")
        # numpy.savetxt(c.stdout, rho_full_sol[:10, :10], fmt="% 6.4f", delimiter=", ")

        # print(f"{rho_full_ref.shape = }")
        # numpy.savetxt(c.stdout, rho_full_ref[:10, :10], fmt="% 6.4f", delimiter=", ")
        # assert abs(rho_full_sol - rho_full_ref).max() < 1e-4

        fswp["z"][:, p0:p1, :] = z_k
        log.info("aoR_loop[%6d:%6d] done", p0, p1)

    # from pyscf.lib import prange
    # blkszie = 8

    from pyscf.pbc.lib.kpts_helper import get_kconserv
    from pyscf.pbc.lib.kpts_helper import get_kconserv_ria
    kconserv2 = get_kconserv_ria(c, vk)
    kconserv3 = get_kconserv(c, vk)

    # for k1, vk1 in enumerate(vk):
    #     for k2, vk2 in enumerate(vk):
    #         q = kconserv2[k1, k2]
    #         vq = vk[q]
    #         vk12 = vk2 - vk1
    #         err = abs(vq - vk12).max()
    #         assert err < 1e-10, f"vq = {vq}, vk12 = {vk12}, err = {err}"

    #         zq = fswp["z"][q, :, :].T
    #         rho_sol = einsum("Ig,Im,In->gmn", zq, x_k[k1].conj(), x_k[k2])

    #         phi_k1 = c.pbc_eval_gto("GTOval_sph", coord, kpts=vk1)
    #         phi_k2 = c.pbc_eval_gto("GTOval_sph", coord, kpts=vk2)
    #         rho_ref = einsum("gm,gn->gmn", phi_k1.conj(), phi_k2)

    #         eri = abs(rho_sol - rho_ref).max()
    #         print(f"{k1 = :2d}, {k2 = :2d}, {q = :2d} {eri = :6.2e}")

    # coul_k = []
    # gv = c.get_Gv(mesh)

    # for q, vq in enumerate(vk):
    #     phase = numpy.exp(-1j * numpy.dot(coord, vq))
    #     assert phase.shape == (ngrid, )

    #     coulg_k = pbctools.get_coulG(c, k=vq, mesh=mesh, Gv=gv) * c.vol / ngrid
    #     assert coulg_k.shape == (ngrid, )

    #     z_k = fswp["z_k"][q, :, :].T * phase
    #     assert z_k.shape == (nip, ngrid)

    #     zeta_g = pbctools.fft(z_k, mesh) * coulg_k
    #     assert zeta_g.shape == (nip, ngrid)

    #     zeta_k = pbctools.ifft(zeta_g, mesh)
    #     zeta_k *= phase.conj()

    #     coul_k.append(einsum("Ig,Jg->IJ", zeta_k, z_k))

    # coul_k = numpy.asarray(coul_k)
    # assert coul_k.shape == (nkpt, nip, nip)
    # return coul_k, x_k

if __name__ == "__main__":
    cell   = pyscf.pbc.gto.Cell()
    cell.a = numpy.ones((3, 3)) * 3.5668 - numpy.eye(3) * 3.5668
    cell.atom = '''C     0.0000  0.0000  0.0000
                C     0.8917  0.8917  0.8917 '''
    cell.basis  = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.max_memory = 100
    cell.build(dump_input=False)

    cell.verbose = 5
    nao = cell.nao_nr()

    kmesh = [2, 2, 2]
    nkpt = nimg = numpy.prod(kmesh)
    c, x = get_coul(cell, ke=50, kmesh=kmesh, cisdf=40.0)
    nkpt, nip, nao = x.shape

    assert c.shape == (nkpt, nip, nip)
    assert x.shape == (nkpt, nip, nao)

    from pyscf.pbc.df.fft import FFTDF
    df_obj = FFTDF(cell)
    df_obj.mesh = [21, 21, 21]

    from pyscf.pbc.lib.kpts_helper import get_kconserv
    from pyscf.pbc.lib.kpts_helper import get_kconserv_ria
    vk = cell.get_kpts(kmesh)
    kconserv3 = get_kconserv(cell, vk)
    kconserv2 = get_kconserv_ria(cell, vk)
    
    for k1, vk1 in enumerate(vk):
        for k2, vk2 in enumerate(vk):
            q = kconserv2[k1, k2]
            vq = vk[q]

            for k3, vk3 in enumerate(vk):
                k4 = kconserv3[k1, k2, k3]
                vk4 = vk[k4]

                # assert k1 == 0 and k2 == 0
                # assert k3 == 0 and k4 == 0

                eri_ref = df_obj.get_eri(kpts=[vk1, vk2, vk3, vk4], compact=False)
                eri_ref = eri_ref.reshape(nao * nao, nao * nao)

                eri_sol = einsum("IJ,Im,In,Jk,Jl->mnkl", c[q], x[k1].conj(), x[k2], x[k3], x[k4].conj())
                eri_sol = eri_sol.reshape(nao * nao, nao * nao)

                eri = abs(eri_sol - eri_ref).max()
                print(f"{k1 = :2d}, {k2 = :2d}, {k3 = :2d}, {k4 = :2d} {eri = :6.2e}")

                if eri > 1e-4:
                    # assert 1 == 2
                    
                    print("q = %2d, vq = [%s]" % (q, ", ".join(f"{v: 6.4f}" for v in vq)))
                    print("k1 = %2d, vk1 = [%s]" % (k1, ", ".join(f"{v: 6.4f}" for v in vk1)))
                    print("k2 = %2d, vk2 = [%s]" % (k2, ", ".join(f"{v: 6.4f}" for v in vk2)))
                    print("k3 = %2d, vk3 = [%s]" % (k3, ", ".join(f"{v: 6.4f}" for v in vk3)))
                    print("k4 = %2d, vk4 = [%s]" % (k4, ", ".join(f"{v: 6.4f}" for v in vk4)))

                    print(f"\n{eri_sol.shape = }")
                    numpy.savetxt(cell.stdout, eri_sol[:10, :10].real, fmt="% 6.4f", delimiter=", ")

                    print(f"\n{eri_ref.shape = }")
                    numpy.savetxt(cell.stdout, eri_ref[:10, :10].real, fmt="% 6.4f", delimiter=", ")

                    assert 1 == 2


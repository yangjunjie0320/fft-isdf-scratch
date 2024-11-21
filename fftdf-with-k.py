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

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 160000))

def gen_interpolating_points(c, gmesh=None, cisdf=10.0):
    if gmesh is None:
        gmesh = [15, 15, 15]

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

def get_coul(c, ke=None, kmesh=None, cisdf=10.0, gmesh=None):
    if ke is None:
        ke = c.ke_cutoff

    if kmesh is None:
        kmesh = [1, 1, 1]

    from pyscf.pbc.tools.k2gamma import get_phase
    vk = c.get_kpts(kmesh)
    sc, phase = get_phase(c, vk, kmesh=kmesh, wrap_around=False)

    c.verbose = 100
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

    # k1 = max(0.01 * ke, 8)
    # gmesh = pbctools.cutoff_to_mesh(l, k1)
    # gmesh = [9, 9, 9]
    ng = numpy.prod(gmesh)
    print(f"{ng = }, {gmesh = }")

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
    gw = 1.0 # c.vol / ng

    # select the interpolating points in the unit cell
    x0 = c.pbc_eval_gto("GTOval_sph", gx)
    x0 = numpy.sqrt(gw) * numpy.asarray(x0)
    assert x0.shape == (ng, nao)
    x4 = (numpy.dot(x0, x0.T)) ** 2

    from pyscf.lib.scipy_helper import pivoted_cholesky
    chol, perm, rank = pivoted_cholesky(x4, tol=1e-32)
    # nip = min(rank, cisdf * nao)
    # nip = 400 # int(cisdf * nao)
    nip = int(ng * cisdf)

    approx_error = chol[nip, nip]

    print(f"{nip = }, {rank = }, {approx_error = :6.2e}")

    mask = perm[:nip]
    x_k = c.pbc_eval_gto("GTOval_sph", gx[mask], kpts=vk)
    x_k = numpy.sqrt(gw) * numpy.array(x_k)
    assert x_k.shape == (nkpt, nip, nao)

    # x_s = einsum("Rk,kIm->RIm", phase, x_k)
    x_s = phase @ x_k.reshape(nkpt, -1)
    x_s = x_s.reshape(nimg, nip, nao)
    assert abs(x_s.imag).max() < 1e-10

    x_f = einsum("kIm,Rk,Sk->RISm", x_k, phase, phase.conj())
    assert x_f.shape == (nimg, nip, nimg, nao)
    assert abs(x_f.imag).max() < 1e-10
    x_f = x_f.reshape(nimg * nip, nimg * nao)

    # x2_f_ref = einsum("RISm,TJSm->RITJ", x_f, x_f)
    # x2_f_ref = x2_f_ref.reshape(nimg * nip, nimg * nip)

    # x2_k = einsum("kIm,kJm->kIJ", x_k.conj(), x_k)
    x2_k = numpy.asarray([xq.conj() @ xq.T for xq in x_k])
    assert x2_k.shape == (nkpt, nip, nip)

    # x2_f = einsum("kIJ,Rk,Sk->RISJ", x2_k, phase, phase.conj())
    # x2_f_sol = x2_f.reshape(nimg * nip, nimg * nip)

    # assert abs(x2_f_sol - x2_f_ref).max() < 1e-10, f"x2_f_sol is not equal to x2_f_ref"

    # assert abs(x2_f_sol.imag).max() < 1e-10, f"imaginary part of x2_f_sol is not zero, {abs(x2_f_sol).max() = }"

    # x2_s = einsum("Rk,kIJ->RIJ", phase, x2_k)
    x2_s = phase @ x2_k.reshape(nkpt, -1)
    x2_s = x2_s.reshape(nimg, nip, nip)
    assert abs(x2_s.imag).max() < 1e-10

    # x2_s = einsum("Rk,kIJ->RIJ", phase, x2_k)
    # assert abs(x2_s).max() < 1e-10, f"imaginary part of x2_s is not zero, {abs(x2_s).max() = }"

    x4_s = x2_s * x2_s
    # x4_k = einsum("Rk,RIJ->kIJ", phase.conj(), x4_s)
    x4_k = phase.conj().T @ x4_s.reshape(nimg, -1)
    x4_k = x4_k.reshape(nkpt, nip, nip)
    assert x4_k.shape == (nkpt, nip, nip)

    x4inv_k = [scipy.linalg.pinv(x4_k[q]) for q in range(nkpt)]
    x4inv_k = numpy.asarray(x4inv_k)

    from pyscf.pbc.df.fft import FFTDF
    df_obj = FFTDF(c)
    df_obj.mesh = c.mesh
    # df_obj.ke_cutoff = ke
    grids = df_obj.grids
    coord = grids.coords
    weigh = grids.weights
    ngrid = coord.shape[0]

    mesh = df_obj.mesh
    assert ngrid == numpy.prod(mesh)

    # the size of the dataset is nkpt * ngrid * nip * 16 bytes
    required_memory = ngrid * nip * 16 / 1e6
    log.info("ngrid = %d, nip = %d", ngrid, nip)
    log.info("Required disk space = %d MB", nkpt * required_memory)
    log.info("Required memory = %d MB", required_memory)

    from pyscf.lib import H5TmpFile
    fswp = H5TmpFile()
    fswp.create_dataset("z", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
    # z = numpy.empty((nkpt, ngrid, nip), dtype=numpy.complex128)
    z = fswp["z"]
    log.info("finished creating fswp: %s", fswp.filename)
    
    # for ao_ks_etc, p0, p1 in df_obj.aoR_loop(grids, vk):
    from pyscf.lib.logger import process_clock, perf_counter
    ni = df_obj._numint
    nao = c.nao_nr()
    p0, p1 = 0, 0
    for ao_k_etc in ni.block_loop(c, grids, nao, deriv=0, kpts=vk, blksize=16000):
        f_k = numpy.asarray(ao_k_etc[0])
        p0, p1 = p1, p1 + f_k.shape[1]
        assert f_k.shape == (nkpt, p1 - p0, nao)
        log.info("\naoR_loop[%6d:%6d] start", p0, p1)

        # fx_k = einsum("kgm,kIm->kgI", f_k.conj(), x_k)
        t0 = (process_clock(), perf_counter())
        fx_k = numpy.asarray([fq.conj() @ xq.T for fq, xq in zip(f_k, x_k)])
        assert fx_k.shape == (nkpt, p1 - p0, nip)
        t1 = log.timer("fx_k method 1", *t0)

        # t0 = (process_clock(), perf_counter())
        # fx_k = einsum("kgm,kIm->kgI", f_k.conj(), x_k)
        # t1 = log.timer("fx_k method 2", *t0)

        # fx_s = einsum("Rk,kgI->RgI", phase, fx_k)
        fx_s = phase @ fx_k.reshape(nkpt, -1)
        fx_s = fx_s.reshape(nimg, p1 - p0, nip)
        assert abs(fx_s.imag).max() < 1e-10
        log.info("fx_s[%6d:%6d] done", p0, p1) 

        y_s = fx_s * fx_s
        # y_k = einsum("RgI,Rk->kgI", y_s, phase)
        y_k = phase.T @ y_s.reshape(nimg, -1)
        y_k = y_k.reshape(nkpt, p1 - p0, nip)
        assert y_k.shape == (nkpt, p1 - p0, nip)
        log.info("y_k[%6d:%6d] done", p0, p1)
        
        # print(f"{y_k.shape = }, {x4inv_k.shape = }")
        # fswp["z"][:, p0:p1, :] = einsum("kgJ,kIJ->kgI", y_k, x4inv_k)
        # z[:, p0:p1, :] = einsum("kgJ,kIJ->kgI", y_k, x4inv_k)
        t0 = (process_clock(), perf_counter())
        z[:, p0:p1, :] = numpy.asarray([yq @ xinvq.T for yq, xinvq in zip(y_k, x4inv_k)])
        t1 = log.timer("z method 1", *t0)

        # t0 = (process_clock(), perf_counter())
        # z[:, p0:p1, :] = einsum("kgJ,kIJ->kgI", y_k, x4inv_k)
        # t1 = log.timer("z method 2", *t0)

        log.info("aoR_loop[%6d:%6d] done", p0, p1)

    coul_k = []
    gv = c.get_Gv(mesh)

    for q, vq in enumerate(vk):
        phase = numpy.exp(-1j * numpy.dot(coord, vq))
        assert phase.shape == (ngrid, )

        coul_q = pbctools.get_coulG(c, k=vq, mesh=mesh, Gv=gv) * c.vol / ngrid
        assert coul_q.shape == (ngrid, )
        log.info("\ncoul_q[%d] done", q)
        
        z_q = z[q, :, :].T
        # zeta_g  = pbctools.fft(fswp["z"][q, :, :].T * phase, mesh) 
        zeta_q = pbctools.fft(z_q * phase, mesh)
        zeta_q *= coul_q
        assert zeta_q.shape == (nip, ngrid)
        log.info("zeta_q[%d] done", q)

        zeta_q = pbctools.ifft(zeta_q, mesh)
        zeta_q *= phase.conj()
        log.info("zeta_q[%d] done", q)

        coul_k.append(zeta_q @ z_q.conj().T)
        # coul_k.append(einsum("Ig,Jg->IJ", zeta_q, z_q.conj()))
        log.info("coul_k[%d] done", q)

    coul_k = numpy.asarray(coul_k)
    assert coul_k.shape == (nkpt, nip, nip)
    log.info("coul_k done")
    return coul_k, x_k

if __name__ == "__main__":
    cell   = pyscf.pbc.gto.Cell()
    cell.a = numpy.ones((3, 3)) * 3.5668 - numpy.eye(3) * 3.5668
    cell.atom = '''C     0.0000  0.0000  0.0000
                C     0.8917  0.8917  0.8917 '''
    cell.basis  = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.ke_cutoff = 100
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.build(dump_input=False)

    print(f"{cell.ke_cutoff = }, {cell.mesh = }")

    cell.verbose = 5
    nao = cell.nao_nr()

    kmesh = [4, 4, 4]
    nkpt = nimg = numpy.prod(kmesh)
    c, x = get_coul(cell, ke=cell.ke_cutoff, kmesh=kmesh, cisdf=0.9, gmesh=[11, 11, 11])
    nkpt, nip, nao = x.shape
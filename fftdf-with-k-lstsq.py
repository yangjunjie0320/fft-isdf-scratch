import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum
import scipy.linalg

import pyscf
from pyscf.lib import logger, current_memory
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc import tools as pbctools
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 160000))

# import line_profiler
# @line_profiler.profile
def get_coul(df_obj, k0=10.0, kmesh=None, cisdf=0.6, verbose=5, blksize=16000):
    log = logger.new_logger(df_obj, verbose)
    cell = df_obj.cell

    if kmesh is None:
        kmesh = [1, 1, 1]

    from pyscf.pbc.tools.k2gamma import get_phase
    vk = df_obj.cell.get_kpts(kmesh)
    sc, phase = get_phase(df_obj.cell, vk, kmesh=kmesh, wrap_around=False)

    lv = df_obj.cell.lattice_vectors()
    gmesh = pbctools.cutoff_to_mesh(lv, k0)
    ng = numpy.prod(gmesh)

    log.info("Parent grid: gmesh = %s, ng = %d", gmesh, ng)

    max_memory = cell.max_memory - current_memory()[0]
    required_memory = ng * ng * 16 / 1e6
    log.info("Max memory = %d MB, required memory = %d MB.", max_memory, required_memory)

    if max_memory < required_memory:
        info  = "Max memory = %d MB is not enough.\n" % max_memory
        info += "Required memory = %d MB." % required_memory
        raise RuntimeError(info)

    nao = cell.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    gx = cell.gen_uniform_grids(gmesh)
    x4 = (lambda x: (x @ x.T) ** 2)(cell.pbc_eval_gto("GTOval", gx))

    from pyscf.lib.scipy_helper import pivoted_cholesky
    chol, perm, rank = pivoted_cholesky(x4, tol=1e-32)
    nip = int(ng * cisdf)
    log.info("nip = %d", nip)

    mask = perm[:nip]
    x_k = cell.pbc_eval_gto("GTOval", gx[mask], kpts=vk)
    x_k = numpy.array(x_k)
    nip = x_k.shape[1]
    assert x_k.shape == (nkpt, nip, nao)

    x_s = phase @ x_k.reshape(nkpt, -1)
    x_s = x_s.reshape(nimg, nip, nao)
    assert abs(x_s.imag).max() < 1e-10

    # x_f = einsum("kIm,Rk,Sk->RISm", x_k, phase, phase.conj())
    # assert x_f.shape == (nimg, nip, nimg, nao)
    # assert abs(x_f.imag).max() < 1e-10
    # x_f = x_f.reshape(nimg * nip, nimg * nao)

    x2_k = numpy.asarray([xq.conj() @ xq.T for xq in x_k])
    assert x2_k.shape == (nkpt, nip, nip)

    x2_s = phase @ x2_k.reshape(nkpt, -1)
    x2_s = x2_s.reshape(nimg, nip, nip)
    assert abs(x2_s.imag).max() < 1e-10

    x4_s = x2_s * x2_s
    x4_k = phase.conj().T @ x4_s.reshape(nimg, -1)
    x4_k = x4_k.reshape(nkpt, nip, nip)
    assert x4_k.shape == (nkpt, nip, nip)

    ip = set()
    # chol = []
    for q in range(nkpt):
        t0 = (process_clock(), perf_counter())
        from scipy.linalg import lapack
        res = lapack.zpstrf(x4_k[q], lower=False)

        chol = res[0]
        chol[numpy.tril_indices(nip, k=-1)] *= 0.0

        rank = res[2]
        # perm = (res[1] - 1)[:rank]
        perm = numpy.zeros((nip, nip))
        perm[res[1]-1, numpy.arange(nip)] = 1

        for ind in perm:
            ip.add(ind)

        tmp = res[0]
        tmp[numpy.tril_indices(nip, k=-1)] = 0
        chol.append(tmp)

        t1 = log.timer("zpstrf[%2d]" % q, *t0)

    for q in range(nkpt):
        perm = ip
        chol[q][chol[pe]]
    
    ip = list(sorted(ip))
    log.info("Selected interpolation points pruned: %d -> %d", nip, len(ip))
    nip = len(ip)
    assert 1 == 2

    x_k = x_k[:, ip, :]

    x4_k = [x4_k[q][ip][:, ip] for q in range(nkpt)]
    x4_k = numpy.asarray(x4_k)
    assert x4_k.shape == (nkpt, nip, nip)

    t0 = (process_clock(), perf_counter())

    grids = df_obj.grids
    coord = grids.coords
    ngrid = coord.shape[0]

    required_disk_space = nkpt * ngrid * nip * 16 / 1e9
    log.info("nkpt = %d, ngrid = %d, nip = %d", nkpt, ngrid, nip)
    log.info("Required disk space = %d GB", required_disk_space)

    from pyscf.lib import H5TmpFile
    fswp = H5TmpFile()
    fswp.create_dataset("y", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
    # z = fswp["y"]
    y = fswp["y"]
    log.debug("finished creating fswp: %s", fswp.filename)
    
    # compute the memory required for the aoR_loop
    required_memory = blksize * nip * nkpt * 16 / 1e6
    log.info("Required memory = %d MB", required_memory)
    
    ni = df_obj._numint
    p0, p1 = 0, 0
    
    t0 = (process_clock(), perf_counter())
    for ao_k_etc in ni.block_loop(cell, grids, nao, deriv=0, kpts=vk, blksize=blksize):
        f_k = numpy.asarray(ao_k_etc[0])
        p0, p1 = p1, p1 + f_k.shape[1]
        assert f_k.shape == (nkpt, p1 - p0, nao)

        fx_k = numpy.asarray([f.conj() @ x.T for f, x in zip(f_k, x_k)])
        assert fx_k.shape == (nkpt, p1 - p0, nip)

        fx_s = phase @ fx_k.reshape(nkpt, -1)
        fx_s = fx_s.reshape(nimg, p1 - p0, nip)
        assert abs(fx_s.imag).max() < 1e-10

        y_s = fx_s * fx_s
        y_k = phase.T @ y_s.reshape(nimg, -1)
        # y_k = y_k.reshape(nkpt, p1 - p0, nip)
        # assert y_k.shape == (nkpt, p1 - p0, nip)
        y[:, p0:p1, :] = y_k.reshape(nkpt, p1 - p0, nip)

        # z[:, p0:p1, :] = numpy.asarray([yq @ xinvq.T for yq, xinvq in zip(y_k, x4inv_k)])
        log.debug("finished aoR_loop[%8d:%8d]", p0, p1)

    t1 = log.timer("building z", *t0)

    mesh = df_obj.mesh
    gv = cell.get_Gv(mesh)
    
    required_memory = nip * ngrid * 16 / 1e6
    log.info("Required memory = %d MB", required_memory)

    coul_q = []
    for q, vq in enumerate(vk):
        t0 = (process_clock(), perf_counter())
        phase = numpy.exp(-1j * numpy.dot(coord, vq))
        assert phase.shape == (ngrid, )
        
        y_q = y[q, :, :].T
        assert y_q.shape == (nip, ngrid)

        x4_q = x4_k[q]
        assert x4_q.shape == (nip, nip)

        # z_q = y_q
        res = scipy.linalg.lstsq(x4_q, y_q) # , lapack_driver="gelsy")
        z_q = res[0]
        rank = res[2]
        assert z_q.shape == (nip, ngrid)
        
        zeta_q = pbctools.fft(z_q * phase, mesh)
        zeta_q *= pbctools.get_coulG(cell, k=vq, mesh=mesh, Gv=gv)
        zeta_q *= cell.vol / ngrid
        assert zeta_q.shape == (nip, ngrid)

        zeta_q = pbctools.ifft(zeta_q, mesh)
        zeta_q *= phase.conj()

        j_q = zeta_q @ z_q.conj().T
        # w_q = t_q @ j_q @ t_q.conj().T
        w_q = j_q
        coul_q.append(w_q)

        t1 = log.timer("coul[%2d], rank = %d / %d" % (q, rank, nip), *t0)

    coul_q = numpy.asarray(coul_q)
    assert coul_q.shape == (nkpt, nip, nip)
    return coul_q, x_k

if __name__ == "__main__":
    cell   = pyscf.pbc.gto.Cell()
    cell.a = numpy.ones((3, 3)) * 3.5668 - numpy.eye(3) * 3.5668
    cell.atom = '''C     0.0000  0.0000  0.0000
                C     0.8917  0.8917  0.8917 '''
    cell.basis  = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.ke_cutoff = 50
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.build(dump_input=False)

    from pyscf.pbc.df.fft import FFTDF
    df_obj = FFTDF(cell)
    df_obj.mesh = [15, 15, 15]
    df_obj.verbose = 5

    # kmesh = [4, 4, 4]
    kmesh = [2, 2, 2]
    nkpt = nimg = numpy.prod(kmesh)
    c, x = get_coul(df_obj, kmesh=kmesh, k0=20.0, cisdf=0.9, blksize=1000)
    nkpt, nip, nao = x.shape

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

                eri_ref = df_obj.get_eri(kpts=[vk1, vk2, vk3, vk4], compact=False)
                eri_ref = eri_ref.reshape(nao * nao, nao * nao)

                x1, x2, x3, x4 = [x[k] for k in [k1, k2, k3, k4]]
                eri_sol = einsum("IJ,Im,In,Jk,Jl->mnkl", c[q], x1.conj(), x2, x3.conj(), x4)
                eri_sol = eri_sol.reshape(nao * nao, nao * nao)

                eri = abs(eri_sol - eri_ref).max()
                print(f"{k1 = :2d}, {k2 = :2d}, {k3 = :2d}, {k4 = :2d} {eri = :6.2e}")

                if eri > 1e-4:

                    print(" q = %2d, vq  = [%s]" % (q, ", ".join(f"{v: 6.4f}" for v in vq)))
                    print("k1 = %2d, vk1 = [%s]" % (k1, ", ".join(f"{v: 6.4f}" for v in vk1)))
                    print("k2 = %2d, vk2 = [%s]" % (k2, ", ".join(f"{v: 6.4f}" for v in vk2)))
                    print("k3 = %2d, vk3 = [%s]" % (k3, ", ".join(f"{v: 6.4f}" for v in vk3)))
                    print("k4 = %2d, vk4 = [%s]" % (k4, ", ".join(f"{v: 6.4f}" for v in vk4)))

                    print(f"\n{eri_sol.shape = }")
                    numpy.savetxt(cell.stdout, eri_sol[:10, :10].real, fmt="% 6.4e", delimiter=", ")

                    print(f"\neri_sol.imag = ")
                    numpy.savetxt(cell.stdout, eri_sol[:10, :10].imag, fmt="% 6.4e", delimiter=", ")

                    print(f"\n{eri_ref.shape = }")
                    numpy.savetxt(cell.stdout, eri_ref[:10, :10].real, fmt="% 6.4e", delimiter=", ")

                    print(f"\neri_ref.imag = ")
                    numpy.savetxt(cell.stdout, eri_ref[:10, :10].imag, fmt="% 6.4e", delimiter=", ")

                    assert 1 == 2

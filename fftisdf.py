import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum
import scipy.linalg

import pyscf
from pyscf.lib import logger, current_memory
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import tools as pbctools
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 160000))

def build(df_obj):
    log = logger.new_logger(df_obj, df_obj.verbose)
    pcell = df_obj.cell

    from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
    kmesh = kpts_to_kmesh(pcell, df_obj.kpts)
    log.debug("Transform kpts to kmesh")
    log.debug("original    kpts  = %s", df_obj.kpts)
    log.debug("transformed kmesh = %s", kmesh)

    from pyscf.pbc.tools.k2gamma import get_phase
    vk = pcell.get_kpts(kmesh)
    scell, phase = get_phase(pcell, vk, kmesh=kmesh, wrap_around=False)
    log.debug("transformed kpts = %s", vk)

    nao = pcell.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    xip = df_obj.select_interpolation_points()
    nip = xip.shape[0]
    
    x_k = pcell.pbc_eval_gto("GTOval", xip, kpts=vk)
    x_k = numpy.array(x_k)
    assert x_k.shape == (nkpt, nip, nao)
    log.info("Number of interpolation points = %d", nip)

    x_s = phase @ x_k.reshape(nkpt, -1)
    x_s = x_s.reshape(nimg, nip, nao)
    assert abs(x_s.imag).max() < 1e-10

    x2_k = numpy.asarray([xq.conj() @ xq.T for xq in x_k])
    assert x2_k.shape == (nkpt, nip, nip)

    x2_s = phase @ x2_k.reshape(nkpt, -1)
    x2_s = x2_s.reshape(nimg, nip, nip)
    assert abs(x2_s.imag).max() < 1e-10

    x4_s = x2_s * x2_s
    x4_k = phase.conj().T @ x4_s.reshape(nimg, -1)
    x4_k = x4_k.reshape(nkpt, nip, nip)
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
    # for ao_k_etc in ni.block_loop(cell, grids, nao, deriv=0, kpts=vk, blksize=blksize):
    #     f_k = numpy.asarray(ao_k_etc[0])
    #     p0, p1 = p1, p1 + f_k.shape[1]
    #     assert f_k.shape == (nkpt, p1 - p0, nao)

    #     fx_k = numpy.asarray([f.conj() @ x.T for f, x in zip(f_k, x_k)])
    #     assert fx_k.shape == (nkpt, p1 - p0, nip)

    #     fx_s = phase @ fx_k.reshape(nkpt, -1)
    #     fx_s = fx_s.reshape(nimg, p1 - p0, nip)
    #     assert abs(fx_s.imag).max() < 1e-10

    #     y_s = fx_s * fx_s
    #     y_k = phase.T @ y_s.reshape(nimg, -1)
    #     # y_k = y_k.reshape(nkpt, p1 - p0, nip)
    #     # assert y_k.shape == (nkpt, p1 - p0, nip)
    #     y[:, p0:p1, :] = y_k.reshape(nkpt, p1 - p0, nip)

    #     # z[:, p0:p1, :] = numpy.asarray([yq @ xinvq.T for yq, xinvq in zip(y_k, x4inv_k)])
    #     log.debug("finished aoR_loop[%8d:%8d]", p0, p1)
    df_obj._solve_y(x_k, x4_k, fswp=fswp, phase=phase)

    t1 = log.timer("building z", *t0)

    mesh = df_obj.mesh
    gv = pcell.get_Gv(mesh)
    
    required_memory = nip * ngrid * 16 / 1e6
    log.info("Required memory = %d MB", required_memory)

    # coul_q = []
    # for q, vq in enumerate(vk):
    #     t0 = (process_clock(), perf_counter())
    #     phase = numpy.exp(-1j * numpy.dot(coord, vq))
    #     assert phase.shape == (ngrid, )
        
    #     y_q = y[q, :, :]
    #     assert y_q.shape == (ngrid, nip)

    #     x4_q = x4_k[q]
    #     assert x4_q.shape == (nip, nip)

    #     res = scipy.linalg.lstsq(x4_q, y_q.T, lapack_driver="gelsy")
    #     z_q = res[0]
    #     rank = res[2]

    #     # res = scipy.linalg.pinvh(x4_q, return_rank=True)
    #     # t_q = res[0]
    #     # rank = res[1]
    #     # z_q = t_q @ y_q.T

    #     assert z_q.shape == (nip, ngrid)
        
    #     # z_q = z[q, :, :].T
    #     zeta_q = pbctools.fft(z_q * phase, mesh)
    #     zeta_q *= pbctools.get_coulG(cell, k=vq, mesh=mesh, Gv=gv)
    #     zeta_q *= cell.vol / ngrid
    #     assert zeta_q.shape == (nip, ngrid)

    #     zeta_q = pbctools.ifft(zeta_q, mesh)
    #     zeta_q *= phase.conj()

    #     coul_q.append(zeta_q @ z_q.conj().T)
    #     t1 = log.timer("coul[%2d], rank = %d / %d" % (q, rank, nip), *t0)
    df_obj._solve_z(x_k, x4_k, fswp=fswp, phase=phase)

    df_obj.z = fswp["z"]
    df_obj.x = x_k

class InterpolativeSeparableDensityFitting(FFTDF):
    k0 = 8.0 # cutoff kinetic energy for the parent grid
    r0 = 0.8 # rate of the interpolation points to the parent grid
    
    blksize = 8000 # block size for the aoR_loop

    def build(self):
        pass
    
    def select_interpolation_points(self, k0=None, r0=None):
        if k0 is None: k0 = self.k0
        if r0 is None: r0 = self.r0

        pcell = self.cell
        nao = pcell.nao_nr()

        # from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
        # kmesh = kpts_to_kmesh(pcell, self.kpts)
        # from pyscf.pbc.tools.k2gamma import get_phase
        # vk = pcell.get_kpts(kmesh)
        # scell, phase = get_phase(pcell, vk, kmesh=kmesh, wrap_around=False)
        
        lv = pcell.lattice_vectors()
        mg = pbctools.cutoff_to_mesh(lv, k0)
        ng = numpy.prod(mg)

        xg = pcell.gen_uniform_grids(mg)
        f  = pcell.pbc_eval_gto("GTOval", xg)
        assert f.shape == (ng, nao)

        f4 = (f @ f.T) ** 2
        assert f4.shape == (ng, ng)

        from pyscf.lib.scipy_helper import pivoted_cholesky
        chol, perm, rank = pivoted_cholesky(f4, tol=1e-32)
        nip = int(ng * r0)
        mask = perm[:nip]
        return xg[mask]

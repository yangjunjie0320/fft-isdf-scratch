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
    log.debug("\nTransform kpts to kmesh")
    log.debug("original    kpts  =\n %s", df_obj.kpts)
    log.debug("transformed kmesh = %s", kmesh)

    from pyscf.pbc.tools.k2gamma import get_phase
    vk = pcell.get_kpts(kmesh)
    scell, phase = get_phase(pcell, vk, kmesh=kmesh, wrap_around=False)
    log.debug("transformed kpts =\n %s", vk)

    nao = pcell.nao_nr()
    nkpt = nimg = numpy.prod(kmesh)

    xip = df_obj.select_interpolation_points()
    nip = xip.shape[1]
    assert xip.shape == (nkpt, nip, nao)
    log.info("Number of interpolation points = %d", nip)

    x2_k = numpy.asarray([xq.conj() @ xq.T for xq in xip])
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
    fswap = H5TmpFile()
    fswap.create_dataset("y", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
    y = fswap["y"]
    log.debug("finished creating fswp: %s", fswap.filename)
    
    # compute the memory required for the aoR_loop
    blksize = df_obj.blksize
    required_memory = blksize * nip * nkpt * 16 / 1e6
    log.info("Required memory = %d MB", required_memory)
    
    ni = df_obj._numint
    g0g1 = range(0, ngrid, blksize)
    
    t0 = (process_clock(), perf_counter())
    block_loop = ni.block_loop(pcell, grids, nao, deriv=0, kpts=vk, blksize=blksize)
    for ig, ao_k_etc in enumerate(block_loop):
        g0, g1 = g0g1[ig:ig+2]
        f_k = numpy.asarray(ao_k_etc[0])
        assert f_k.shape == (nkpt, g1 - g0, nao)

        fx_k = numpy.asarray([f.conj() @ x.T for f, x in zip(f_k, xip)])
        assert fx_k.shape == (nkpt, g1 - g0, nip)

        fx_s = phase @ fx_k.reshape(nkpt, -1)
        fx_s = fx_s.reshape(nimg, g1 - g0, nip)
        assert abs(fx_s.imag).max() < 1e-10

        y_s = fx_s * fx_s
        y_k = phase.T @ y_s.reshape(nimg, -1)
        y[:, g0:g1, :] = y_k.reshape(nkpt, g1 - g0, nip)

        log.debug("finished aoR_loop[%8d:%8d]", g0, g1)
    t1 = log.timer("building y", *t0)

    mesh = df_obj.mesh
    gv = pcell.get_Gv(mesh)
    
    required_memory = nip * ngrid * 16 / 1e6
    log.info("Required memory = %d MB", required_memory)

    coul_q = []
    for q, vq in enumerate(vk):
        t0 = (process_clock(), perf_counter())
        phase = numpy.exp(-1j * numpy.dot(coord, vq))
        assert phase.shape == (ngrid, )
        
        y_q = y[q, :, :]
        assert y_q.shape == (ngrid, nip)

        x4_q = x4_k[q]
        assert x4_q.shape == (nip, nip)

        res = scipy.linalg.lstsq(x4_q, y_q.T, lapack_driver="gelsy")
        z_q = res[0]
        rank = res[2]
        assert z_q.shape == (nip, ngrid)
        
        # z_q = z[q, :, :].T
        zeta_q = pbctools.fft(z_q * phase, mesh)
        zeta_q *= pbctools.get_coulG(pcell, k=vq, mesh=mesh, Gv=gv)
        zeta_q *= pcell.vol / ngrid
        assert zeta_q.shape == (nip, ngrid)

        zeta_q = pbctools.ifft(zeta_q, mesh)
        zeta_q *= phase.conj()

        coul_q.append(zeta_q @ z_q.conj().T)
        t1 = log.timer("coul[%2d], rank = %d / %d" % (q, rank, nip), *t0)

    return xip, coul_q

class InterpolativeSeparableDensityFitting(FFTDF):
    blksize = 8000 # block size for the aoR_loop

    def __init__(self, cell, kpts, m0=None, c0=20.0):
        super().__init__(cell, kpts)

        self.m0 = m0 if m0 is not None else [15, 15, 15]
        self.c0 = c0

    def build(self):
        return build(self)
    
    def select_interpolation_points(self, x0=None, phase=None):
        c0 = self.c0
        m0 = self.m0

        # the primitive cell
        pcell = self.cell
        nao = pcell.nao_nr()

        x0 = pcell.pbc_eval_gto(
            "GTOval", pcell.gen_uniform_grids(m0), 
            kpts=self.kpts
        )
        x0 = numpy.asarray(x0)

        nkpt, ng = x0.shape[:2]
        assert x0.shape == (nkpt, ng, nao)

        x2 = numpy.zeros((ng, ng), dtype=numpy.double)
        for q in range(nkpt):
            x2 += (x0[q].conj() @ x0[q].T).real
        x4 = (x2 * x2 / nkpt).real

        from pyscf.lib.scipy_helper import pivoted_cholesky
        chol, perm, rank = pivoted_cholesky(x4)
        nip = min(int(nao * c0), rank)
        mask = perm[:nip]
        return x0[:, mask, :]
    
ISDF = InterpolativeSeparableDensityFitting

if __name__ == "__main__":
    from ase.build import bulk
    atoms = bulk("NiO", "rocksalt", a=4.18)

    from pyscf.pbc.gto import Cell
    from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf

    cell = Cell()
    cell.atom = ase_atoms_to_pyscf(atoms)
    cell.a = numpy.array(atoms.cell)
    cell.basis = 'gth-szv-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.ke_cutoff = 200
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.build(dump_input=False)

    from pyscf.pbc.df.fft import FFTDF
    df_obj = FFTDF(cell)

    # kmesh = [4, 4, 4]
    kmesh = [2, 2, 2]
    nkpt = nimg = numpy.prod(kmesh)

    isdf = ISDF(cell, kpts=cell.get_kpts(kmesh))
    xip, coul_q = isdf.build()

import itertools, os, sys
from itertools import product

import numpy, scipy
from opt_einsum import contract as einsum

import pyscf
from pyscf.pbc import tools
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

c   = pyscf.pbc.gto.Cell()
c.a = numpy.eye(3) * 3.5668
c.atom = '''C     0.0000  0.0000  0.0000
            C     0.8917  0.8917  0.8917
            C     1.7834  1.7834  0.0000
            C     2.6751  2.6751  0.8917
            C     1.7834  0.0000  1.7834
            C     2.6751  0.8917  2.6751
            C     0.0000  1.7834  1.7834
            C     0.8917  2.6751  2.6751'''
c.basis  = 'gth-szv'
c.pseudo = 'gth-pade'
c.verbose = 0
c.unit = 'aa'
c.max_memory = 100
c.build()

nao = c.nao_nr()
print(f"{nao = }")

from pyscf.pbc.df.fft import FFTDF
df = FFTDF(c)
df.max_memory = 4
gmesh = c.mesh

from pyscf.pbc.dft.gen_grid import gen_uniform_grids, gen_becke_grids
coord = gen_uniform_grids(c, mesh=[16, 16, 16], wrap_around=False)
weigh = c.vol / coord.shape[0]
print(f"{coord.shape = }, {weigh.shape = }")

# coord, weigh = gen_becke_grids(c, level=1)
# print(f"{coord.shape = }, {weigh.shape = }")

phi  = numpy.sqrt(weigh) * c.pbc_eval_gto("GTOval_sph", coord)
phi  = phi.T
nao, ng = phi.shape
print(f"{ng = }, {nao = }")

ovlp_sol = einsum("mg,ng->mn", phi, phi)
ovlp_ref = c.pbc_intor('int1e_ovlp', hermi=0)
print(f"{ovlp_sol.shape = }")
numpy.savetxt(c.stdout, ovlp_sol, fmt="% 6.4f", delimiter=", ")
print(f"{ovlp_ref.shape = }")
numpy.savetxt(c.stdout, ovlp_ref, fmt="% 6.4f", delimiter=", ")
err = abs(ovlp_ref - ovlp_sol).max()
print(f"{err = }")
assert 1 == 2

from pyscf.lib.scipy_helper import pivoted_cholesky
zeta = einsum("mI,nI,mJ,nJ->IJ", phi, phi, phi, phi)
chol, perm, rank = pivoted_cholesky(zeta, tol=1e-30)
mask = perm[:rank]
print(f"{mask.shape = }")

rho_sol = einsum("gm,gn->gmn", phi, phi)
rho_ref = einsum("mI,nI,gI->gmn", phi[mask], phi[mask], zeta)
err = abs(rho_ref - rho_sol).max()
print(f"{err = }")
assert 1 == 2

# build the supercell
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import super_cell
from pyscf.pbc.tools.pbc import _build_supcell_

nimg = [2, 2, 2]
tv = k2gamma.translation_vectors_for_kmesh(c, nimg, wrap_around=False)
print(f"{tv.shape = }")
sc = c.copy(deep=False)
sc.a = numpy.einsum("i,ij->ij", nimg, c.a)
mesh = numpy.asarray(nimg) * numpy.asarray(c.mesh)
sc.mesh = (mesh // 2) * 2 + 1
_build_supcell_(sc, c, tv)

df_sc = FFTDF(sc)
df_sc.max_memory = 2000

phase = numpy.exp(1j * numpy.einsum("g,ij->gi", coord, c.reciprocal_vectors()))
print(f"{phase.shape = }")

coord = coord[mask]
coord = coord[None, :, :] + tv[:, None, :]
coord = coord.reshape(-1, 3)

phi = numpy.sqrt(weigh) * c.pbc_eval_gto("GTOval_sph", coord)
phi = phi.T

for ao, p0, p1 in df_sc.aoR_loop(deriv=0):
    print(f"{p0 = }, {p1 = }")
    weigh = ao[3]
    coord = ao[4]   
    ng = weigh.size
    assert ao[0][0].shape == (ng, nao)

    rhs = einsum("gm,gn,mI,nI->gI", ao[0][0], ao[0][0], phi, phi)
    print(f"{rhs.shape = }")

    res = scipy.linalg.lstsq(zeta[mask][:, mask], rhs.T)
    z = res[0].T
    resid = res[1]
    rank = res[2]
    print(f"{z.shape = }, {rank = }, {resid = }")

    resid = zeta[mask][:, mask] @ z.T - rhs.T
    print(f"{resid.shape = }, {resid.max() = }")

    rho_ref = einsum("gm,gn->gmn", ao[0][0], ao[0][0])
    print(f"{rho_ref.shape = }, {rho_ref.max() = }")

    rho_sol = einsum("gI,mI,nI->gmn", z, phi[:, mask], phi[:, mask])
    print(f"{rho_sol.shape = }, {rho_sol.max() = }")

    err = abs(rho_ref - rho_sol).max()
    print(f"{err = }")


# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Unrestricted algebraic diagrammatic construction
'''

import numpy as np
from pyscf import lib, symm
from pyscf.lib import logger
from pyscf.adc import uadc
from pyscf.adc import uadc_ao2mo, uadc_amplitudes
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
from pyscf import scf
from pyscf.data import nist

##@profile
def get_imds(adc, eris=None):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    t1 = adc.t1
    t2 = adc.t2

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    n_singles_a = nocc_a * nvir_a
    dim_a = int(n_singles_a)
    n_singles_b = nocc_b * nvir_b
    dim_b = int(n_singles_b)
    M_ia_jb_a = np.zeros((dim_a, dim_a))
    M_ia_jb_b = np.zeros((dim_b, dim_b))

    if eris is None:
        eris = adc.transform_integrals()

    d_ai_a = adc.mo_energy_a[nocc_a:][:,None] - adc.mo_energy_a[:nocc_a]
    np.fill_diagonal(M_ia_jb_a, d_ai_a.transpose().reshape(-1))
    M_ia_jb_a = M_ia_jb_a.reshape(nocc_a,nvir_a,nocc_a,nvir_a).copy()

    d_ai_b = adc.mo_energy_b[nocc_b:][:,None] - adc.mo_energy_b[:nocc_b]
    np.fill_diagonal(M_ia_jb_b, d_ai_b.transpose().reshape(-1))

    M_ia_jb_b = M_ia_jb_b.reshape(nocc_b,nvir_b,nocc_b,nvir_b).copy()
    M_ia_jb_a -= lib.einsum('ijba->iajb', eris.oovv, optimize = True).copy()
    M_ia_jb_a += lib.einsum('jbai->iajb', eris.ovvo, optimize = True)
    M_ia_jb_b -= lib.einsum('ijba->iajb', eris.OOVV, optimize = True).copy()
    M_ia_jb_b += lib.einsum('jbai->iajb', eris.OVVO, optimize = True)
    M_aabb = lib.einsum('jbai->iajb', eris.OVvo, optimize = True).copy()

#    #M^(2)_0 term 3 iemf
    vir_list_a = np.array(range(nvir_a))
    vir_list_b = np.array(range(nvir_b))
    occ_list_a = np.array(range(nocc_a))
    occ_list_b = np.array(range(nocc_b))

    t2_1_a = t2[0][0][:]
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.25*lib.einsum('jmef,iefm->ij',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] += 0.25*lib.einsum('jmef,ifem->ij',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.25*lib.einsum('imef,jefm->ij',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] += 0.25*lib.einsum('imef,jfem->ij',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.25*lib.einsum('mnae,mben->ab',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] += 0.25*lib.einsum('mnae,mebn->ab',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.25*lib.einsum('mnbe,maen->ab',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] += 0.25*lib.einsum('mnbe,mean->ab',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a += 0.5*lib.einsum('jmbe,iaem->iajb', t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a += 0.5*lib.einsum('imae,jbem->iajb', t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a -= 0.5*lib.einsum('imae,jebm->iajb',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a -= 0.5*lib.einsum('jmbe,ieam->iajb',t2_1_a, eris.ovvo, optimize = True)
    M_aabb += 0.5*lib.einsum('imae,jbem->iajb',t2_1_a, eris.OVvo, optimize = True)
    del t2_1_a

    t2_1_ab = t2[0][1][:]
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.5*lib.einsum('jmef,iefm->ij',t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.5*lib.einsum('imef,jefm->ij',t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.5*lib.einsum('mnae,mben->ab',t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.5*lib.einsum('mnbe,maen->ab',t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a += 0.5*lib.einsum('jmbe,iaem->iajb', t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a += 0.5*lib.einsum('imae,jbem->iajb', t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.5*lib.einsum('mjfe,iefm->ij',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.5*lib.einsum('mife,jefm->ij',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.5*lib.einsum('nmea,mben->ab',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.5*lib.einsum('nmeb,maen->ab',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b += 0.5*lib.einsum('mjeb,iaem->iajb',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b += 0.5*lib.einsum('miea,jbem->iajb',t2_1_ab, eris.OVvo, optimize = True)
    M_aabb += 0.5*lib.einsum('mjeb,iaem->iajb',t2_1_ab, eris.ovvo, optimize = True)
    M_aabb -= 0.5*lib.einsum('mjeb,ieam->iajb',t2_1_ab, eris.ovvo, optimize = True)
    M_aabb += 0.5*lib.einsum('imae,jbem->iajb',t2_1_ab, eris.OVVO, optimize = True)
    M_aabb -= 0.5*lib.einsum('imae,jebm->iajb',t2_1_ab, eris.OVVO, optimize = True)
    del t2_1_ab

    t2_1_b = t2[0][2][:]
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.25*lib.einsum('jmef,iefm->ij',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] += 0.25*lib.einsum('jmef,ifem->ij',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.25*lib.einsum('imef,jefm->ij',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] += 0.25*lib.einsum('imef,jfem->ij',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.25*lib.einsum('mnae,mben->ab',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 0.25*lib.einsum('mnae,mebn->ab',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.25*lib.einsum('mnbe,maen->ab',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 0.25*lib.einsum('mnbe,mean->ab',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b += 0.5*lib.einsum('jmbe,iaem->iajb',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b -= 0.5*lib.einsum('jmbe,ieam->iajb',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b += 0.5*lib.einsum('imae,jbem->iajb',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b -= 0.5*lib.einsum('imae,jebm->iajb',t2_1_b, eris.OVVO, optimize = True)
    M_aabb += 0.5*lib.einsum('mjeb,iaem->iajb',t2_1_b, eris.ovVO, optimize = True)
    del t2_1_b

    if isinstance(adc._scf, scf.rohf.ROHF):



        t1_1_a = t1[2][0][:]
        f_ov_a = adc.f_ov[0]
        M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.5*lib.einsum('je,ie->ij',t1_1_a,f_ov_a,optimize = True)
        M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.5*lib.einsum('ie,je->ij',t1_1_a,f_ov_a,optimize = True)
        M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.5*lib.einsum('ma,mb->ab',f_ov_a,t1_1_a,optimize = True)
        M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.5*lib.einsum('mb,ma->ab',f_ov_a,t1_1_a,optimize = True)
        del f_ov_a

        M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 2*lib.einsum('me,meij->ij',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_a[:, vir_list_a, :, vir_list_a] += lib.einsum('me,iemj->ij',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_a -= lib.einsum('ma,jbmi->iajb',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_a += lib.einsum('ma,mbji->iajb',t1_1_a,eris.ovoo,optimize = True)
        M_aabb -= lib.einsum('ma,jbmi->iajb',t1_1_a,eris.OVoo,optimize = True)
        M_ia_jb_a -= lib.einsum('mb,iamj->iajb',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_a += lib.einsum('mb,maij->iajb',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 2*lib.einsum('me,meij->ij',t1_1_a,eris.ovOO,optimize = True)
        M_ia_jb_a[:, vir_list_a, :, vir_list_a] += lib.einsum('me,jemi->ij',t1_1_a,eris.ovoo,optimize = True)
        del t1_1_a

        f_ov_b = adc.f_ov[1]
        t1_1_b = t1[2][1][:]
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.5*lib.einsum('je,ie->ij',t1_1_b,f_ov_b,optimize = True)
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.5*lib.einsum('ie,je->ij',t1_1_b,f_ov_b,optimize = True)
        M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.5*lib.einsum('ma,mb->ab',f_ov_b,t1_1_b,optimize = True)
        M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.5*lib.einsum('mb,ma->ab',f_ov_b,t1_1_b,optimize = True)
        del f_ov_b

        M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 2*lib.einsum('me,meij->ij',t1_1_b,eris.OVoo,optimize = True)
        M_aabb -= lib.einsum('mb,iamj->iajb',t1_1_b,eris.ovOO,optimize = True)
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 2*lib.einsum('me,meij->ij',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] += lib.einsum('me,iemj->ij',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] += lib.einsum('me,jemi->ij',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b -= lib.einsum('ma,jbmi->iajb',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b += lib.einsum('ma,mbji->iajb',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b -= lib.einsum('mb,iamj->iajb',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b += lib.einsum('mb,maij->iajb',t1_1_b,eris.OVOO,optimize = True)
        del t1_1_b

        t1_1_a = t1[2][0][:]
        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovvv = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                k = eris_ovvv.shape[0]
                M_ia_jb_a[occ_list_a, :, occ_list_a, :] += 2*lib.einsum('me,meba->ab',t1_1_a[a:a+k],eris_ovvv,optimize = True)
                M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= lib.einsum('me,mabe->ab',t1_1_a[a:a+k],eris_ovvv,optimize = True)
                M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= lib.einsum('me,mbea->ab',t1_1_a[a:a+k],eris_ovvv,optimize = True)
                M_ia_jb_a[a:a+k] -= lib.einsum('je,ieab->iajb',t1_1_a,eris_ovvv,optimize = True)
                M_ia_jb_a[a:a+k] += lib.einsum('je,iaeb->iajb',t1_1_a,eris_ovvv,optimize = True)
                M_ia_jb_a[:,:,a:a+k] -= lib.einsum('ie,jeba->iajb',t1_1_a,eris_ovvv,optimize = True)
                M_ia_jb_a[:,:,a:a+k] += lib.einsum('ie,jbea->iajb',t1_1_a,eris_ovvv,optimize = True)
                del eris_ovvv
                a += k
        else:
            eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            M_ia_jb_a[occ_list_a, :, occ_list_a, :] += 2*lib.einsum('me,meba->ab',t1_1_a,eris_ovvv,optimize = True)
            M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= lib.einsum('me,mabe->ab',t1_1_a,eris_ovvv,optimize = True)
            M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= lib.einsum('me,mbea->ab',t1_1_a,eris_ovvv,optimize = True)
            M_ia_jb_a -= lib.einsum('je,ieab->iajb',t1_1_a,eris_ovvv,optimize = True)
            M_ia_jb_a += lib.einsum('je,iaeb->iajb',t1_1_a,eris_ovvv,optimize = True)
            M_ia_jb_a -= lib.einsum('ie,jeba->iajb',t1_1_a,eris_ovvv,optimize = True)
            M_ia_jb_a += lib.einsum('ie,jbea->iajb',t1_1_a,eris_ovvv,optimize = True)
            del eris_ovvv
        del t1_1_a


        t1_1_b = t1[2][1][:]
        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVVV = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                k = eris_OVVV.shape[0]
                M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 2*lib.einsum('me,meba->ab',t1_1_b[a:a+k],eris_OVVV,optimize = True)
                M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= lib.einsum('me,mabe->ab',t1_1_b[a:a+k],eris_OVVV,optimize = True)
                M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= lib.einsum('me,mbea->ab',t1_1_b[a:a+k],eris_OVVV,optimize = True)
                M_ia_jb_b[a:a+k] -= lib.einsum('je,ieab->iajb',t1_1_b,eris_OVVV,optimize = True)
                M_ia_jb_b[a:a+k] += lib.einsum('je,iaeb->iajb',t1_1_b,eris_OVVV,optimize = True)
                M_ia_jb_b[:,:,a:a+k] -= lib.einsum('ie,jeba->iajb',t1_1_b,eris_OVVV,optimize = True)
                M_ia_jb_b[:,:,a:a+k] += lib.einsum('ie,jbea->iajb',t1_1_b,eris_OVVV,optimize = True)
                del eris_OVVV
                a += k
        else:
            eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 2*lib.einsum('me,meba->ab',t1_1_b,eris_OVVV,optimize = True)
            M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= lib.einsum('me,mabe->ab',t1_1_b,eris_OVVV,optimize = True)
            M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= lib.einsum('me,mbea->ab',t1_1_b,eris_OVVV,optimize = True)
            M_ia_jb_b -= lib.einsum('je,ieab->iajb',t1_1_b,eris_OVVV,optimize = True)
            M_ia_jb_b += lib.einsum('je,iaeb->iajb',t1_1_b,eris_OVVV,optimize = True)
            M_ia_jb_b -= lib.einsum('ie,jeba->iajb',t1_1_b,eris_OVVV,optimize = True)
            M_ia_jb_b += lib.einsum('ie,jbea->iajb',t1_1_b,eris_OVVV,optimize = True)
            del eris_OVVV
        del t1_1_b


        t1_1_a = t1[2][0][:]
        t1_1_b = t1[2][1][:]
        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVvv = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                k = eris_OVvv.shape[0]
                M_ia_jb_a[occ_list_a, :, occ_list_a, :] += 2*lib.einsum('me,meba->ab',t1_1_b[a:a+k],eris_OVvv,optimize = True)
                M_aabb[:,:,a:a+k] += lib.einsum('ie,jbea->iajb',t1_1_a,eris_OVvv,optimize = True)
                del eris_OVvv
                a += k
        else:
            eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            M_ia_jb_a[occ_list_a, :, occ_list_a, :] += 2*lib.einsum('me,meba->ab',t1_1_b,eris_OVvv,optimize = True)
            M_aabb += lib.einsum('ie,jbea->iajb',t1_1_a,eris_OVvv,optimize = True)
            del eris_OVvv
        del t1_1_a
        del t1_1_b

        t1_1_a = t1[2][0][:]
        t1_1_b = t1[2][1][:]
        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovVV = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                k = eris_ovVV.shape[0]
                M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 2*lib.einsum('me,meba->ab',t1_1_a[a:a+k],eris_ovVV,optimize = True)
                M_aabb[a:a+k] += lib.einsum('je,iaeb->iajb',t1_1_b,eris_ovVV,optimize = True)
                del eris_ovVV
                a += k
        else:
            eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 2*lib.einsum('me,meba->ab',t1_1_a,eris_ovVV,optimize = True)
            M_aabb += lib.einsum('je,iaeb->iajb',t1_1_b,eris_ovVV,optimize = True)
            del eris_ovVV
        del t1_1_a
        del t1_1_b

    #print("M_aaaa", np.linalg.norm(M_ia_jb_a))
    #print("M_bbbb", np.linalg.norm(M_ia_jb_b))
    #print("M_aabb", np.linalg.norm(M_aabb))
    #exit()

    if (method == "adc(3)"):

            t1 = adc.t1
            t2 = adc.t2

            if adc.f_ov is None:
                h_ce_aa = np.zeros((nocc_a, nvir_a))
                h_ce_bb = np.zeros((nocc_b, nvir_b))
                t1_ce_aa = np.zeros((nocc_a, nvir_a))
                t1_ce_bb = np.zeros((nocc_b, nvir_b))
            else:
                h_ce_aa, h_ce_bb = adc.f_ov
                t1_ce_aa = t1[2][0][:]
                t1_ce_bb = t1[2][1][:]

            einsum_type = True
            
            v_cccc_aaaa = eris.oooo
            v_cccc_bbbb = eris.OOOO
            v_cccc_aabb = eris.ooOO
            

            v_ceec_aaaa = eris.ovvo
            v_ceec_bbbb = eris.OVVO
            v_ceec_aabb = eris.ovVO
            v_ceec_bbaa = eris.OVvo
            
            
            v_ccee_aaaa = eris.oovv
            v_ccee_bbbb = eris.OOVV
            v_ccee_aabb = eris.ooVV
            v_ccee_bbaa = eris.OOvv

            v_cecc_aaaa = eris.ovoo
            v_cecc_bbbb = eris.OVOO
            v_cecc_aabb = eris.ovOO
            v_cecc_bbaa = eris.OVoo

            eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)

            v_ceee_aaaa = eris_ovvv.copy()
            v_ceee_bbbb = eris_OVVV.copy()
            v_ceee_aabb = eris_ovVV.copy()
            v_ceee_bbaa = eris_OVvv.copy()
            t2_ce_aa = t1[0][0][:]
            t2_ce_bb = t1[0][1][:]

            t1_ccee_aaaa = t2[0][0][:].copy()
            t1_ccee_abab = t2[0][1][:].copy()
            t1_ccee_bbbb = t2[0][2][:].copy()

            t2_ccee_aaaa = t2[1][0][:].copy()
            t2_ccee_abab = t2[1][1][:].copy()
            t2_ccee_bbbb = t2[1][2][:].copy()

            
            ncore_a = nocc_a
            nextern_a = nvir_a

            e_core_a = adc.mo_energy_a[:nocc_a].copy()
            e_extern_a = adc.mo_energy_a[nocc_a:].copy()
            
            ncore_b = nocc_b
            nextern_b = nvir_b

            e_core_b = adc.mo_energy_b[:nocc_b].copy()
            e_extern_b = adc.mo_energy_b[nocc_b:].copy()




            if isinstance(eris.vvvv_p, np.ndarray):
                v_eeee_aaaa = radc_ao2mo.unpack_eri_2(eris.vvvv_p, nvir_a)

                M_030_aa  =- lib.einsum('ADab,Ia,Lb->IDLA', v_eeee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                M_030_aa -= lib.einsum('ADab,Iiac,Libc->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aa -= lib.einsum('ADab,Iiac,Libc->IDLA', v_eeee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa += lib.einsum('AabD,La,Ib->IDLA', v_eeee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                M_030_aa += lib.einsum('AabD,Liac,Iibc->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aa += lib.einsum('AabD,Liac,Iibc->IDLA', v_eeee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa += lib.einsum('Aabc,Liac,IiDb->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aa += lib.einsum('Dabc,Iiac,LiAb->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('abcd,Iiac,Libd->IL',  v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ADab,ia,ib->DA',  v_eeee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ADab,ijac,ijbc->DA',  v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ADab,ijac,ijbc->DA',  v_eeee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('AabD,ia,ib->DA',  v_eeee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('AabD,ijac,ijbc->DA',  v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('AabD,ijac,ijbc->DA',  v_eeee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('Aabc,ijac,ijDb->DA',  v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('Dabc,ijac,ijAb->DA',  v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)

                M_030_aabb = lib.einsum('Dbcd,Iibd,ilca->IDla', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
                del v_eeee_aaaa

                v_eeee_bbbb = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)

                M_030_bb  = -lib.einsum('adbc,ib,lc->idla', v_eeee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                M_030_bb -= lib.einsum('adbc,ijbe,ljce->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_bb -= lib.einsum('adbc,jieb,jlec->idla', v_eeee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb += lib.einsum('abcd,lb,ic->idla', v_eeee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                M_030_bb += lib.einsum('abcd,ljbe,ijce->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_bb += lib.einsum('abcd,jleb,jiec->idla', v_eeee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb += lib.einsum('abce,ljbe,ijdc->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_bb += lib.einsum('dbce,ijbe,ljac->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('adbc,jb,jc->da',  v_eeee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('adbc,jkeb,jkec->da',  v_eeee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('adbc,jkbe,jkce->da',  v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('abcd,jb,jc->da',  v_eeee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('abcd,jkeb,jkec->da',  v_eeee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('abcd,jkbe,jkce->da',  v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('abce,jkbe,jkdc->da',  v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('dbce,jkbe,jkac->da',  v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('bcef,ijbe,ljcf->il',  v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            
                M_030_aabb += lib.einsum('abcd,libd,IiDc->IDla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
                del v_eeee_bbbb


            if isinstance(eris.vvvv_p, list):
    
                a = 0
                temp_ = np.zeros((nocc_a,nvir_a,nocc_a,nvir_a))
                for dataset in eris.vvvv_p:
                    k = dataset.shape[0]
                    vvvv = dataset[:]
                    v_eeee_aaaa = np.zeros((k,nvir_a,nvir_a,nvir_a))
                    v_eeee_aaaa[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv
                    v_eeee_aaaa[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv


                    temp_ = -lib.einsum('AaDb,Ia,Lb->IDLA', v_eeee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                    temp_ -= lib.einsum('AaDb,Iiac,Libc->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_ -= lib.einsum('AaDb,Iiac,Libc->IDLA', v_eeee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                    temp_ += 1/2 * lib. einsum('Aabc,IiDa,Libc->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_ += 1/2 * lib. einsum('Dabc,LiAa,Iibc->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_ += lib.einsum('IL,AaDb,ia,ib->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                    temp_ += 1/2 * lib.einsum('IL,AaDb,ijac,ijbc->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_ += lib.einsum('IL,AaDb,ijac,ijbc->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                    temp_ -= 1/4 *lib.einsum('AD,abcd,Iiab,Licd->IDLA', np.identity(nextern), v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_ -= 1/4 *lib.einsum('IL,Aabc,ijDa,ijbc->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_ -= 1/4 *lib.einsum('IL,Dabc,ijAa,ijbc->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)

                    del v_eeee_aaaa
                    a += k
                M_ia_jb_a  += temp_

            if isinstance(eris.VVVV_p, list):
    
                a = 0
                temp_ = np.zeros((nvir_b,nvir_b))
                for dataset in eris.vvvv_p:
                    k = dataset.shape[0]
                    VVVV = dataset[:]
                    v_eeee_bbbb = np.zeros((k,nvir_b,nvir_b,nvir_b))
                    v_eeee_bbbb[:,:,ab_ind_b[0],ab_ind_b[1]] = VVVV
                    v_eeee_bbbb[:,:,ab_ind_b[1],ab_ind_b[0]] = -VVVV

                    temp_[a:a+k] = -lib.einsum('abdc,ib,lc->idla', v_eeee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                    temp_[a:a+k] -= lib.einsum('abdc,ijbe,ljce->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] -= lib.einsum('abdc,jieb,jlec->idla', v_eeee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('abce,ijdb,ljce->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('dbce,ljab,ijce->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('ad,bcef,ijbc,ljef->idla', np.identity(nextern), v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] += lib.einsum('il,abdc,jb,jc->idla', np.identity(ncore), v_eeee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                    temp_[a:a+k] += lib.einsum('il,abdc,jkeb,jkec->idla', np.identity(ncore), v_eeee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('il,abdc,jkbe,jkce->idla', np.identity(ncore), v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('il,abce,jkdb,jkce->idla', np.identity(ncore), v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('il,dbce,jkab,jkce->idla', np.identity(ncore), v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    
                    del v_eeee_bbbb
                    a += k
                M_ia_jb_b  += temp_

            elif isinstance(eris.vvvv_p, type(None)):

                a = 0
                temp = np.zeros((nvir_a,nvir_a))
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for p in range(0,nvir_a,chnk_size):
                    vvvv = dfadc.get_vvvv_antisym_df(adc, eris.Lvv, p, chnk_size)
                    k = vvvv.shape[0]

                    eris_vvvv = np.zeros((k,nvir_a,nvir_a,nvir_a))
                    eris_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv
                    eris_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv

                    temp_[a:a+k] = -lib.einsum('AaDb,Ia,Lb->IDLA', v_eeee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                    temp_[a:a+k] -= lib.einsum('AaDb,Iiac,Libc->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_[a:a+k] -= lib.einsum('AaDb,Iiac,Libc->IDLA', v_eeee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('Aabc,IiDa,Libc->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('Dabc,LiAa,Iibc->IDLA', v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_[a:a+k] += lib.einsum('IL,AaDb,ia,ib->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('IL,AaDb,ijac,ijbc->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_[a:a+k] += lib.einsum('IL,AaDb,ijac,ijbc->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('AD,abcd,Iiab,Licd->IDLA', np.identity(nextern), v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('IL,Aabc,ijDa,ijbc->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('IL,Dabc,ijAa,ijbc->IDLA', np.identity(ncore), v_eeee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)

                    del v_eeee_aaaa
                    a += k
                M_ia_jb_a  += temp

            elif isinstance(eris.VVVV_p, type(None)):

                a = 0
                temp = np.zeros((nvir_b,nvir_b))
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for p in range(0,nvir_b,chnk_size):
                    VVVV = dfadc.get_vvvv_antisym_df(adc, eris.LVV, p, chnk_size)
                    k = VVVV.shape[0]

                    eris_VVVV = np.zeros((k,nvir_b,nvir_b,nvir_b))
                    eris_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = VVVV
                    eris_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -VVVV

                    temp_[a:a+k] = -lib.einsum('abdc,ib,lc->idla', v_eeee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                    temp_[a:a+k] -= lib.einsum('abdc,ijbe,ljce->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] -= lib.einsum('abdc,jieb,jlec->idla', v_eeee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('abce,ijdb,ljce->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('dbce,ljab,ijce->idla', v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('ad,bcef,ijbc,ljef->idla', np.identity(nextern), v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] += lib.einsum('il,abdc,jb,jc->idla', np.identity(ncore), v_eeee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                    temp_[a:a+k] += lib.einsum('il,abdc,jkeb,jkec->idla', np.identity(ncore), v_eeee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                    temp_[a:a+k] += 1/2 * lib.einsum('il,abdc,jkbe,jkce->idla', np.identity(ncore), v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('il,abce,jkdb,jkce->idla', np.identity(ncore), v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    temp_[a:a+k] -= 1/4 * lib.einsum('il,dbce,jkab,jkce->idla', np.identity(ncore), v_eeee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                    
                    del v_eeee_bbbb
                    a += k
                M_ia_jb_b  += temp_

            if isinstance(eris.vvvv_p, list):
    
                a = 0
                temp_ = np.zeros((nvir_a,nvir_a))
                for dataset in eris.vvvv_p:
                    k = dataset.shape[0]
                    vvvv = dataset[:]
                    v_eeee_aaaa = np.zeros((k,nvir_a,nvir_a,nvir_a))
                    v_eeee_aaaa[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv
                    v_eeee_aaaa[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv

                    temp_[a:a+k] = 1/2 * lib.einsum('Dbcd,ilba,Iicd->IDla', v_eeee_aaaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
                    del v_eeee_aaaa
                    a += k
                M_aabb  += temp_

            elif isinstance(eris.vvvv_p, type(None)):

                a = 0
                temp = np.zeros((nvir_a,nvir_a))
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for p in range(0,nvir_a,chnk_size):
                    vvvv = dfadc.get_vvvv_antisym_df(adc, eris.Lvv, p, chnk_size)
                    k = vvvv.shape[0]

                    eris_vvvv = np.zeros((k,nvir_a,nvir_a,nvir_a))
                    eris_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv
                    eris_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv

                    temp_[a:a+k] = 1/2 * lib.einsum('Dbcd,ilba,Iicd->IDla', v_eeee_aaaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
                    del v_eeee_aaaa
                    a += k
                M_aabb  += temp_

            elif isinstance(eris.VVVV_p, type(None)):

                a = 0
                temp = np.zeros((nvir_b,nvir_b))
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for p in range(0,nvir_b,chnk_size):
                    VVVV = dfadc.get_vvvv_antisym_df(adc, eris.LVV, p, chnk_size)
                    k = VVVV.shape[0]

                    eris_VVVV = np.zeros((k,nvir_b,nvir_b,nvir_b))
                    eris_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = VVVV
                    eris_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -VVVV


                    temp_[a:a+k] = 1/2 * lib.einsum('Dbcd,ilba,Iicd->IDla', v_eeee_aaaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
                    del v_eeee_aaaa
                    a += k
                M_aabb  += temp_

            if isinstance(eris.VVVV_p, list):
    
                a = 0
                temp_ = np.zeros((nvir_b,nvir_b))
                for dataset in eris.vvvv_p:
                    k = dataset.shape[0]
                    VVVV = dataset[:]
                    v_eeee_bbbb = np.zeros((k,nvir_b,nvir_b,nvir_b))
                    v_eeee_bbbb[:,:,ab_ind_b[0],ab_ind_b[1]] = VVVV
                    v_eeee_bbbb[:,:,ab_ind_b[1],ab_ind_b[0]] = -VVVV


                    temp_[a:a+k] = 1/2 * lib.einsum('abcd,IiDb,licd->IDla', v_eeee_bbbb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
                    del v_eeee_bbbb
                    a += k
                M_aabb  += temp_



            if isinstance(eris.vVvV_p,np.ndarray):

                v_eeee_abab = eris.vVvV_p
                v_eeee_abab = v_eeee_abab.reshape(nvir_a,nvir_b,nvir_a,nvir_b)

                M_030_aa -= lib.einsum('AaDb,Iica,Licb->IDLA', v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa += lib.einsum('Aabc,IiDa,Libc->IDLA', v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa += lib.einsum('Dabc,LiAa,Iibc->IDLA', v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa -= lib.einsum('AD,abcd,Iiab,Licd->IDLA', np.identity(nextern_a), v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa += lib.einsum('IL,AaDb,ia,ib->IDLA', np.identity(ncore_a), v_eeee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
                M_030_aa += lib.einsum('IL,AaDb,ijca,ijcb->IDLA', np.identity(ncore_a), v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa += 1/2 * lib.einsum('IL,AaDb,ijac,ijbc->IDLA', np.identity(ncore_a), v_eeee_abab, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
                M_030_aa -= lib.einsum('IL,Aabc,ijDa,ijbc->IDLA', np.identity(ncore_a), v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_aa -= lib.einsum('IL,Dabc,ijAa,ijbc->IDLA', np.identity(ncore_a), v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                
                M_030_bb -= lib.einsum('bacd,jibe,jlce->idla', v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb += lib.einsum('bace,jibd,jlce->idla', v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb -= lib.einsum('ad,bcef,jibc,jlef->idla', np.identity(nextern_b), v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb += 1/2 * lib.einsum('il,bacd,jkbe,jkce->idla', np.identity(ncore_b), v_eeee_abab, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
                M_030_bb += lib.einsum('il,bacd,jb,jc->idla', np.identity(ncore_b), v_eeee_abab, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
                M_030_bb += lib.einsum('il,bacd,jkbe,jkce->idla', np.identity(ncore_b), v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb -= lib.einsum('il,bace,jkbd,jkce->idla', np.identity(ncore_b), v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb += lib.einsum('bdce,jlba,jice->idla', v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
                M_030_bb -= lib.einsum('il,bdce,jkba,jkce->idla', np.identity(ncore_b), v_eeee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)

                M_030_aabb += lib.einsum('Dbca,lb,Ic->IDla', v_eeee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
                M_030_aabb += lib.einsum('Dbca,libd,Iicd->IDla', v_eeee_abab, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
                M_030_aabb += lib.einsum('Dbca,ildb,Iicd->IDla', v_eeee_abab, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
                M_030_aabb += lib.einsum('Dbcd,liab,Iicd->IDla', v_eeee_abab, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
                M_030_aabb += lib.einsum('bacd,IiDb,ilcd->IDla', v_eeee_abab, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
                del v_eeee_abab


            #M_030_aabb += lib.einsum('Dbca,Ib,lc->IDla', v_eeee_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            #M_030_aabb += lib.einsum('Dbca,Iibd,ildc->IDla', v_eeee_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            #M_030_aabb += lib.einsum('Dbca,Iibd,licd->IDla', v_eeee_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            #M_030_aabb += lib.einsum('Dbcd,Iibd,liac->IDla', v_eeee_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)

            #M_030_aabb += lib.einsum('abcd,ildb,IiDc->IDla', v_eeee_bbaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)


            M_030_aa -= lib.einsum('iA,IDiL->IDLA', t2_ce_aa, v_cecc_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iA,iDIL->IDLA', t2_ce_aa, v_cecc_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('iD,LAiI->IDLA', t2_ce_aa, v_cecc_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iD,iALI->IDLA', t2_ce_aa, v_cecc_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('IiDa,LAai->IDLA', t2_ccee_aaaa, v_ceec_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('IiDa,iAaL->IDLA', t2_ccee_aaaa, v_ceec_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('Ia,LaDA->IDLA', t2_ce_aa, v_ceee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('Ia,LADa->IDLA', t2_ce_aa, v_ceee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('IiDa,LAai->IDLA', t2_ccee_abab, v_ceec_aabb, optimize = einsum_type)
            M_030_aa += lib.einsum('LiAa,IDai->IDLA', t2_ccee_aaaa, v_ceec_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('LiAa,iDaI->IDLA', t2_ccee_aaaa, v_ceec_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('La,IaAD->IDLA', t2_ce_aa, v_ceee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('La,IDAa->IDLA', t2_ce_aa, v_ceee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('LiAa,IDai->IDLA', t2_ccee_abab, v_ceec_aabb, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('A,IiDa,LiAa->IDLA', e_extern_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('A,IiDa,LiAa->IDLA', e_extern_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('A,LiAa,IiDa->IDLA', e_extern_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('A,LiAa,IiDa->IDLA', e_extern_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('D,IiDa,LiAa->IDLA', e_extern_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('D,IiDa,LiAa->IDLA', e_extern_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('D,LiAa,IiDa->IDLA', e_extern_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('D,LiAa,IiDa->IDLA', e_extern_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('I,IiDa,LiAa->IDLA', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('I,IiDa,LiAa->IDLA', e_core_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('I,LiAa,IiDa->IDLA', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('I,LiAa,IiDa->IDLA', e_core_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('L,IiDa,LiAa->IDLA', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('L,IiDa,LiAa->IDLA', e_core_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('L,LiAa,IiDa->IDLA', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('L,LiAa,IiDa->IDLA', e_core_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('i,IiDa,LiAa->IDLA', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('i,LiAa,IiDa->IDLA', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('a,IiDa,LiAa->IDLA', e_extern_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('a,LiAa,IiDa->IDLA', e_extern_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('i,IiDa,LiAa->IDLA', e_core_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('i,LiAa,IiDa->IDLA', e_core_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa += lib.einsum('a,IiDa,LiAa->IDLA', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa += lib.einsum('a,LiAa,IiDa->IDLA', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iA,iD->DA', h_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iD,iA->DA', h_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Ia,La->IL', h_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('La,Ia->IL', h_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iA,IiDa,La->IDLA', h_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iD,LiAa,Ia->IDLA', h_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('Ia,LiAa,iD->IDLA', h_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('La,IiDa,iA->IDLA', h_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iiab,Labi->IL', t2_ccee_aaaa, v_ceec_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iiab,Labi->IL', t2_ccee_abab, v_ceec_aabb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Liab,Iabi->IL', t2_ccee_aaaa, v_ceec_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Liab,Iabi->IL', t2_ccee_abab, v_ceec_aabb, optimize = einsum_type)

            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('ia,iaIL->IL',t2_ce_aa, v_cecc_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ia,LaiI->IL',t2_ce_aa, v_cecc_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('ia,iaLI->IL',t2_ce_aa, v_cecc_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ia,IaiL->IL',t2_ce_aa, v_cecc_aaaa, optimize = einsum_type)

            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('ia,iaIL->IL',t2_ce_bb, v_cecc_bbaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('ia,iaLI->IL',t2_ce_bb, v_cecc_bbaa, optimize = einsum_type)

            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijAa,iDaj->DA',t2_ccee_aaaa, v_ceec_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijDa,iAaj->DA',t2_ccee_aaaa, v_ceec_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ia,iaAD->DA',t2_ce_aa, v_ceee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ia,iADa->DA',t2_ce_aa, v_ceee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijAa,iDaj->DA',t2_ccee_abab, v_ceec_aabb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijDa,iAaj->DA',t2_ccee_abab, v_ceec_aabb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ia,iaDA->DA',t2_ce_aa, v_ceee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ia,iDAa->DA',t2_ce_aa, v_ceee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ia,iaAD->DA',t2_ce_bb, v_ceee_bbaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ia,iaDA->DA',t2_ce_bb, v_ceee_bbaa, optimize = einsum_type)
            M_030_aa += 1/4 * lib.einsum('IiAD,ijab,Ljab->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('IiAD,ia,La->IDLA', v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('IiAD,ijab,Ljab->IDLA', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('IaAD,Liab,ib->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('IaAD,Liab,ib->IDLA', v_ceee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa += 1/4 * lib.einsum('iLAD,ijab,Ijab->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('iLAD,ia,Ia->IDLA', v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('iLAD,ijab,Ijab->IDLA', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('ijAD,Liab,Ijab->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('iaAD,Liab,Ib->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('LaDA,Iiab,ib->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('LaDA,Iiab,ib->IDLA', v_ceee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('iaDA,Iiab,Lb->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('ijAD,Liab,Ijab->IDLA', v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iaAD,Liba,Ib->IDLA', v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iaDA,Iiba,Lb->IDLA', v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)

            M_030_aa -= 1/2 * lib.einsum('LAiI,ijDa,ja->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('LAiI,ijDa,ja->IDLA', v_cecc_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)

            M_030_aa -= 1/2 * lib.einsum('LAaI,ia,iD->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/4 * lib.einsum('LAaI,ijab,ijDb->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('LAaI,ijab,ijDb->IDLA', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/4 * lib.einsum('LADi,ijab,Ijab->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('LADi,ia,Ia->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('LADi,ijab,Ijab->IDLA', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)

            M_030_aa -= lib.einsum('LAij,ja,IiDa->IDLA', v_cecc_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)

            M_030_aa += 1/2 * lib.einsum('LAai,ijab,IjDb->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('LAai,ijab,IjDb->IDLA', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('LADa,Iiab,ib->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('LADa,Iiab,ib->IDLA', v_ceee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa -= lib.einsum('LAai,Ia,iD->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('LAab,IiDb,ia->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa -= lib.einsum('LAij,ja,IiDa->IDLA', v_cecc_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa += 1/2 * lib.einsum('LAai,ijab,IjDb->IDLA', v_ceec_aabb, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('LAai,jiba,IjDb->IDLA', v_ceec_aabb, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('LAab,IiDb,ia->IDLA', v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa -= 1/4 * lib.einsum('iADI,ijab,Ljab->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iADI,ia,La->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iADI,ijab,Ljab->IDLA', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)

            M_030_aa += 1/2 * lib.einsum('iALI,ijDa,ja->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('iALI,ijDa,ja->IDLA', v_cecc_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('jAiI,jiDa,La->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)

            M_030_aa += lib.einsum('iAaI,iD,La->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('iAaI,ijDb,Ljab->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iAaI,ijDb,Ljab->IDLA', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iADj,Iiab,Ljab->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)

            M_030_aa += lib.einsum('jALi,IjDa,ia->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)

            M_030_aa += lib.einsum('iAaj,IiDb,Ljab->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iADa,Iiab,Lb->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('iAaL,iD,Ia->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= lib.einsum('iAab,IiDb,La->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= lib.einsum('iAaj,IiDb,Ljba->IDLA', v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('IDAa,Liab,ib->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('IDAa,Liab,ib->IDLA', v_ceee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('ILAa,ia,iD->IDLA', v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/4 * lib.einsum('ILAa,ijab,ijDb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('ILAa,ijab,ijDb->IDLA', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('IiAa,La,iD->IDLA', v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= lib.einsum('IiAa,Ljab,ijDb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('IiAa,Ljab,ijDb->IDLA', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('IaAb,Liba,iD->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iDAa,Liab,Ib->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iLAa,ijab,IjDb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iLAa,ijab,IjDb->IDLA', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('ijAa,Liab,IjDb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('LabA,ib,IiDa->IDLA', v_ceee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iabA,Lb,IiDa->IDLA', v_ceee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('ijAa,Liab,IjDb->IDLA', v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += lib.einsum('iabA,Lb,IiDa->IDLA', v_ceee_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)

            M_030_aa -= 1/2 * lib.einsum('IDiL,ijAa,ja->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('IDiL,ijAa,ja->IDLA', v_cecc_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)

            M_030_aa -= 1/2 * lib.einsum('IDaL,ia,iA->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/4 * lib.einsum('IDaL,ijab,ijAb->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('IDaL,ijab,ijAb->IDLA', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)

            M_030_aa -= lib.einsum('IDij,ja,LiAa->IDLA', v_cecc_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)

            M_030_aa += 1/2 * lib.einsum('IDai,ijab,LjAb->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('IDai,ijab,LjAb->IDLA', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('IDai,La,iA->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('IDab,LiAb,ia->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa -= lib.einsum('IDij,ja,LiAa->IDLA', v_cecc_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa += 1/2 * lib.einsum('IDai,ijab,LjAb->IDLA', v_ceec_aabb, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('IDai,jiba,LjAb->IDLA', v_ceec_aabb, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('IDab,LiAb,ia->IDLA', v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)

            M_030_aa += 1/2 * lib.einsum('iDIL,ijAa,ja->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('iDIL,ijAa,ja->IDLA', v_cecc_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('jDiL,jiAa,Ia->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)

            M_030_aa += lib.einsum('iDaL,iA,Ia->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('iDaL,ijAb,Ijab->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iDaL,ijAb,Ijab->IDLA', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)

            M_030_aa += lib.einsum('jDIi,LjAa,ia->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)

            M_030_aa += lib.einsum('iDaj,LiAb,Ijab->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iDaI,iA,La->IDLA', v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= lib.einsum('iDab,LiAb,Ia->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= lib.einsum('iDaj,LiAb,Ijba->IDLA', v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('LIDa,ia,iA->IDLA', v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/4 * lib.einsum('LIDa,ijab,ijAb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('LIDa,ijab,ijAb->IDLA', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('LiDa,Ia,iA->IDLA', v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= lib.einsum('LiDa,Ijab,ijAb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('LiDa,Ijab,ijAb->IDLA', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('LaDb,Iiba,iA->IDLA', v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iIDa,ijab,LjAb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iIDa,ijab,LjAb->IDLA', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('ijDa,Iiab,LjAb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('IabD,ib,LiAa->IDLA', v_ceee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iabD,Ib,LiAa->IDLA', v_ceee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('ijDa,Iiab,LjAb->IDLA', v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += lib.einsum('iabD,Ib,LiAa->IDLA', v_ceee_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('ILij,iA,jD->IDLA', v_cccc_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= lib.einsum('ILij,ikAa,jkDa->IDLA', v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('ILij,ikAa,jkDa->IDLA', v_cccc_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)

            M_030_aa -= 1/2 * lib.einsum('iaIL,ijAa,jD->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 1/2 * lib.einsum('iaLI,ijDa,jA->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)

            M_030_aa += 1/2 * lib.einsum('ILab,ijDa,ijAb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('ILij,kiAa,kjDa->IDLA', v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('iaIL,jiAa,jD->IDLA', v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('iaLI,jiDa,jA->IDLA', v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('ILab,ijDa,ijAb->IDLA', v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += lib.einsum('IijL,iD,jA->IDLA', v_cccc_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('IijL,ikDa,jkAa->IDLA', v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('IijL,ikDa,jkAa->IDLA', v_cccc_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += lib.einsum('Iijk,ikDa,LjAa->IDLA', v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aa -= lib.einsum('jaIi,iD,LjAa->IDLA', v_cecc_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 1/2 * lib.einsum('LaiI,ijDa,jA->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa -= lib.einsum('Iiab,ijDa,LjAb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('Iijk,ikDa,LjAa->IDLA', v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('jaIi,iD,LjAa->IDLA', v_cecc_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('Iiab,ijDa,LjAb->IDLA', v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa += 1/2 * lib.einsum('IaiL,ijAa,jD->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += lib.einsum('Iaij,LiAa,jD->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa -= 1/2 * lib.einsum('IabL,ijAa,ijDb->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('Iabi,LjAa,ijDb->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('Iabi,LjAa,jiDb->IDLA', v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += lib.einsum('Lijk,ikAa,IjDa->IDLA', v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aa -= lib.einsum('jaLi,iA,IjDa->IDLA', v_cecc_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aa -= lib.einsum('Liab,ijAa,IjDb->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('Lijk,ikAa,IjDa->IDLA', v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('jaLi,iA,IjDa->IDLA', v_cecc_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('Liab,ijAa,IjDb->IDLA', v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa += lib.einsum('Laij,IiDa,jA->IDLA', v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa += lib.einsum('Labi,IjDa,ijAb->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= lib.einsum('Labi,IjDa,jiAb->IDLA', v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('ijab,IiDb,LjAa->IDLA', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iabj,IiDa,LjAb->IDLA', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iabj,IiDa,LjAb->IDLA', v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa -= lib.einsum('ijab,IiDb,LjAa->IDLA', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa += lib.einsum('iabj,IiDa,LjAb->IDLA', v_ceec_bbaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += lib.einsum('iabj,IiDa,LjAb->IDLA', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('A,iA,iD->DA', e_extern_a,t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('A,iD,iA->DA', e_extern_a,t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('A,ijAa,ijDa->DA', e_extern_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('A,ijDa,ijAa->DA', e_extern_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('A,ijAa,ijDa->DA', e_extern_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('A,ijDa,ijAa->DA', e_extern_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('D,iA,iD->DA', e_extern_a,t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('D,iD,iA->DA', e_extern_a,t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('D,ijAa,ijDa->DA', e_extern_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('D,ijDa,ijAa->DA', e_extern_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('D,ijAa,ijDa->DA', e_extern_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('D,ijDa,ijAa->DA', e_extern_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/4 * lib.einsum('I,Iiab,Liab->IL', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('I,Ia,La->IL', e_core_a,  t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('I,Iiab,Liab->IL', e_core_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/4 * lib.einsum('I,Liab,Iiab->IL', e_core_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('I,La,Ia->IL', e_core_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('I,Liab,Iiab->IL', e_core_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/4 * lib.einsum('L,Iiab,Liab->IL', e_core_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('L,Ia,La->IL', e_core_a,  t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('L,Iiab,Liab->IL', e_core_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/4 * lib.einsum('L,Liab,Iiab->IL', e_core_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('L,La,Ia->IL', e_core_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('L,Liab,Iiab->IL', e_core_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('i,Iiab,Liab->IL', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('i,Liab,Iiab->IL', e_core_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('i,iA,iD->DA', e_core_a,t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('i,iD,iA->DA', e_core_a,t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('i,ijAa,ijDa->DA', e_core_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('i,ijDa,ijAa->DA', e_core_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('i,ijAa,ijDa->DA', e_core_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('i,ijDa,ijAa->DA', e_core_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('a,Ia,La->IL', e_extern_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('a,Iiab,Liab->IL', e_extern_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('a,Iiab,Liab->IL', e_extern_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('a,La,Ia->IL', e_extern_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('a,Liab,Iiab->IL', e_extern_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('a,Liab,Iiab->IL', e_extern_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('a,ijAa,ijDa->DA', e_extern_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('a,ijDa,ijAa->DA', e_extern_a,t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('i,Iiab,Liab->IL', e_core_b,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('i,Liab,Iiab->IL', e_core_b,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('i,jiAa,jiDa->DA', e_core_b,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('i,jiDa,jiAa->DA', e_core_b,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('a,Iiba,Liba->IL', e_extern_b,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('a,Liba,Iiba->IL', e_extern_b,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('a,ijAa,ijDa->DA', e_extern_b,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('a,ijDa,ijAa->DA', e_extern_b,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_aa -= 0.166666667 * lib.einsum('A,iA,IiDa,La->IDLA', e_extern_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 0.333333334 * lib.einsum('A,iD,Ia,LiAa->IDLA', e_extern_a, t1_ce_aa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa -= 0.333333334 * lib.einsum('D,iA,IiDa,La->IDLA', e_extern_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 0.166666667 * lib.einsum('D,iD,Ia,LiAa->IDLA', e_extern_a, t1_ce_aa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 0.333333334 * lib.einsum('I,iA,IiDa,La->IDLA', e_core_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 0.166666667 * lib.einsum('I,iD,Ia,LiAa->IDLA', e_core_a, t1_ce_aa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 0.166666667 * lib.einsum('L,iA,IiDa,La->IDLA', e_core_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 0.333333334 * lib.einsum('L,iD,Ia,LiAa->IDLA', e_core_a, t1_ce_aa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa += 0.500000001 * lib.einsum('i,iD,LiAa,Ia->IDLA', e_core_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa += 0.500000001 * lib.einsum('i,IiDa,iA,La->IDLA', e_core_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 0.500000001 * lib.einsum('a,La,IiDa,iA->IDLA', e_extern_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa -= 0.500000001 * lib.einsum('a,LiAa,Ia,iD->IDLA', e_extern_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iA,ijDa,ja->DA', h_ce_aa,t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iA,ijDa,ja->DA', h_ce_aa,t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iD,ijAa,ja->DA', h_ce_aa,t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iD,ijAa,ja->DA', h_ce_aa,t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('Ia,Liab,ib->IL', h_ce_aa,t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('Ia,Liab,ib->IL', h_ce_aa,t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('La,Iiab,ib->IL', h_ce_aa,t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('La,Iiab,ib->IL', h_ce_aa,t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('ia,Iiab,Lb->IL', h_ce_aa,t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('ia,Liab,Ib->IL', h_ce_aa,t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ia,ijAa,jD->DA', h_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ia,ijDa,jA->DA', h_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('ia,Iiba,Lb->IL', h_ce_bb,t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('ia,Liba,Ib->IL', h_ce_bb,t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ia,jiAa,jD->DA', h_ce_bb,t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ia,jiDa,jA->DA', h_ce_bb,t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('ILij,ikab,jkab->IL',  v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ILij,ia,ja->IL',  v_cccc_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ILij,ikab,jkab->IL',  v_cccc_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('iaIL,ijab,jb->IL',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('iaIL,ijab,jb->IL',  v_cecc_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('iaLI,ijab,jb->IL',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('iaLI,ijab,jb->IL',  v_cecc_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('ILab,ia,ib->IL',  v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('ILab,ijac,ijbc->IL',  v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('ILab,ijac,ijbc->IL',  v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('ILij,ikab,jkab->IL',  v_cccc_aabb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ILij,ia,ja->IL',  v_cccc_aabb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ILij,kiab,kjab->IL',  v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('iaIL,ijab,jb->IL',  v_cecc_bbaa, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('iaIL,jiba,jb->IL',  v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('iaLI,ijab,jb->IL',  v_cecc_bbaa, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('iaLI,jiba,jb->IL',  v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('ILab,ia,ib->IL',  v_ccee_aabb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('ILab,ijca,ijcb->IL',  v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('ILab,ijac,ijbc->IL',  v_ccee_aabb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('IijL,ikab,jkab->IL',  v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('IijL,ia,ja->IL',  v_cccc_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('IijL,ikab,jkab->IL',  v_cccc_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('Iijk,ikab,Ljab->IL',  v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('jaIi,ib,Ljab->IL',  v_cecc_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('LaiI,ijab,jb->IL',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('LaiI,ijab,jb->IL',  v_cecc_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('jaiI,ijab,Lb->IL',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Iiab,ia,Lb->IL',  v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Iiab,ijac,Ljbc->IL',  v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Iiab,ijac,Ljbc->IL',  v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iijk,ikab,Ljab->IL',  v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('jaIi,ib,Ljba->IL',  v_cecc_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('jaiI,ijba,Lb->IL',  v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Iiab,ijca,Ljcb->IL',  v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('IaiL,ijab,jb->IL',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('IaiL,ijab,jb->IL',  v_cecc_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Iaij,Liab,jb->IL',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabi,La,ib->IL',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Iabi,ia,Lb->IL',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('IabL,ia,ib->IL',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('IabL,ijac,ijbc->IL',  v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('IabL,ijac,ijbc->IL',  v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabi,La,ib->IL',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabi,Ljac,ijbc->IL',  v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabi,Ljac,ijbc->IL',  v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabc,Liac,ib->IL',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Iaij,Liab,jb->IL',  v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabi,La,ib->IL',  v_ceec_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabi,La,ib->IL',  v_ceec_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabi,Ljac,jicb->IL',  v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabi,Ljac,ijbc->IL',  v_ceec_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Iabc,Liac,ib->IL',  v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('Lijk,ikab,Ijab->IL',  v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('jaLi,ib,Ijab->IL',  v_cecc_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 1/2 * lib.einsum('jaiL,ijab,Ib->IL',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Liab,ia,Ib->IL',  v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Liab,ijac,Ijbc->IL',  v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Liab,ijac,Ijbc->IL',  v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Lijk,ikab,Ijab->IL',  v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('jaLi,ib,Ijba->IL',  v_cecc_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 1/2 * lib.einsum('jaiL,ijba,Ib->IL',  v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Liab,ijca,Ijcb->IL',  v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Laij,Iiab,jb->IL',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labi,Ia,ib->IL',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Labi,ia,Ib->IL',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labi,Ia,ib->IL',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labi,Ijac,ijbc->IL',  v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labi,Ijac,ijbc->IL',  v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labc,Iiac,ib->IL',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('Laij,Iiab,jb->IL',  v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labi,Ia,ib->IL',  v_ceec_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labi,Ia,ib->IL',  v_ceec_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labi,Ijac,jicb->IL',  v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labi,Ijac,ijbc->IL',  v_ceec_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('Labc,Iiac,ib->IL',  v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ijab,Iibc,Ljac->IL',  v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('iabj,Iiac,Ljbc->IL',  v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('iabc,Iiac,Lb->IL',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('iabj,Iiac,Ljcb->IL',  v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('iabc,Ib,Lica->IL',  v_ceee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('iabc,Ib,Lica->IL',  v_ceee_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ijab,Iibc,Ljac->IL',  v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('ijab,Iicb,Ljca->IL',  v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += lib.einsum('iabj,Iica,Ljbc->IL',  v_ceec_bbaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('iabc,Iica,Lb->IL',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= lib.einsum('iabj,Iica,Ljcb->IL',  v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ijAD,ikab,jkab->DA',  v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijAD,ia,ja->DA',  v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijAD,ikab,jkab->DA',  v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaAD,ijab,jb->DA',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaAD,ijab,jb->DA',  v_ceee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaDA,ijab,jb->DA',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaDA,ijab,jb->DA',  v_ceee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ijAD,ikab,jkab->DA',  v_ccee_bbaa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijAD,ia,ja->DA',  v_ccee_bbaa, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijAD,kiab,kjab->DA',  v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaAD,ijab,jb->DA',  v_ceee_bbaa, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaAD,jiba,jb->DA',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaDA,ijab,jb->DA',  v_ceee_bbaa, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaDA,jiba,jb->DA',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iADj,ikab,jkab->DA',  v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('iADj,ia,ja->DA',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('iADj,ikab,jkab->DA',  v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('kAij,kiDa,ja->DA',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAaj,iD,ja->DA',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAaj,ikDb,jkab->DA',  v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAaj,ikDb,jkab->DA',  v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iADa,ijab,jb->DA',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iADa,ijab,jb->DA',  v_ceee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAaj,iD,ja->DA',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('iAaj,ia,jD->DA',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAab,ijDb,ja->DA',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('kAij,kiDa,ja->DA',  v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAaj,iD,ja->DA',  v_ceec_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAaj,ikDb,kjba->DA',  v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAaj,ikDb,jkab->DA',  v_ceec_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAaj,iD,ja->DA',  v_ceec_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iAab,ijDb,ja->DA',  v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iDAa,ijab,jb->DA',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iDAa,ijab,jb->DA',  v_ceee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijAa,ia,jD->DA',  v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijAa,ikab,jkDb->DA',  v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijAa,ikab,jkDb->DA',  v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaAb,ijba,jD->DA',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('iabA,jb,ijDa->DA',  v_ceee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijAa,kiab,kjDb->DA',  v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iaAb,jiba,jD->DA',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iabA,jb,jiDa->DA',  v_ceee_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('kDij,kiAa,ja->DA',  v_cecc_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDaj,iA,ja->DA',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDaj,ikAb,jkab->DA',  v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDaj,ikAb,jkab->DA',  v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDaj,iA,ja->DA',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('iDaj,ia,jA->DA',  v_ceec_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDab,ijAb,ja->DA',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('kDij,kiAa,ja->DA',  v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDaj,iA,ja->DA',  v_ceec_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDaj,ikAb,kjba->DA',  v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDaj,ikAb,jkab->DA',  v_ceec_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDaj,iA,ja->DA',  v_ceec_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iDab,ijAb,ja->DA',  v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijDa,ia,jA->DA',  v_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijDa,ikab,jkAb->DA',  v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijDa,ikab,jkAb->DA',  v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('iaDb,ijba,jA->DA',  v_ceee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('iabD,jb,ijAa->DA',  v_ceee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijDa,kiab,kjAb->DA',  v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('iaDb,jiba,jA->DA',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iabD,jb,jiAa->DA',  v_ceee_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ijkl,ikAa,jlDa->DA',  v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('kaij,jA,ikDa->DA',  v_cecc_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('kaij,jD,ikAa->DA',  v_cecc_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijab,ikAb,jkDa->DA',  v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('ijkl,ikAa,jlDa->DA',  v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('kaij,jA,ikDa->DA',  v_cecc_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('kaij,jD,ikAa->DA',  v_cecc_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijab,ikAb,jkDa->DA',  v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iabj,ikAa,jkDb->DA',  v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('iabj,ikAa,kjDb->DA',  v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('iabj,ikDa,kjAb->DA',  v_ceec_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += lib.einsum('ijab,kiAb,kjDa->DA',  v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= lib.einsum('iabj,kiAa,kjDb->DA',  v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.166666667 * lib.einsum('A,iA,ijDa,ja->DA', e_extern_a,  t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.166666667 * lib.einsum('A,iA,ijDa,ja->DA', e_extern_a,  t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.333333334 * lib.einsum('A,iD,ijAa,ja->DA', e_extern_a,  t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.333333334 * lib.einsum('A,iD,ijAa,ja->DA', e_extern_a,  t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.333333334 * lib.einsum('D,iA,ijDa,ja->DA', e_extern_a,  t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.333333334 * lib.einsum('D,iA,ijDa,ja->DA', e_extern_a,  t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.166666667 * lib.einsum('D,iD,ijAa,ja->DA', e_extern_a,  t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.166666667 * lib.einsum('D,iD,ijAa,ja->DA', e_extern_a,  t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.333333334 * lib.einsum('I,Iiab,La,ib->IL', e_core_a,t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.166666667 * lib.einsum('I,Ia,Liab,ib->IL', e_core_a,t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.166666667 * lib.einsum('I,Ia,Liab,ib->IL', e_core_a,t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.333333334 * lib.einsum('I,Iiab,La,ib->IL', e_core_a,t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.166666667 * lib.einsum('L,Iiab,La,ib->IL', e_core_a,t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.333333334 * lib.einsum('L,Ia,Liab,ib->IL', e_core_a,t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.333333334 * lib.einsum('L,Ia,Liab,ib->IL', e_core_a,t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.166666667 * lib.einsum('L,Iiab,La,ib->IL', e_core_a,t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.333333334 * lib.einsum('i,Iiab,ia,Lb->IL', e_core_a,t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.333333334 * lib.einsum('i,Liab,ia,Ib->IL', e_core_a,t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.166666667 * lib.einsum('i,ia,Iiab,Lb->IL', e_core_a,t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.166666667 * lib.einsum('i,ia,Liab,Ib->IL', e_core_a,t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.500000001 * lib.einsum('i,iD,ijAa,ja->DA', e_core_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.500000001 * lib.einsum('i,iD,ijAa,ja->DA', e_core_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.333333334 * lib.einsum('i,ijAa,ia,jD->DA', e_core_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.500000001 * lib.einsum('i,ijDa,iA,ja->DA', e_core_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.333333334 * lib.einsum('i,ijDa,ia,jA->DA', e_core_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.166666667 * lib.einsum('i,ia,ijAa,jD->DA', e_core_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.166666667 * lib.einsum('i,ia,ijDa,jA->DA', e_core_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.500000001 * lib.einsum('i,ijDa,iA,ja->DA', e_core_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.500000001 * lib.einsum('a,La,Iiab,ib->IL', e_extern_a,t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.500000001 * lib.einsum('a,La,Iiab,ib->IL', e_extern_a,t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.500000001 * lib.einsum('a,Liab,Ia,ib->IL', e_extern_a,t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.500000001 * lib.einsum('a,Liab,ia,Ib->IL', e_extern_a,t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.500000001 * lib.einsum('a,Liab,Ia,ib->IL', e_extern_a,t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.500000001 * lib.einsum('a,ia,Iiab,Lb->IL', e_extern_a,t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.500000001 * lib.einsum('a,ia,ijDa,jA->DA', e_extern_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.500000001 * lib.einsum('a,ijAa,ia,jD->DA', e_extern_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.333333334 * lib.einsum('i,Iiab,ib,La->IL', e_core_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.333333334 * lib.einsum('i,Liab,ib,Ia->IL', e_core_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.166666667 * lib.einsum('i,ia,Iiba,Lb->IL', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] += 0.166666667 * lib.einsum('i,ia,Liba,Ib->IL', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.166666667 * lib.einsum('i,ia,jiAa,jD->DA', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.166666667 * lib.einsum('i,ia,jiDa,jA->DA', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.333333334 * lib.einsum('i,jiAa,ia,jD->DA', e_core_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] += 0.333333334 * lib.einsum('i,jiDa,ia,jA->DA', e_core_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.500000001 * lib.einsum('a,Liba,ia,Ib->IL', e_extern_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_aa[:,vir_list_a,:,vir_list_a] -= 0.500000001 * lib.einsum('a,ia,Iiba,Lb->IL', e_extern_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.500000001 * lib.einsum('a,ia,jiDa,jA->DA', e_extern_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aa[occ_list_a,:,occ_list_a,:] -= 0.500000001 * lib.einsum('a,ijAa,ja,iD->DA', e_extern_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
########################################################################################################################################################################

            M_030_bb -= lib.einsum('ja,idjl->idla', t2_ce_bb, v_cecc_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('ja,jdil->idla', t2_ce_bb, v_cecc_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('jd,laji->idla', t2_ce_bb, v_cecc_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('jd,jali->idla', t2_ce_bb, v_cecc_bbbb, optimize = einsum_type)
            
            M_030_bb += lib.einsum('ijdb,labj->idla', t2_ccee_bbbb, v_ceec_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ijdb,jabl->idla', t2_ccee_bbbb, v_ceec_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ib,lbda->idla', t2_ce_bb, v_ceee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('ib,ladb->idla', t2_ce_bb, v_ceee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('ljab,idbj->idla', t2_ccee_bbbb, v_ceec_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ljab,jdbi->idla', t2_ccee_bbbb, v_ceec_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('lb,ibad->idla', t2_ce_bb, v_ceee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('lb,idab->idla', t2_ce_bb, v_ceee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('jibd,labj->idla', t2_ccee_abab, v_ceec_bbaa, optimize = einsum_type)
            M_030_bb += lib.einsum('jlba,idbj->idla', t2_ccee_abab, v_ceec_bbaa, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('a,ijdb,ljab->idla', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('a,ljab,ijdb->idla', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('a,jibd,jlba->idla', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('a,jlba,jibd->idla', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('d,ijdb,ljab->idla', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('d,ljab,ijdb->idla', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('d,jibd,jlba->idla', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('d,jlba,jibd->idla', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('i,ijdb,ljab->idla', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('i,ljab,ijdb->idla', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('i,jibd,jlba->idla', e_core_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('i,jlba,jibd->idla', e_core_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('l,ijdb,ljab->idla', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('l,ljab,ijdb->idla', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('l,jibd,jlba->idla', e_core_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('l,jlba,jibd->idla', e_core_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('j,jibd,jlba->idla', e_core_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('j,jlba,jibd->idla', e_core_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('b,jibd,jlba->idla', e_extern_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('b,jlba,jibd->idla', e_extern_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('j,ijdb,ljab->idla', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('j,ljab,ijdb->idla', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('b,ijdb,ljab->idla', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('b,ljab,ijdb->idla', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] -= lib.einsum('ja,jd->da', h_ce_bb,t2_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] -= lib.einsum('jd,ja->da', h_ce_bb,t2_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ib,lb->il', h_ce_bb,t2_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lb,ib->il', h_ce_bb,t2_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('ja,ijdb,lb->idla', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jd,ljab,ib->idla', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('ib,ljab,jd->idla', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('lb,ijdb,ja->idla', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ijbc,lbcj->il', t2_ccee_bbbb, v_ceec_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ljbc,ibcj->il', t2_ccee_bbbb, v_ceec_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jibc,lcbj->il', t2_ccee_abab, v_ceec_bbaa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jlbc,icbj->il', t2_ccee_abab, v_ceec_bbaa, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jb,jbil->il', t2_ce_aa, v_cecc_aabb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jb,jbli->il', t2_ce_aa, v_cecc_aabb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jb,jbil->il', t2_ce_bb, v_cecc_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jb,lbji->il', t2_ce_bb, v_cecc_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jb,jbli->il', t2_ce_bb, v_cecc_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jb,ibjl->il', t2_ce_bb, v_cecc_bbbb, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b] += lib.einsum('jb,jbad->da',  t2_ce_aa, v_ceee_aabb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] -= lib.einsum('jkba,kdbj->da',t2_ccee_abab, v_ceec_bbaa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] -= lib.einsum('jkbd,kabj->da',t2_ccee_abab, v_ceec_bbaa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] += lib.einsum('jb,jbda->da',  t2_ce_aa, v_ceee_aabb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] -= lib.einsum('jkab,jdbk->da', t2_ccee_bbbb, v_ceec_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] -= lib.einsum('jkdb,jabk->da', t2_ccee_bbbb, v_ceec_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] += lib.einsum('jb,jbad->da', t2_ce_bb, v_ceee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] -= lib.einsum('jb,jadb->da', t2_ce_bb, v_ceee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] += lib.einsum('jb,jbda->da', t2_ce_bb, v_ceee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b] -= lib.einsum('jb,jdab->da', t2_ce_bb, v_ceee_bbbb, optimize = einsum_type)
            M_030_bb += 1/4 * lib.einsum('ijad,jkbc,lkbc->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('ijad,jb,lb->idla', v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('ijad,kjbc,klbc->idla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('ibad,ljbc,jc->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('ibad,jlcb,jc->idla', v_ceee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb += lib.einsum('jkad,jlbc,kibc->idla', v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jbad,jlbc,ic->idla', v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jbda,jibc,lc->idla', v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/4 * lib.einsum('jlad,jkbc,ikbc->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('jlad,jb,ib->idla', v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('jlad,kjbc,kibc->idla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('jkad,ljbc,ikbc->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('jbad,ljbc,ic->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('lbda,ijbc,jc->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('lbda,jicb,jc->idla', v_ceee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('jbda,ijbc,lc->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb -= 1/2 * lib.einsum('laji,jkdb,kb->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('laji,kjbd,kb->idla', v_cecc_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_bb -= 1/2 * lib.einsum('labi,jb,jd->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('labi,jkcb,jkcd->idla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/4 * lib.einsum('labi,jkbc,jkdc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('lajk,kb,jibd->idla', v_cecc_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('labj,jkbc,kicd->idla', v_ceec_bbaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('labj,jkbc,ikdc->idla', v_ceec_bbaa, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('labc,jicd,jb->idla', v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb -= 1/4 * lib.einsum('ladj,jkbc,ikbc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('ladj,jb,ib->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('ladj,kjbc,kibc->idla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('lajk,kb,ijdb->idla', v_cecc_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb += 1/2 * lib.einsum('labj,jkbc,ikdc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('labj,kjcb,kicd->idla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('ladb,ijbc,jc->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('ladb,jicb,jc->idla', v_ceee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb -= lib.einsum('labj,ib,jd->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += lib.einsum('labc,ijdc,jb->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/4 * lib.einsum('jadi,jkbc,lkbc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jadi,jb,lb->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jadi,kjbc,klbc->idla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb += 1/2 * lib.einsum('jali,jkdb,kb->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('jali,kjbd,kb->idla', v_cecc_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('kaji,kjdb,lb->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb += lib.einsum('jabi,jd,lb->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += lib.einsum('jabi,jkdc,lkbc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('jabi,kjcd,klcb->idla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('jabk,ijdc,klbc->idla', v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jadk,ijbc,lkbc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb += lib.einsum('kalj,ikdb,jb->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb += lib.einsum('jabk,ijdc,lkbc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jadb,ijbc,lc->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += lib.einsum('jabl,jd,ib->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= lib.einsum('jabc,ijdc,lb->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('idab,ljbc,jc->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('idab,jlcb,jc->idla', v_ceee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('ilab,jb,jd->idla', v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('ilab,jkcb,jkcd->idla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/4 * lib.einsum('ilab,jkbc,jkdc->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ijab,lb,jd->idla', v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ijab,lkbc,jkdc->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ijab,klcb,kjcd->idla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('ibac,ljcb,jd->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= lib.einsum('jkab,jlcb,kicd->idla', v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('jbca,lc,jibd->idla', v_ceee_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jdab,ljbc,ic->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jlab,jkbc,ikdc->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jlab,kjcb,kicd->idla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('jkab,ljbc,ikdc->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('lbca,jc,ijdb->idla', v_ceee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('jbca,lc,ijdb->idla', v_ceee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb -= 1/2 * lib.einsum('idjl,jkab,kb->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('idjl,kjba,kb->idla', v_cecc_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_bb -= 1/2 * lib.einsum('idbl,jb,ja->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('idbl,jkcb,jkca->idla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/4 * lib.einsum('idbl,jkbc,jkac->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('idjk,kb,jlba->idla', v_cecc_bbaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('idbj,jkbc,klca->idla', v_ceec_bbaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('idbj,jkbc,lkac->idla', v_ceec_bbaa, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('idbc,jlca,jb->idla', v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('idjk,kb,ljab->idla', v_cecc_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb += 1/2 * lib.einsum('idbj,jkbc,lkac->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('idbj,kjcb,klca->idla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('idbj,lb,ja->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += lib.einsum('idbc,ljac,jb->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb += 1/2 * lib.einsum('jdil,jkab,kb->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('jdil,kjba,kb->idla', v_cecc_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('kdjl,kjab,ib->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb += lib.einsum('jdbl,ja,ib->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += lib.einsum('jdbl,jkac,ikbc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('jdbl,kjca,kicb->idla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('jdbk,ljac,kibc->idla', v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb += lib.einsum('kdij,lkab,jb->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb += lib.einsum('jdbk,ljac,ikbc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('jdbi,ja,lb->idla', v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= lib.einsum('jdbc,ljac,ib->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('lidb,jb,ja->idla', v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('lidb,jkcb,jkca->idla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += 1/4 * lib.einsum('lidb,jkbc,jkac->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ljdb,ib,ja->idla', v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ljdb,ikbc,jkac->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ljdb,kicb,kjca->idla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('lbdc,ijcb,ja->idla', v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= lib.einsum('jkdb,jicb,klca->idla', v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('jbcd,ic,jlba->idla', v_ceee_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jidb,jkbc,lkac->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jidb,kjcb,klca->idla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('jkdb,ijbc,lkac->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ibcd,jc,ljab->idla', v_ceee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('jbcd,ic,ljab->idla', v_ceee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('jkil,jmba,kmbd->idla', v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb += 1/2 * lib.einsum('jbil,jkba,kd->idla', v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('jbli,jkbd,ka->idla', v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb += lib.einsum('ilbc,jkbd,jkca->idla', v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('iljk,ja,kd->idla', v_cccc_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= lib.einsum('iljk,jmab,kmdb->idla', v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('iljk,mjba,mkbd->idla', v_cccc_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb -= 1/2 * lib.einsum('jbil,jkab,kd->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 1/2 * lib.einsum('jbli,jkdb,ka->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb += 1/2 * lib.einsum('ilbc,jkdb,jkac->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('kmij,mjbd,klba->idla', v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('kbij,jd,klba->idla', v_cecc_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('ijbc,kjbd,klca->idla', v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('ijkl,jd,ka->idla', v_cccc_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += lib.einsum('ijkl,jmdb,kmab->idla', v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('ijkl,mjbd,mkba->idla', v_cccc_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('ijkm,jmdb,lkab->idla', v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('kbij,jd,lkab->idla', v_cecc_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 1/2 * lib.einsum('lbji,jkdb,ka->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('ijbc,jkdb,lkac->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('ibcj,lkab,jkcd->idla', v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb += 1/2 * lib.einsum('ibjl,jkab,kd->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += lib.einsum('ibjk,ljab,kd->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb -= 1/2 * lib.einsum('ibcl,jkab,jkdc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('ibcj,lkab,jkdc->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('kmlj,mjba,kibd->idla', v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('kblj,ja,kibd->idla', v_cecc_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('ljbc,kjba,kicd->idla', v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('ljkm,jmab,ikdb->idla', v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('kblj,ja,ikdb->idla', v_cecc_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb -= lib.einsum('ljbc,jkab,ikdc->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('lbcj,ikdb,jkca->idla', v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb += lib.einsum('lbjk,ijdb,ka->idla', v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb += lib.einsum('lbcj,ikdb,jkac->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= lib.einsum('jkbc,jicd,klba->idla', v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('jbck,jibd,klca->idla', v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb -= lib.einsum('jkbc,ijdc,lkab->idla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += lib.einsum('jbck,ijdb,klca->idla', v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('jbck,ljab,kicd->idla', v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb += lib.einsum('jbck,ijdb,lkac->idla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('a,ja,jd->da', e_extern_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('a,jd,ja->da', e_extern_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('a,jkba,jkbd->da', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('a,jkbd,jkba->da', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('a,jkab,jkdb->da', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('a,jkdb,jkab->da', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('d,ja,jd->da', e_extern_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('d,jd,ja->da', e_extern_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('d,jkba,jkbd->da', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('d,jkbd,jkba->da', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('d,jkab,jkdb->da', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('d,jkdb,jkab->da', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/4 * lib.einsum('i,ijbc,ljbc->il', e_core_b,  t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('i,ib,lb->il', e_core_b,  t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/4 * lib.einsum('i,ljbc,ijbc->il', e_core_b,  t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('i,lb,ib->il', e_core_b,  t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('i,jibc,jlbc->il', e_core_b,  t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('i,jlbc,jibc->il', e_core_b,  t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/4 * lib.einsum('l,ijbc,ljbc->il', e_core_b,  t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('l,ib,lb->il', e_core_b,  t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/4 * lib.einsum('l,ljbc,ijbc->il', e_core_b,  t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('l,lb,ib->il', e_core_b,  t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('l,jibc,jlbc->il', e_core_b,  t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('l,jlbc,jibc->il', e_core_b,  t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('j,jibc,jlbc->il', e_core_a,  t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('j,jlbc,jibc->il', e_core_a,  t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('j,jkba,jkbd->da', e_core_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('j,jkbd,jkba->da', e_core_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('b,jibc,jlbc->il', e_extern_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('b,jlbc,jibc->il', e_extern_a,t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('b,jkba,jkbd->da', e_extern_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('b,jkbd,jkba->da', e_extern_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('j,ijbc,ljbc->il', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('j,ljbc,ijbc->il', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('j,ja,jd->da', e_core_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('j,jd,ja->da', e_core_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('j,jkab,jkdb->da', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('j,jkdb,jkab->da', e_core_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('j,kjba,kjbd->da', e_core_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('j,kjbd,kjba->da', e_core_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('b,ib,lb->il', e_extern_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('b,ijbc,ljbc->il', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('b,lb,ib->il', e_extern_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('b,ljbc,ijbc->il', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('b,jicb,jlcb->il', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('b,jlcb,jicb->il', e_extern_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('b,jkab,jkdb->da', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('b,jkdb,jkab->da', e_extern_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 0.166666667 * lib.einsum('a,ja,ijdb,lb->idla', e_extern_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 0.333333334 * lib.einsum('a,jd,ib,ljab->idla', e_extern_b, t1_ce_bb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb -= 0.333333334 * lib.einsum('d,ja,ijdb,lb->idla', e_extern_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 0.166666667 * lib.einsum('d,jd,ib,ljab->idla', e_extern_b, t1_ce_bb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 0.333333334 * lib.einsum('i,ja,ijdb,lb->idla', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 0.166666667 * lib.einsum('i,jd,ib,ljab->idla', e_core_b, t1_ce_bb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 0.166666667 * lib.einsum('l,ja,ijdb,lb->idla', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 0.333333334 * lib.einsum('l,jd,ib,ljab->idla', e_core_b, t1_ce_bb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb += 0.500000001 * lib.einsum('j,jd,ljab,ib->idla', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb += 0.500000001 * lib.einsum('j,ijdb,ja,lb->idla', e_core_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 0.500000001 * lib.einsum('b,lb,ijdb,ja->idla', e_extern_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb -= 0.500000001 * lib.einsum('b,ljab,ib,jd->idla', e_extern_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('ja,jkdb,kb->da', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('ja,kjbd,kb->da', h_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jd,jkab,kb->da', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jd,kjba,kb->da', h_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('ib,ljbc,jc->il', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('ib,jlcb,jc->il', h_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('lb,ijbc,jc->il', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('lb,jicb,jc->il', h_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jb,jibc,lc->il', h_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jb,jlbc,ic->il', h_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jb,jkba,kd->da', h_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jb,jkbd,ka->da', h_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('jb,ijbc,lc->il', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('jb,ljbc,ic->il', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jb,jkab,kd->da', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jb,jkdb,ka->da', h_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('jkil,jmbc,kmbc->il',  v_cccc_aabb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jkil,jb,kb->il',  v_cccc_aabb, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jkil,jmbc,kmbc->il',  v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jbil,jkbc,kc->il',  v_cecc_aabb, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jbil,jkbc,kc->il',  v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jbli,jkbc,kc->il',  v_cecc_aabb, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jbli,jkbc,kc->il',  v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ilbc,jb,jc->il',  v_ccee_bbaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('ilbc,jkbe,jkce->il',  v_ccee_bbaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ilbc,jkbe,jkce->il',  v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('iljk,jmbc,kmbc->il',  v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('iljk,jb,kb->il',  v_cccc_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('iljk,mjbc,mkbc->il',  v_cccc_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jbil,jkbc,kc->il',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jbil,kjcb,kc->il',  v_cecc_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jbli,jkbc,kc->il',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('jbli,kjcb,kc->il',  v_cecc_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ilbc,jb,jc->il',  v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ilbc,jkeb,jkec->il',  v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('ilbc,jkbe,jkce->il',  v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('kmij,mjbc,klbc->il',  v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('kbij,jc,klbc->il',  v_cecc_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('kbji,kjbc,lc->il',  v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ijbc,kjbe,klce->il',  v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('ijkl,jmbc,kmbc->il',  v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ijkl,jb,kb->il',  v_cccc_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ijkl,mjbc,mkbc->il',  v_cccc_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('ijkm,jmbc,lkbc->il',  v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('kbij,jc,lkbc->il',  v_cecc_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('lbji,jkbc,kc->il',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('lbji,kjcb,kc->il',  v_cecc_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('kbji,jkbc,lc->il',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ijbc,jb,lc->il',  v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ijbc,jkbe,lkce->il',  v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ijbc,kjeb,klec->il',  v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ibjk,jlcb,kc->il',  v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibcj,lb,jc->il',  v_ceec_bbaa, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibcj,lb,jc->il',  v_ceec_bbaa, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibcj,lkbe,jkce->il',  v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibcj,kleb,jkce->il',  v_ceec_bbaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibce,jleb,jc->il',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('ibjl,jkbc,kc->il',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('ibjl,kjcb,kc->il',  v_cecc_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ibjk,ljbc,kc->il',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibcj,lb,jc->il',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ibcj,jb,lc->il',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ibcl,jb,jc->il',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ibcl,jkeb,jkec->il',  v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('ibcl,jkbe,jkce->il',  v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibcj,lb,jc->il',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibcj,lkbe,jkce->il',  v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibcj,kleb,kjec->il',  v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('ibce,ljbe,jc->il',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('kmlj,mjbc,kibc->il',  v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('kblj,jc,kibc->il',  v_cecc_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 1/2 * lib.einsum('kbjl,kjbc,ic->il',  v_cecc_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ljbc,kjbe,kice->il',  v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('ljkm,jmbc,ikbc->il',  v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('kblj,jc,ikbc->il',  v_cecc_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 1/2 * lib.einsum('kbjl,jkbc,ic->il',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ljbc,jb,ic->il',  v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ljbc,jkbe,ikce->il',  v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('ljbc,kjeb,kiec->il',  v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('lbjk,jicb,kc->il',  v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbcj,ib,jc->il',  v_ceec_bbaa, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbcj,ib,jc->il',  v_ceec_bbaa, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbcj,ikbe,jkce->il',  v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbcj,kieb,jkce->il',  v_ceec_bbaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbce,jieb,jc->il',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('lbjk,ijbc,kc->il',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbcj,ib,jc->il',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('lbcj,jb,ic->il',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbcj,ib,jc->il',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbcj,ikbe,jkce->il',  v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbcj,kieb,kjec->il',  v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('lbce,ijbe,jc->il',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jkbc,jice,klbe->il',  v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jkbc,jiec,kleb->il',  v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jbck,jibe,klce->il',  v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jkbc,ijce,lkbe->il',  v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jbck,ijbe,klce->il',  v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jbck,ljbe,kice->il',  v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jbck,ijbe,lkce->il',  v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += lib.einsum('jbce,ijbe,lc->il',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jbce,ic,jlbe->il',  v_ceee_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jbce,lc,jibe->il',  v_ceee_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= lib.einsum('jbce,ic,ljeb->il',  v_ceee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jkad,jmbc,kmbc->da',  v_ccee_aabb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jkad,jb,kb->da',  v_ccee_aabb, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jkad,jmbc,kmbc->da',  v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbad,jkbc,kc->da',  v_ceee_aabb, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbad,jkbc,kc->da',  v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbda,jkbc,kc->da',  v_ceee_aabb, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbda,jkbc,kc->da',  v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jkad,jmbc,kmbc->da',  v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jkad,jb,kb->da',  v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jkad,mjbc,mkbc->da',  v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbad,jkbc,kc->da',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbad,kjcb,kc->da',  v_ceee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbda,jkbc,kc->da',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbda,kjcb,kc->da',  v_ceee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('majk,jmbd,kb->da',  v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabk,jd,kb->da',  v_ceec_bbaa, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabk,jmdc,kmbc->da',  v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabk,mjcd,kmbc->da',  v_ceec_bbaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabk,jd,kb->da',  v_ceec_bbaa, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabc,kjcd,kb->da',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jadk,jmbc,kmbc->da',  v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jadk,jb,kb->da',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jadk,mjbc,mkbc->da',  v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('majk,mjdb,kb->da',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabk,jd,kb->da',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabk,jmdc,kmbc->da',  v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabk,mjcd,mkcb->da',  v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jadb,jkbc,kc->da',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jadb,kjcb,kc->da',  v_ceee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabk,jd,kb->da',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jabk,jb,kd->da',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jabc,jkdc,kb->da',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkab,jmcb,kmcd->da',  v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jbac,jkbc,kd->da',  v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jbca,kc,jkbd->da',  v_ceee_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jdab,jkbc,kc->da',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jdab,kjcb,kc->da',  v_ceee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkab,jb,kd->da',  v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkab,jmbc,kmdc->da',  v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkab,mjcb,mkcd->da',  v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbac,jkcb,kd->da',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jbca,kc,jkdb->da',  v_ceee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('mdjk,jmba,kb->da',  v_cecc_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbk,ja,kb->da',  v_ceec_bbaa, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbk,jmac,kmbc->da',  v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbk,mjca,kmbc->da',  v_ceec_bbaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbk,ja,kb->da',  v_ceec_bbaa, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbc,kjca,kb->da',  v_ceee_bbaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('mdjk,mjab,kb->da',  v_cecc_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbk,ja,kb->da',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbk,jmac,kmbc->da',  v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbk,mjca,mkcb->da',  v_ceec_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbk,ja,kb->da',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jdbk,jb,ka->da',  v_ceec_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jdbc,jkac,kb->da',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkdb,jmcb,kmca->da',  v_ccee_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jbdc,jkbc,ka->da',  v_ceee_aabb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jbcd,kc,jkba->da',  v_ceee_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkdb,jb,ka->da',  v_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkdb,jmbc,kmac->da',  v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkdb,mjcb,mkca->da',  v_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jbdc,jkcb,ka->da',  v_ceee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jbcd,kc,jkab->da',  v_ceee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkbc,jmca,kmbd->da',  v_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jkmn,jmba,knbd->da',  v_cccc_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jbck,jmba,kmcd->da',  v_ceec_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('mbjk,ka,mjbd->da',  v_cecc_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('mbjk,kd,mjba->da',  v_cecc_aabb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkbc,mjca,mkbd->da',  v_ccee_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jkmn,jmab,kndb->da',  v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('mbjk,ka,jmdb->da',  v_cecc_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('mbjk,kd,jmab->da',  v_cecc_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jkbc,jmac,kmdb->da',  v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jbck,jmab,kmcd->da',  v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += lib.einsum('jbck,jmdb,kmca->da',  v_ceec_bbaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= lib.einsum('jbck,jmab,kmdc->da',  v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.166666667 * lib.einsum('a,ja,jkdb,kb->da', e_extern_b,  t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.166666667 * lib.einsum('a,ja,kjbd,kb->da', e_extern_b,  t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.333333334 * lib.einsum('a,jd,jkab,kb->da', e_extern_b,  t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.333333334 * lib.einsum('a,jd,kjba,kb->da', e_extern_b,  t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.333333334 * lib.einsum('d,ja,jkdb,kb->da', e_extern_b,  t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.333333334 * lib.einsum('d,ja,kjbd,kb->da', e_extern_b,  t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.166666667 * lib.einsum('d,jd,jkab,kb->da', e_extern_b,  t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.166666667 * lib.einsum('d,jd,kjba,kb->da', e_extern_b,  t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.333333334 * lib.einsum('i,ijbc,lb,jc->il', e_core_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.166666667 * lib.einsum('i,ib,ljbc,jc->il', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.166666667 * lib.einsum('i,ib,jlcb,jc->il', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.333333334 * lib.einsum('i,lb,jicb,jc->il', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.166666667 * lib.einsum('l,ijbc,lb,jc->il', e_core_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.333333334 * lib.einsum('l,ib,ljbc,jc->il', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.333333334 * lib.einsum('l,ib,jlcb,jc->il', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.166666667 * lib.einsum('l,lb,jicb,jc->il', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.333333334 * lib.einsum('j,jibc,jb,lc->il', e_core_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.333333334 * lib.einsum('j,jlbc,jb,ic->il', e_core_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.166666667 * lib.einsum('j,jb,jibc,lc->il', e_core_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.166666667 * lib.einsum('j,jb,jlbc,ic->il', e_core_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.166666667 * lib.einsum('j,jb,jkba,kd->da', e_core_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.166666667 * lib.einsum('j,jb,jkbd,ka->da', e_core_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.333333334 * lib.einsum('j,jkba,jb,kd->da', e_core_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.333333334 * lib.einsum('j,jkbd,jb,ka->da', e_core_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.500000001 * lib.einsum('b,jb,jibc,lc->il', e_extern_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.500000001 * lib.einsum('b,jlbc,jb,ic->il', e_extern_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.500000001 * lib.einsum('b,jb,jkbd,ka->da', e_extern_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.500000001 * lib.einsum('b,jkba,jb,kd->da', e_extern_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.333333334 * lib.einsum('j,ijbc,jb,lc->il', e_core_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.333333334 * lib.einsum('j,ljbc,jb,ic->il', e_core_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.166666667 * lib.einsum('j,jb,ijbc,lc->il', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.166666667 * lib.einsum('j,jb,ljbc,ic->il', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.500000001 * lib.einsum('j,jd,jkab,kb->da', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.500000001 * lib.einsum('j,jd,kjba,kb->da', e_core_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.333333334 * lib.einsum('j,jkab,jb,kd->da', e_core_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.500000001 * lib.einsum('j,jkdb,ja,kb->da', e_core_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.333333334 * lib.einsum('j,jkdb,jb,ka->da', e_core_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.166666667 * lib.einsum('j,jb,jkab,kd->da', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] -= 0.166666667 * lib.einsum('j,jb,jkdb,ka->da', e_core_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.500000001 * lib.einsum('j,kjbd,ja,kb->da', e_core_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.500000001 * lib.einsum('b,lb,ijbc,jc->il', e_extern_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.500000001 * lib.einsum('b,lb,jicb,jc->il', e_extern_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.500000001 * lib.einsum('b,ljbc,ib,jc->il', e_extern_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.500000001 * lib.einsum('b,ljbc,jb,ic->il', e_extern_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] += 0.500000001 * lib.einsum('b,jb,ijbc,lc->il', e_extern_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[:,vir_list_b,:,vir_list_b] -= 0.500000001 * lib.einsum('b,jlcb,ib,jc->il', e_extern_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.500000001 * lib.einsum('b,jb,jkdb,ka->da', e_extern_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_bb[occ_list_b,:,occ_list_b,:] += 0.500000001 * lib.einsum('b,jkab,jb,kd->da', e_extern_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('iD,laiI->IDla', t2_ce_aa, v_cecc_bbaa, optimize = einsum_type)
            M_030_aabb += lib.einsum('IiDb,labi->IDla', t2_ccee_aaaa, v_ceec_bbaa, optimize = einsum_type)
            M_030_aabb += lib.einsum('Ib,laDb->IDla', t2_ce_aa, v_ceee_bbaa, optimize = einsum_type)
            M_030_aabb += lib.einsum('IiDb,labi->IDla', t2_ccee_abab, v_ceec_bbbb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('IiDb,iabl->IDla', t2_ccee_abab, v_ceec_bbbb, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('ia,IDil->IDla', t2_ce_bb, v_cecc_aabb, optimize = einsum_type)
            
            M_030_aabb += lib.einsum('liab,IDbi->IDla', t2_ccee_bbbb, v_ceec_aabb, optimize = einsum_type)
            M_030_aabb += lib.einsum('lb,IDab->IDla', t2_ce_bb, v_ceee_aabb, optimize = einsum_type)
            M_030_aabb += lib.einsum('ilba,IDbi->IDla', t2_ccee_abab, v_ceec_aaaa, optimize = einsum_type)
            M_030_aabb -= lib.einsum('ilba,iDbI->IDla', t2_ccee_abab, v_ceec_aaaa, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('D,IiDb,ilba->IDla', e_extern_a, t1_ccee_aaaa, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('D,IiDb,liab->IDla', e_extern_a, t1_ccee_abab, t2_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('D,liab,IiDb->IDla', e_extern_a, t1_ccee_bbbb, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('D,ilba,IiDb->IDla', e_extern_a, t1_ccee_abab, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('I,IiDb,ilba->IDla', e_core_a, t1_ccee_aaaa, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('I,IiDb,liab->IDla', e_core_a, t1_ccee_abab, t2_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('I,liab,IiDb->IDla', e_core_a, t1_ccee_bbbb, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('I,ilba,IiDb->IDla', e_core_a, t1_ccee_abab, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('a,IiDb,ilba->IDla', e_extern_b, t1_ccee_aaaa, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('a,IiDb,liab->IDla', e_extern_b, t1_ccee_abab, t2_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('a,liab,IiDb->IDla', e_extern_b, t1_ccee_bbbb, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('a,ilba,IiDb->IDla', e_extern_b, t1_ccee_abab, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('l,IiDb,ilba->IDla', e_core_b, t1_ccee_aaaa, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('l,IiDb,liab->IDla', e_core_b, t1_ccee_abab, t2_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('l,liab,IiDb->IDla', e_core_b, t1_ccee_bbbb, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('l,ilba,IiDb->IDla', e_core_b, t1_ccee_abab, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= lib.einsum('i,IiDb,ilba->IDla', e_core_a, t1_ccee_aaaa, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('i,ilba,IiDb->IDla', e_core_a, t1_ccee_abab, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aabb += lib.einsum('b,IiDb,ilba->IDla', e_extern_a, t1_ccee_aaaa, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('b,ilba,IiDb->IDla', e_extern_a, t1_ccee_abab, t2_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= lib.einsum('i,IiDb,liab->IDla', e_core_b, t1_ccee_abab, t2_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('i,liab,IiDb->IDla', e_core_b, t1_ccee_bbbb, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('b,IiDb,liab->IDla', e_extern_b, t1_ccee_abab, t2_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += lib.einsum('b,liab,IiDb->IDla', e_extern_b, t1_ccee_bbbb, t2_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('iD,ilba,Ib->IDla', h_ce_aa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('Ib,ilba,iD->IDla', h_ce_aa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('ia,IiDb,lb->IDla', h_ce_bb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('lb,IiDb,ia->IDla', h_ce_bb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_aabb -= 1/2 * lib.einsum('IDil,ijab,jb->IDla', v_cecc_aabb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('IDil,jiba,jb->IDla', v_cecc_aabb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_aabb -= 1/2 * lib.einsum('IDbl,ib,ia->IDla', v_ceec_aabb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('IDbl,ijcb,ijca->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/4 * lib.einsum('IDbl,ijbc,ijac->IDla', v_ceec_aabb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)

            M_030_aabb -= lib.einsum('IDij,jb,ilba->IDla', v_cecc_aaaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)

            M_030_aabb += 1/2 * lib.einsum('IDbi,ijbc,jlca->IDla', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('IDbi,ijbc,ljac->IDla', v_ceec_aaaa, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += lib.einsum('IDbc,ilca,ib->IDla', v_ceee_aaaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aabb -= 1/4 * lib.einsum('IDai,ijbc,ljbc->IDla', v_ceec_aabb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('IDai,ib,lb->IDla', v_ceec_aabb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('IDai,jibc,jlbc->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('IDij,jb,liab->IDla', v_cecc_aabb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            
            M_030_aabb += 1/2 * lib.einsum('IDbi,ijbc,ljac->IDla', v_ceec_aabb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('IDbi,jicb,jlca->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('IDab,libc,ic->IDla', v_ceee_aabb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('IDab,ilcb,ic->IDla', v_ceee_aabb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aabb -= lib.einsum('IDbi,lb,ia->IDla', v_ceec_aabb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb += lib.einsum('IDbc,liac,ib->IDla', v_ceee_aabb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= 1/4 * lib.einsum('iDal,ijbc,Ijbc->IDla', v_ceec_aabb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('iDal,ib,Ib->IDla', v_ceec_aabb, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('iDal,ijbc,Ijbc->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aabb += 1/2 * lib.einsum('jDil,jiba,Ib->IDla', v_cecc_aabb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_aabb += lib.einsum('iDbl,ijca,Ijcb->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)

            M_030_aabb += lib.einsum('jDIi,jlba,ib->IDla', v_cecc_aaaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)

            M_030_aabb += lib.einsum('iDbj,ilca,Ijbc->IDla', v_ceec_aaaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= lib.einsum('iDbc,ilca,Ib->IDla', v_ceee_aaaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aabb += lib.einsum('iDaj,ilbc,Ijbc->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('iDbj,ilca,Ijcb->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('iDab,ilcb,Ic->IDla', v_ceee_aabb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('laDb,Iibc,ic->IDla', v_ceee_bbaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('laDb,Iibc,ic->IDla', v_ceee_bbaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('liDb,Ib,ia->IDla', v_ccee_bbaa, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('liDb,Ijbc,jica->IDla', v_ccee_bbaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('liDb,Ijbc,ijac->IDla', v_ccee_bbaa, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('lbDc,Iicb,ia->IDla', v_ceee_bbaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('iIDb,ijbc,jlca->IDla', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('iIDb,ijbc,ljac->IDla', v_ccee_aaaa, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('ijDb,Iibc,jlca->IDla', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('IbcD,ic,ilba->IDla', v_ceee_aaaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('ibcD,Ic,ilba->IDla', v_ceee_aaaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('iaDb,Iibc,lc->IDla', v_ceee_bbaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('ijDb,Iibc,ljac->IDla', v_ccee_bbaa, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += lib.einsum('ibcD,Ic,liab->IDla', v_ceee_bbaa, t1_ce_aa, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('laiI,ijDb,jb->IDla', v_cecc_bbaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('laiI,ijDb,jb->IDla', v_cecc_bbaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('jaiI,ijDb,lb->IDla', v_cecc_bbaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('Iiab,iD,lb->IDla', v_ccee_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('Iiab,ijDc,jlcb->IDla', v_ccee_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('Iiab,ijDc,ljbc->IDla', v_ccee_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += lib.einsum('Iijk,ikDb,jlba->IDla', v_cccc_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)

            M_030_aabb -= lib.einsum('jbIi,iD,jlba->IDla', v_cecc_aaaa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('Iibc,ijDb,jlca->IDla', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('Iijl,iD,ja->IDla', v_cccc_aabb, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            M_030_aabb += lib.einsum('Iijl,ikDb,kjba->IDla', v_cccc_aabb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('Iijl,ikDb,jkab->IDla', v_cccc_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += lib.einsum('Iijk,ikDb,ljab->IDla', v_cccc_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('jbIi,iD,ljab->IDla', v_cecc_bbaa, t1_ce_aa, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('lbiI,ijDb,ja->IDla', v_cecc_bbaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('Iibc,ijDb,ljac->IDla', v_ccee_aabb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('Ibal,ib,iD->IDla', v_ceec_aabb, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aabb -= 1/4 * lib.einsum('Ibal,ijbc,ijDc->IDla', v_ceec_aabb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('Ibal,ijbc,ijDc->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('Ibai,jlbc,jiDc->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('Ibac,ilbc,iD->IDla', v_ceee_aabb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)

            M_030_aabb += lib.einsum('Ibij,ilba,jD->IDla', v_cecc_aaaa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_aabb += lib.einsum('Ibci,jlba,ijDc->IDla', v_ceec_aaaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aabb += 1/2 * lib.einsum('Ibil,jiba,jD->IDla', v_cecc_aabb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_030_aabb += lib.einsum('Ibcl,ijba,ijDc->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('Ibci,jlba,jiDc->IDla', v_ceec_aabb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('laij,jb,IiDb->IDla', v_cecc_bbaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('labi,ijbc,IjDc->IDla', v_ceec_bbaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('labi,ijbc,IjDc->IDla', v_ceec_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('labi,Ib,iD->IDla', v_ceec_bbaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aabb += lib.einsum('labc,IiDc,ib->IDla', v_ceee_bbaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('laij,jb,IiDb->IDla', v_cecc_bbbb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aabb += 1/2 * lib.einsum('labi,ijbc,IjDc->IDla', v_ceec_bbbb, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += 1/2 * lib.einsum('labi,jicb,IjDc->IDla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb += lib.einsum('labc,IiDc,ib->IDla', v_ceee_bbbb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('iabj,IiDc,jlbc->IDla', v_ceec_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aabb += lib.einsum('jali,IjDb,ib->IDla', v_cecc_bbbb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_aabb += lib.einsum('iabj,IiDc,ljbc->IDla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('iabc,IiDc,lb->IDla', v_ceee_bbbb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('ijab,ilcb,IjDc->IDla', v_ccee_aabb, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb += lib.einsum('ibca,lc,IiDb->IDla', v_ceee_aabb, t1_ce_bb, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('ilab,ijbc,IjDc->IDla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 1/2 * lib.einsum('ilab,jicb,IjDc->IDla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb -= lib.einsum('ijab,libc,IjDc->IDla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('lbca,ic,IiDb->IDla', v_ceee_bbbb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('ibca,lc,IiDb->IDla', v_ceee_bbbb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('jkli,kiba,IjDb->IDla', v_cccc_aabb, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('jbli,ia,IjDb->IDla', v_cecc_aabb, t1_ce_bb, t1_ccee_aaaa, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('libc,jiba,IjDc->IDla', v_ccee_bbaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            M_030_aabb += lib.einsum('lijk,ikab,IjDb->IDla', v_cccc_bbbb, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('jbli,ia,IjDb->IDla', v_cecc_bbbb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aabb -= lib.einsum('libc,ijab,IjDc->IDla', v_ccee_bbbb, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= lib.einsum('lbci,IjDb,ijca->IDla', v_ceec_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            M_030_aabb += lib.einsum('lbij,IiDb,ja->IDla', v_cecc_bbbb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            
            M_030_aabb += lib.einsum('lbci,IjDb,ijac->IDla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('ijbc,IiDc,jlba->IDla', v_ccee_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('ibcj,IiDb,jlca->IDla', v_ceec_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('ibcj,IiDb,ljac->IDla', v_ceec_aabb, t1_ccee_aaaa, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= lib.einsum('ijbc,IiDc,ljab->IDla', v_ccee_bbbb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb += lib.einsum('ibcj,IiDb,jlca->IDla', v_ceec_bbaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += lib.einsum('ibcj,IiDb,ljac->IDla', v_ceec_bbbb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            M_030_aabb -= 0.166666667 * lib.einsum('D,iD,Ib,ilba->IDla', e_extern_a, t1_ce_aa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 0.333333334 * lib.einsum('D,IiDb,ia,lb->IDla', e_extern_a, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb += 0.166666667 * lib.einsum('I,iD,Ib,ilba->IDla', e_core_a, t1_ce_aa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += 0.333333334 * lib.einsum('I,IiDb,ia,lb->IDla', e_core_a, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= 0.333333334 * lib.einsum('a,iD,Ib,ilba->IDla', e_extern_b, t1_ce_aa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb -= 0.166666667 * lib.einsum('a,IiDb,ia,lb->IDla', e_extern_b, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb += 0.333333334 * lib.einsum('l,iD,Ib,ilba->IDla', e_core_b, t1_ce_aa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_030_aabb += 0.166666667 * lib.einsum('l,IiDb,ia,lb->IDla', e_core_b, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb += 0.500000001 * lib.einsum('i,iD,ilba,Ib->IDla', e_core_a, t1_ce_aa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_030_aabb -= 0.500000001 * lib.einsum('b,ilba,Ib,iD->IDla', e_extern_a, t1_ccee_abab, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            M_030_aabb += 0.500000001 * lib.einsum('i,IiDb,ia,lb->IDla', e_core_b, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            M_030_aabb -= 0.500000001 * lib.einsum('b,lb,IiDb,ia->IDla', e_extern_b, t1_ce_bb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)

            M_ia_jb_a += M_030_aa
            M_ia_jb_b += M_030_bb
            M_aabb += M_030_aabb

    M_ia_jb_a = M_ia_jb_a.reshape(n_singles_a, n_singles_a)
    M_ia_jb_b = M_ia_jb_b.reshape(n_singles_b, n_singles_b)
    M_aabb = M_aabb.reshape(n_singles_a, n_singles_b)


    if not isinstance(eris.oovv, np.ndarray):
        M_ia_jb_a = radc_ao2mo.write_dataset(M_ia_jb_a)
        M_ia_jb_b = radc_ao2mo.write_dataset(M_ia_jb_b)
        M_aabb = radc_ao2mo.write_dataset(M_aabb)

    M_ia_jb = (M_ia_jb_a, M_ia_jb_b, M_aabb)
    cput0 = log.timer_debug1("Completed M_ia_jb  ADC calculation", *cput0)

    return M_ia_jb


#@profile
def get_diag(adc,M_ia_jb=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nocc_a*nvir_a
    n_singles_b = nocc_b*nvir_b
    n_doubles_aaaa = nocc_a * (nocc_a - 1) * nvir_a * (nvir_a -1) // 4
    n_doubles_abab = nocc_a * nocc_b * nvir_a * nvir_b
    n_doubles_bbbb = nocc_b * (nocc_b - 1) * nvir_b * (nvir_b -1) // 4

    dim = n_singles_a + n_singles_b + n_doubles_aaaa + n_doubles_abab +  n_doubles_bbbb
    
    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    diag = np.zeros(dim)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaaa = f_b
    f_aaaa = s_aaaa + n_doubles_aaaa
    s_abab = f_aaaa
    f_abab = s_abab + n_doubles_abab
    s_bbbb = f_abab
    f_bbbb = s_bbbb + n_doubles_bbbb

    if eris is None:
        eris = adc.transform_integrals()

    d_ij_a = e_occ_a[:,None]+e_occ_a
    d_ij_b = e_occ_b[:,None]+e_occ_b
    d_ab_a = e_vir_a[:,None]+e_vir_a
    d_ab_b = e_vir_b[:,None]+e_vir_b
    
    D_ijab_a = (-d_ij_a.reshape(-1,1) + d_ab_a.reshape(-1)).reshape((nocc_a,nocc_a,nvir_a,nvir_a))[:,:,ab_ind_a[0],ab_ind_a[1]]
    diag[s_aaaa:f_aaaa] = D_ijab_a[ij_ind_a[0],ij_ind_a[1]].reshape(-1)
    del D_ijab_a

    D_ijab_b = (-d_ij_b.reshape(-1,1) + d_ab_b.reshape(-1)).reshape((nocc_b,nocc_b,nvir_b,nvir_b))[:,:,ab_ind_b[0],ab_ind_b[1]]
    diag[s_bbbb:f_bbbb] = D_ijab_b[ij_ind_b[0],ij_ind_b[1]].reshape(-1)
    del D_ijab_b

    d_ij_abab = e_occ_a[:,None]+e_occ_b
    d_ab_abab = e_vir_a[:,None]+e_vir_b
    diag[s_abab:f_abab] = (-d_ij_abab.reshape(-1,1) + d_ab_abab.reshape(-1)).reshape(-1)


    if M_ia_jb is None:
        M_ia_jb  = adc.get_imds()

    M_ia_jb_a, M_ia_jb_b, M_aabb = M_ia_jb[0], M_ia_jb[1], M_ia_jb[2] 


    # Compute precond
    diag[s_a:f_a] = np.diagonal(M_ia_jb_a)
    diag[s_b:f_b] = np.diagonal(M_ia_jb_b)

    # Compute precond


#    print("diag", np.linalg.norm(diag))
#    exit()
  

    if not isinstance(eris.oovv, np.ndarray):
        diag = radc_ao2mo.write_dataset(diag)

    return diag

#@profile
def matvec(adc, M_ia_jb=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    
    if M_ia_jb is None:
        M_ia_jb  = adc.get_imds()


    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    n_singles_a = nocc_a * nvir_a
    n_singles_b = nocc_b * nvir_b
    n_doubles_aaaa = nocc_a * (nocc_a - 1) * nvir_a * (nvir_a -1) // 4
    n_doubles_ab = nocc_a * nocc_b * nvir_a * nvir_b
    n_doubles_bbbb = nocc_b * (nocc_b - 1) * nvir_b * (nvir_b -1) // 4

    dim = n_singles_a + n_singles_b + n_doubles_aaaa + n_doubles_ab + n_doubles_bbbb

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaaa = f_b
    f_aaaa = s_aaaa + n_doubles_aaaa
    s_abab = f_aaaa
    f_ab = s_abab + n_doubles_ab
    s_bbbb = f_ab
    f_bbbb = s_bbbb + n_doubles_bbbb
    
    d_ij_a = e_occ_a[:,None]+e_occ_a
    d_ij_b = e_occ_b[:,None]+e_occ_b
    d_ab_a = e_vir_a[:,None]+e_vir_a
    d_ab_b = e_vir_b[:,None]+e_vir_b


    d_ij_abab = e_occ_a[:,None]+e_occ_b
    d_ab_abab = e_vir_a[:,None]+e_vir_b

    #Calculate sigma vector
    #@profile
    def sigma_(r):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        r1_a = r[s_a:f_a]
        r1_b = r[s_b:f_b]

        r1_a_ov = r1_a.reshape(nocc_a, nvir_a)
        r1_b_ov = r1_b.reshape(nocc_b, nvir_b)
        
        
        r1_ab = r[s_abab:f_ab]

        r_vv_u_a = np.zeros((int((nocc_a * (nocc_a - 1))/2),nvir_a, nvir_a))
        r_vv_u_a[:,ab_ind_a[0],ab_ind_a[1]]= r[s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2))
        r_vv_u_a[:,ab_ind_a[1],ab_ind_a[0]]= -r[s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2))
        r2_a = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        r2_a[ij_ind_a[0],ij_ind_a[1],:,:]= r_vv_u_a
        r2_a[ij_ind_a[1],ij_ind_a[0],:,:]= -r_vv_u_a
        
        del r_vv_u_a

        r_vv_u_b = np.zeros((int((nocc_b * (nocc_b - 1))/2),nvir_b, nvir_b))
        r_vv_u_b[:,ab_ind_b[0],ab_ind_b[1]]= r[s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        r_vv_u_b[:,ab_ind_b[1],ab_ind_b[0]]= -r[s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        r2_b = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        r2_b[ij_ind_b[0],ij_ind_b[1],:,:]= r_vv_u_b
        r2_b[ij_ind_b[1],ij_ind_b[0],:,:]= -r_vv_u_b

        del r_vv_u_b

        s = np.zeros(dim)


############## ADC(2) 1 block ############################
#
        

        s[s_a:f_a] = lib.einsum('ab,b->a',M_ia_jb[0],r1_a, optimize = True)
       
        s[s_b:f_b] = lib.einsum('ab,b->a',M_ia_jb[1],r1_b, optimize = True)

        s[s_a:f_a] += lib.einsum('ab,b->a',M_ia_jb[2],r1_b, optimize = True)
        s[s_b:f_b] += lib.einsum('ba,b->a',M_ia_jb[2],r1_a, optimize = True)

        D_ijab_a = (-d_ij_a.reshape(-1,1) + d_ab_a.reshape(-1)).reshape((nocc_a,nocc_a,nvir_a,nvir_a))[:,:,ab_ind_a[0],ab_ind_a[1]]
        s[s_aaaa:f_aaaa] = (D_ijab_a[ij_ind_a[0],ij_ind_a[1]].reshape(-1))*r[s_aaaa:f_aaaa]
        del D_ijab_a
    
        D_ijab_b = (-d_ij_b.reshape(-1,1) + d_ab_b.reshape(-1)).reshape((nocc_b,nocc_b,nvir_b,nvir_b))[:,:,ab_ind_b[0],ab_ind_b[1]]
        s[s_bbbb:f_bbbb] = (D_ijab_b[ij_ind_b[0],ij_ind_b[1]].reshape(-1))*r[s_bbbb:f_bbbb]
        del D_ijab_b

        s[s_abab:f_ab] = ((-d_ij_abab.reshape(-1,1) + d_ab_abab.reshape(-1)).reshape(-1))*r1_ab

        r1_ab = r1_ab.reshape(nocc_a, nocc_b, nvir_a, nvir_b)
        # M^(1)_h0_h1
        temp_a = np.zeros((nocc_a, nocc_a, nvir_a, nvir_a))

        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovvv = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                k = eris_ovvv.shape[0]
                s[s_a:f_a] += 0.5*lib.einsum('imef,mfea->ia',r2_a[:,a:a+k], eris_ovvv, optimize = True).reshape(-1)
                s[s_a:f_a] -= 0.5*lib.einsum('imef,mefa->ia',r2_a[:,a:a+k], eris_ovvv, optimize = True).reshape(-1)
                temp_a[:,a:a+k] = -lib.einsum('ie,jabe->ijab',r1_a_ov, eris_ovvv, optimize = True)
                temp_a[:,a:a+k] += lib.einsum('ie,jbae->ijab',r1_a_ov, eris_ovvv, optimize = True)
                temp_a[a:a+k] += lib.einsum('je,iabe->ijab',r1_a_ov, eris_ovvv, optimize = True)
                temp_a[a:a+k] -= lib.einsum('je,ibae->ijab',r1_a_ov, eris_ovvv, optimize = True)
                del eris_ovvv
                a += k
            temp_a = temp_a[:,:,ab_ind_a[0],ab_ind_a[1]]
            s[s_aaaa:f_aaaa] += temp_a[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)
            del temp_a
        else:
            eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            s[s_a:f_a] += 0.5*lib.einsum('imef,mfea->ia',r2_a, eris_ovvv, optimize = True).reshape(-1)
            s[s_a:f_a] -= 0.5*lib.einsum('imef,mefa->ia',r2_a, eris_ovvv, optimize = True).reshape(-1)
            temp_a = -lib.einsum('ie,jabe->ijab',r1_a_ov, eris_ovvv, optimize = True)
            temp_a += lib.einsum('ie,jbae->ijab',r1_a_ov, eris_ovvv, optimize = True)
            temp_a += lib.einsum('je,iabe->ijab',r1_a_ov, eris_ovvv, optimize = True)
            temp_a -= lib.einsum('je,ibae->ijab',r1_a_ov, eris_ovvv, optimize = True)
            temp_a = temp_a[:,:,ab_ind_a[0],ab_ind_a[1]]
            s[s_aaaa:f_aaaa] += temp_a[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)
            del temp_a
            del eris_ovvv

        temp_abab = np.zeros((nocc_a, nocc_b, nvir_a, nvir_b))
        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVvv = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                k = eris_OVvv.shape[0]
                s[s_a:f_a] += lib.einsum('imef,mfea->ia',r1_ab[:,a:a+k], eris_OVvv, optimize = True).reshape(-1)
                temp_abab[:,a:a+k] = lib.einsum('ie,jbae->ijab',r1_a_ov, eris_OVvv, optimize = True)
                del eris_OVvv
                a += k
            s[s_abab:f_ab] += temp_abab.reshape(-1)
            del temp_abab
        else:
            eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            s[s_a:f_a] += lib.einsum('imef,mfea->ia',r1_ab, eris_OVvv, optimize = True).reshape(-1)
            temp_abab = lib.einsum('ie,jbae->ijab',r1_a_ov, eris_OVvv, optimize = True)
            s[s_abab:f_ab] += temp_abab.reshape(-1)
            del temp_abab
            del eris_OVvv
#
#
        temp_b = np.zeros((nocc_b, nocc_b, nvir_b, nvir_b))
        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVVV = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                k = eris_OVVV.shape[0]
                s[s_b:f_b] += 0.5*lib.einsum('imef,mfea->ia',r2_b[:,a:a+k], eris_OVVV, optimize = True).reshape(-1)
                s[s_b:f_b] -= 0.5*lib.einsum('imef,mefa->ia',r2_b[:,a:a+k], eris_OVVV, optimize = True).reshape(-1)
                temp_b[:,a:a+k] = -lib.einsum('ie,jabe->ijab',r1_b_ov, eris_OVVV, optimize = True)
                temp_b[:,a:a+k] += lib.einsum('ie,jbae->ijab',r1_b_ov, eris_OVVV, optimize = True)
                temp_b[a:a+k] += lib.einsum('je,iabe->ijab',r1_b_ov, eris_OVVV, optimize = True)
                temp_b[a:a+k] -= lib.einsum('je,ibae->ijab',r1_b_ov, eris_OVVV, optimize = True)
                del eris_OVVV
                a += k
            temp_b = temp_b[:,:,ab_ind_b[0],ab_ind_b[1]]
            s[s_bbbb:f_bbbb] += temp_b[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
            del temp_b
        else:
            eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            s[s_b:f_b] += 0.5*lib.einsum('imef,mfea->ia',r2_b, eris_OVVV, optimize = True).reshape(-1)
            s[s_b:f_b] -= 0.5*lib.einsum('imef,mefa->ia',r2_b, eris_OVVV, optimize = True).reshape(-1)
            temp_b = -lib.einsum('ie,jabe->ijab',r1_b_ov, eris_OVVV, optimize = True)
            temp_b += lib.einsum('ie,jbae->ijab',r1_b_ov, eris_OVVV, optimize = True)
            temp_b += lib.einsum('je,iabe->ijab',r1_b_ov, eris_OVVV, optimize = True)
            temp_b -= lib.einsum('je,ibae->ijab',r1_b_ov, eris_OVVV, optimize = True)
            temp_b = temp_b[:,:,ab_ind_b[0],ab_ind_b[1]]
            s[s_bbbb:f_bbbb] += temp_b[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
            del temp_b
            del eris_OVVV

#
        temp_abab = np.zeros((nocc_a, nocc_b, nvir_a, nvir_b))
        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovVV = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                k = eris_ovVV.shape[0]
                s[s_b:f_b] += lib.einsum('mife,mfea->ia',r1_ab[a:a+k], eris_ovVV, optimize = True).reshape(-1)
                temp_abab[a:a+k] = lib.einsum('je,iabe->ijab',r1_b_ov, eris_ovVV, optimize = True)
                del eris_ovVV
                a += k
            s[s_abab:f_ab] += temp_abab.reshape(-1)
            del temp_abab
        else:
            eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            s[s_b:f_b] += lib.einsum('mife,mfea->ia',r1_ab, eris_ovVV, optimize = True).reshape(-1)
            temp_abab = lib.einsum('je,iabe->ijab',r1_b_ov, eris_ovVV, optimize = True)
            s[s_abab:f_ab] += temp_abab.reshape(-1)
            del temp_abab
            del eris_ovVV

        s[s_a:f_a] -= 0.5*lib.einsum('mnae,neim->ia',r2_a, eris.ovoo, optimize = True).reshape(-1)
        s[s_a:f_a] += 0.5*lib.einsum('mnae,mein->ia',r2_a, eris.ovoo, optimize = True).reshape(-1)
        s[s_a:f_a] -= lib.einsum('mnae,neim->ia',r1_ab, eris.OVoo, optimize = True).reshape(-1)

        s[s_b:f_b] -= 0.5*lib.einsum('mnae,neim->ia',r2_b, eris.OVOO, optimize = True).reshape(-1)
        s[s_b:f_b] += 0.5*lib.einsum('mnae,mein->ia',r2_b, eris.OVOO, optimize = True).reshape(-1)
        s[s_b:f_b] -= lib.einsum('mnea,mein->ia',r1_ab, eris.ovOO, optimize = True).reshape(-1)

#        # # M^(1)_h1_h0

        temp_a = lib.einsum('ma,ibjm->ijab',r1_a_ov, eris.ovoo, optimize = True)
        temp_a -= lib.einsum('ma,jbim->ijab',r1_a_ov, eris.ovoo, optimize = True)
        temp_a -= lib.einsum('mb,iajm->ijab',r1_a_ov, eris.ovoo, optimize = True)
        temp_a += lib.einsum('mb,jaim->ijab',r1_a_ov, eris.ovoo, optimize = True)
        temp_a = temp_a[:,:,ab_ind_a[0],ab_ind_a[1]]
        s[s_aaaa:f_aaaa] += temp_a[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)
        del temp_a

        temp_b = lib.einsum('ma,ibjm->ijab',r1_b_ov, eris.OVOO, optimize = True)
        temp_b -= lib.einsum('ma,jbim->ijab',r1_b_ov, eris.OVOO, optimize = True)
        temp_b -= lib.einsum('mb,iajm->ijab',r1_b_ov, eris.OVOO, optimize = True)
        temp_b += lib.einsum('mb,jaim->ijab',r1_b_ov, eris.OVOO, optimize = True)
        temp_b = temp_b[:,:,ab_ind_b[0],ab_ind_b[1]]
        s[s_bbbb:f_bbbb] += temp_b[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
        del temp_b

        s[s_abab:f_ab] -= lib.einsum('ma,jbim->ijab',r1_a_ov, eris.OVoo, optimize = True).reshape(-1)
        s[s_abab:f_ab] -= lib.einsum('mb,iajm->ijab',r1_b_ov, eris.ovOO, optimize = True).reshape(-1)

        if (method == "adc(2)"):
            del r1_ab
            del r2_a
            del r2_b

        #exit()

#        print("norm of s after", np.linalg.norm(s))

        if (method == "adc(2)-x") or (method == "adc(3)"):
            
            if isinstance(eris.vvvv_p, np.ndarray):
                interim = np.ascontiguousarray(r2_a[:,:,ab_ind_a[0],ab_ind_a[1]]).reshape(nocc_a*nocc_a,-1)
                s[s_aaaa:f_aaaa] += np.dot(interim,eris.vvvv_p.T).reshape(nocc_a, nocc_a, -1)[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)
                del interim

            elif isinstance(eris.vvvv_p, list) :
                s[s_aaaa:f_aaaa] += uadc_amplitudes.contract_ladder_antisym(adc,r2_a,eris.vvvv_p)[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)

            else:
                s[s_aaaa:f_aaaa] += uadc_amplitudes.contract_ladder_antisym(adc,r2_a,eris.Lvv)[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)

            if isinstance(eris.vVvV_p, np.ndarray):
                s[s_abab:f_ab] += np.dot(r1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b),eris.vVvV_p.T).reshape(-1)
            elif isinstance(eris.vVvV_p, list):
                s[s_abab:f_ab] += uadc_amplitudes.contract_ladder(adc,r1_ab,eris.vVvV_p).reshape(-1)
            else:
                s[s_abab:f_ab] += uadc_amplitudes.contract_ladder(adc,r1_ab,(eris.Lvv,eris.LVV)).reshape(-1)

            if isinstance(eris.VVVV_p, np.ndarray):
                interim = np.ascontiguousarray(r2_b[:,:,ab_ind_b[0],ab_ind_b[1]]).reshape(nocc_b*nocc_b,-1)
                s[s_bbbb:f_bbbb] += np.dot(interim,eris.VVVV_p.T).reshape(nocc_b, nocc_b, -1)[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
                del interim

            elif isinstance(eris.VVVV_p, list) :
                s[s_bbbb:f_bbbb] += uadc_amplitudes.contract_ladder_antisym(adc,r2_b,eris.VVVV_p)[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
            else:
                s[s_bbbb:f_bbbb] += uadc_amplitudes.contract_ladder_antisym(adc,r2_b,eris.LVV)[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)

            interim_a = lib.einsum('imae,jbem->ijab', r2_a, eris.ovvo, optimize = True)
            interim_a -= lib.einsum('imae,mjbe->ijab', r2_a, eris.oovv, optimize = True)
            interim_a += lib.einsum('imae,jbem->ijab', r1_ab, eris.ovVO, optimize = True)

            interim_a -= lib.einsum('jmae,ibem->ijab', r2_a, eris.ovvo, optimize = True)
            interim_a += lib.einsum('jmae,mibe->ijab', r2_a, eris.oovv, optimize = True)
            interim_a -= lib.einsum('jmae,ibem->ijab', r1_ab, eris.ovVO, optimize = True)

            interim_a += 0.5*lib.einsum('mnab,minj->ijab', r2_a, eris.oooo, optimize = True)
            interim_a -= 0.5*lib.einsum('mnab,mjni->ijab', r2_a, eris.oooo, optimize = True)

            interim_a -= lib.einsum('imbe,jaem->ijab', r2_a, eris.ovvo, optimize = True)
            interim_a += lib.einsum('imbe,jmea->ijab', r2_a, eris.oovv, optimize = True)
            interim_a -= lib.einsum('imbe,jaem->ijab', r1_ab, eris.ovVO, optimize = True)

            interim_a += lib.einsum('jmbe,iaem->ijab', r2_a, eris.ovvo, optimize = True)
            interim_a -= lib.einsum('jmbe,imea->ijab', r2_a, eris.oovv, optimize = True)
            interim_a += lib.einsum('jmbe,iaem->ijab', r1_ab, eris.ovVO, optimize = True)

            interim_a = interim_a[:,:,ab_ind_a[0],ab_ind_a[1]]
            s[s_aaaa:f_aaaa] += interim_a[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)
            del interim_a

            interim_b = lib.einsum('imae,jbem->ijab', r2_b, eris.OVVO, optimize = True)
            interim_b -= lib.einsum('imae,mjbe->ijab', r2_b, eris.OOVV, optimize = True)
            interim_b += lib.einsum('miea,mebj->ijab', r1_ab, eris.ovVO, optimize = True)

            interim_b -= lib.einsum('jmae,ibem->ijab', r2_b, eris.OVVO, optimize = True)
            interim_b += lib.einsum('jmae,mibe->ijab', r2_b, eris.OOVV, optimize = True)
            interim_b -= lib.einsum('mjea,mebi->ijab', r1_ab, eris.ovVO, optimize = True)

            interim_b += 0.5*lib.einsum('mnab,minj->ijab', r2_b, eris.OOOO, optimize = True)
            interim_b -= 0.5*lib.einsum('mnab,mjni->ijab', r2_b, eris.OOOO, optimize = True)

            interim_b -= lib.einsum('imbe,jaem->ijab', r2_b, eris.OVVO, optimize = True)
            interim_b += lib.einsum('imbe,jmea->ijab', r2_b, eris.OOVV, optimize = True)
            interim_b -= lib.einsum('mieb,meaj->ijab', r1_ab, eris.ovVO, optimize = True)
            
            interim_b += lib.einsum('jmbe,iaem->ijab', r2_b, eris.OVVO, optimize = True)
            interim_b -= lib.einsum('jmbe,imea->ijab', r2_b, eris.OOVV, optimize = True)
            interim_b += lib.einsum('mjeb,meai->ijab', r1_ab, eris.ovVO, optimize = True)
            
            interim_b = interim_b[:,:,ab_ind_b[0],ab_ind_b[1]]
            s[s_bbbb:f_bbbb] += interim_b[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
            del interim_b
            
            s[s_abab:f_ab] += lib.einsum('imae,jbem->ijab', r1_ab, eris.OVVO, optimize = True).reshape(-1)
            s[s_abab:f_ab] -= lib.einsum('imae,mjbe->ijab', r1_ab, eris.OOVV, optimize = True).reshape(-1)
            s[s_abab:f_ab] += lib.einsum('imae,mebj->ijab', r2_a, eris.ovVO, optimize = True).reshape(-1)

            s[s_abab:f_ab] -= lib.einsum('mjae,mibe->ijab', r1_ab, eris.ooVV, optimize = True).reshape(-1)
            s[s_abab:f_ab] += lib.einsum('mnab,minj->ijab', r1_ab, eris.ooOO, optimize = True).reshape(-1)
            s[s_abab:f_ab] -= lib.einsum('imeb,jmea->ijab', r1_ab, eris.OOvv, optimize = True).reshape(-1)
            s[s_abab:f_ab] += lib.einsum('mjeb,iaem->ijab', r1_ab, eris.ovvo, optimize = True).reshape(-1)
            s[s_abab:f_ab] -= lib.einsum('mjeb,imea->ijab', r1_ab, eris.oovv, optimize = True).reshape(-1)
            s[s_abab:f_ab] += lib.einsum('jmbe,iaem->ijab', r2_b, eris.ovVO, optimize = True).reshape(-1)

        if (method == "adc(3)"):
            #M_Y120_aa
            t1 = adc.t1
            t2 = adc.t2

            Y_aa = r1_a_ov.copy()
            Y_bb = r1_b_ov.copy()
            
            Y_aaaa = r2_a.copy()
            Y_bbbb = r2_b.copy()

            Y_abab = r1_ab.copy()


            if adc.f_ov is None:
                h_ce_aa = np.zeros((nocc_a, nvir_a))
                h_ce_bb = np.zeros((nocc_b, nvir_b))
                t1_ce_aa = np.zeros((nocc_a, nvir_a))
                t1_ce_bb = np.zeros((nocc_b, nvir_b))
            else:
                h_ce_aa, h_ce_bb = adc.f_ov
                t1_ce_aa = t1[2][0][:]
                t1_ce_bb = t1[2][1][:]

            t2_ce_aa = t1[0][0][:]
            t2_ce_bb = t1[0][1][:]

            t1_ccee_aaaa = t2[0][0][:].copy()
            t1_ccee_abab = t2[0][1][:].copy()
            t1_ccee_bbbb = t2[0][2][:].copy()

            t2_ccee_aaaa = t2[1][0][:].copy()
            t2_ccee_abab = t2[1][1][:].copy()
            t2_ccee_bbbb = t2[1][2][:].copy()

            einsum_type = True

            v_cccc_aaaa = eris.oooo
            v_cccc_bbbb = eris.OOOO
            v_cccc_aabb = eris.ooOO
            
            v_ceec_aaaa = eris.ovvo
            v_ceec_bbbb = eris.OVVO
            v_ceec_aabb = eris.ovVO
            v_ceec_bbaa = eris.OVvo
            
            v_ccee_aaaa = eris.oovv
            v_ccee_bbbb = eris.OOVV
            v_ccee_aabb = eris.ooVV
            v_ccee_bbaa = eris.OOvv

            v_cecc_aaaa = eris.ovoo
            v_cecc_bbbb = eris.OVOO
            v_cecc_aabb = eris.ovOO
            v_cecc_bbaa = eris.OVoo

            eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)

            v_ceee_aaaa = eris_ovvv.copy()
            v_ceee_bbbb = eris_OVVV.copy()
            v_ceee_aabb = eris_ovVV.copy()
            v_ceee_bbaa = eris_OVvv.copy()


            e_core_a = adc.mo_energy_a[:nocc_a].copy()
            e_extern_a = adc.mo_energy_a[nocc_a:].copy()
            
            e_core_b = adc.mo_energy_b[:nocc_b].copy()
            e_extern_b = adc.mo_energy_b[nocc_b:].copy()

            if isinstance(eris.vvvv_p, np.ndarray):
                v_eeee_aaaa = radc_ao2mo.unpack_eri_2(eris.vvvv_p, nvir_a)

                M_12Y0_aa = lib.einsum('Ia,Jb,CaDb->IJCD', Y_aa, t1_ce_aa, v_eeee_aaaa, optimize = einsum_type)
                M_12Y0_aa -= lib.einsum('Ia,Jb,CbDa->IJCD', Y_aa, t1_ce_aa, v_eeee_aaaa, optimize = einsum_type)
                M_12Y0_aa -= lib.einsum('Ja,Ib,CaDb->IJCD', Y_aa, t1_ce_aa, v_eeee_aaaa, optimize = einsum_type)
                M_12Y0_aa += lib.einsum('Ja,Ib,CbDa->IJCD', Y_aa, t1_ce_aa, v_eeee_aaaa, optimize = einsum_type)
                M_02Y1_aa = lib.einsum('Iiab,ic,Dacb->ID', Y_aaaa, t1_ce_aa, v_eeee_aaaa, optimize = einsum_type)
                del v_eeee_aaaa
    
                v_eeee_bbbb = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)

                M_12Y0_bb = lib.einsum('ia,jb,cadb->ijcd', Y_bb, t1_ce_bb, v_eeee_bbbb, optimize = einsum_type)
                M_12Y0_bb -= lib.einsum('ia,jb,cbda->ijcd', Y_bb, t1_ce_bb, v_eeee_bbbb, optimize = einsum_type)
                M_12Y0_bb -= lib.einsum('ja,ib,cadb->ijcd', Y_bb, t1_ce_bb, v_eeee_bbbb, optimize = einsum_type)
                M_12Y0_bb += lib.einsum('ja,ib,cbda->ijcd', Y_bb, t1_ce_bb, v_eeee_bbbb, optimize = einsum_type)
                M_02Y1_bb = lib.einsum('ijab,jc,dacb->id', Y_bbbb, t1_ce_bb, v_eeee_bbbb, optimize = einsum_type)
                del v_eeee_bbbb


            if isinstance(eris.vVvV_p,np.ndarray):

                v_eeee_abab = eris.vVvV_p
                v_eeee_abab = v_eeee_abab.reshape(nvir_a,nvir_b,nvir_a,nvir_b)

                M_12Y0_ab = lib.einsum('Ia,jb,Cdab->IjCd', Y_aa, t1_ce_bb, v_eeee_abab, optimize = einsum_type)
                M_12Y0_ab += lib.einsum('ja,Ib,Cdba->IjCd', Y_bb, t1_ce_aa, v_eeee_abab, optimize = einsum_type)
                
                M_02Y1_aa += lib.einsum('Iiab,ic,Dcab->ID', Y_abab, t1_ce_bb, v_eeee_abab, optimize = einsum_type)
                M_02Y1_bb += lib.einsum('jiab,jc,cdab->id', Y_abab, t1_ce_aa, v_eeee_abab, optimize = einsum_type)

                del v_eeee_abab

#            if isinstance(eris.vvvv_p, list):
#    
#                a = 0
#                temp_ = np.zeros((nvir_a,nvir_a))
#                for dataset in eris.vvvv_p:
#                    k = dataset.shape[0]
#                    vvvv = dataset[:]
#                    v_eeee_aaaa = np.zeros((k,nvir_a,nvir_a,nvir_a))
#                    v_eeee_aaaa[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv
#                    v_eeee_aaaa[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv
#
#                    M_12Y0_aa += einsum('Ia,Jb,CDab->IJCD', Y_aa, t1_ce_aa, v_eeee_aaaa, optimize = einsum_type)
#                    M_12Y0_aa -= einsum('Ja,Ib,CDab->IJCD', Y_aa, t1_ce_aa, v_eeee_aaaa, optimize = einsum_type)
#
#                    M_02Y1_aa += 1/2 * einsum('Iiab,ic,Dcab->ID', Y_aaaa, t1_ce_aa, v_eeee_aaaa, optimize = einsum_type)
#
#                    del v_eeee_aaaa
#                    a += k
#                M_ia_jb_a  += temp_
                 #M_12Y0_bb += einsum('ia,jb,cdab->ijcd', Y_bb, t1_ce_bb, v_eeee_bbbb, optimize = einsum_type)
                 #M_12Y0_bb -= einsum('ja,ib,cdab->ijcd', Y_bb, t1_ce_bb, v_eeee_bbbb, optimize = einsum_type)
                 #M_02Y1_bb += 1/2 * einsum('ijab,jc,dcab->id', Y_bbbb, t1_ce_bb, v_eeee_bbbb, optimize = einsum_type)
#

#            M_12Y0_ab = lib.einsum('Ia,jb,Cadb->IjCd', Y_aa, t1_ce_bb, v_eeee_aabb, optimize = einsum_type)
#            M_12Y0_ab += lib.einsum('ja,Ib,Cbda->IjCd', Y_bb, t1_ce_aa, v_eeee_aabb, optimize = einsum_type)
#            M_02Y1_aa += lib.einsum('Iiab,ic,Dacb->ID', Y_abab, t1_ce_bb, v_eeee_aabb, optimize = einsum_type)
#            M_02Y1_bb += lib.einsum('jiab,jc,dbca->id', Y_abab, t1_ce_aa, v_eeee_bbaa, optimize = einsum_type)

            M_12Y0_aa += lib.einsum('Ia,ia,JiCD->IJCD', Y_aa, h_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ja,ia,IiCD->IJCD', Y_aa, h_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iC,ia,IJDa->IJCD', Y_aa, h_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iD,ia,IJCa->IJCD', Y_aa, h_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ia,iC,JDai->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ia,iC,iJDa->IJCD', Y_aa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ia,iD,JCai->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ia,iD,iJCa->IJCD', Y_aa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ia,JiCb,ibDa->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ia,JiCb,iaDb->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ia,JiDb,ibCa->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ia,JiDb,iaCb->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ia,JiCb,ibDa->IJCD', Y_aa, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ia,JiDb,ibCa->IJCD', Y_aa, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ia,ia,JCDi->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ia,ia,iCDJ->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)

            M_12Y0_aa -= lib.einsum('Ia,ijCD,jaiJ->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)

            M_12Y0_aa += lib.einsum('Ja,iC,IDai->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ja,iC,iIDa->IJCD', Y_aa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ja,iD,ICai->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ja,iD,iICa->IJCD', Y_aa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ja,IiCb,ibDa->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ja,IiCb,iaDb->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ja,IiDb,ibCa->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ja,IiDb,iaCb->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ja,IiCb,ibDa->IJCD', Y_aa, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ja,IiDb,ibCa->IJCD', Y_aa, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ja,ia,ICDi->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ja,ia,iCDI->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            
            M_12Y0_aa += lib.einsum('Ja,ijCD,jaiI->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_12Y0_aa += lib.einsum('iC,jD,IiJj->IJCD', Y_aa, t1_ce_aa, v_cccc_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iC,jD,IjJi->IJCD', Y_aa, t1_ce_aa, v_cccc_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iC,IJab,ibDa->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            
            M_12Y0_aa += lib.einsum('iC,IjDa,jaiJ->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iC,IjDa,iajJ->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_12Y0_aa -= lib.einsum('iC,Ia,JDai->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iC,Ia,iJDa->IJCD', Y_aa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iC,IjDa,jaiJ->IJCD', Y_aa, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            
            M_12Y0_aa -= lib.einsum('iC,JjDa,jaiI->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iC,JjDa,iajI->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_12Y0_aa += lib.einsum('iC,Ja,IDai->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iC,Ja,iIDa->IJCD', Y_aa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iC,JjDa,jaiI->IJCD', Y_aa, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iC,ia,IDaJ->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iC,ia,JDaI->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iD,jC,IiJj->IJCD', Y_aa, t1_ce_aa, v_cccc_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iD,jC,IjJi->IJCD', Y_aa, t1_ce_aa, v_cccc_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iD,IJab,ibCa->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            
            M_12Y0_aa -= lib.einsum('iD,IjCa,jaiJ->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iD,IjCa,iajJ->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_12Y0_aa += lib.einsum('iD,Ia,JCai->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iD,Ia,iJCa->IJCD', Y_aa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iD,IjCa,jaiJ->IJCD', Y_aa, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            
            M_12Y0_aa += lib.einsum('iD,JjCa,jaiI->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iD,JjCa,iajI->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_12Y0_aa -= lib.einsum('iD,Ja,ICai->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iD,Ja,iICa->IJCD', Y_aa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iD,JjCa,jaiI->IJCD', Y_aa, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iD,ia,ICaJ->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iD,ia,JCaI->IJCD', Y_aa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('ia,IJCb,ibDa->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('ia,IJCb,iaDb->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('ia,IJDb,ibCa->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('ia,IJDb,iaCb->IJCD', Y_aa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            
            M_12Y0_aa += lib.einsum('ia,IjCD,jaiJ->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('ia,IjCD,iajJ->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('ia,JjCD,jaiI->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('ia,JjCD,iajI->IJCD', Y_aa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_12Y0_aa += 1/2 * lib.einsum('Ia,C,ia,JiCD->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += 1/2 * lib.einsum('Ia,D,ia,JiCD->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= 1/2 * lib.einsum('Ia,J,ia,JiCD->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += 1/2 * lib.einsum('Ia,a,ia,JiCD->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('Ia,i,ia,JiCD->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= 1/2 * lib.einsum('Ja,C,ia,IiCD->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= 1/2 * lib.einsum('Ja,D,ia,IiCD->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += 1/2 * lib.einsum('Ja,I,ia,IiCD->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= 1/2 * lib.einsum('Ja,a,ia,IiCD->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('Ja,i,ia,IiCD->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += 1/2 * lib.einsum('iC,D,ia,IJDa->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= 1/2 * lib.einsum('iC,I,ia,IJDa->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= 1/2 * lib.einsum('iC,J,ia,IJDa->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= 1/2 * lib.einsum('iC,i,ia,IJDa->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('iC,a,ia,IJDa->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= 1/2 * lib.einsum('iD,C,ia,IJCa->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += 1/2 * lib.einsum('iD,I,ia,IJCa->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += 1/2 * lib.einsum('iD,J,ia,IJCa->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa += 1/2 * lib.einsum('iD,i,ia,IJCa->IJCD', Y_aa, e_core_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('iD,a,ia,IJCa->IJCD', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('ia,ka,jkcd->ijcd', Y_bb, h_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ja,ka,ikcd->ijcd', Y_bb, h_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kc,ka,ijda->ijcd', Y_bb, h_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kd,ka,ijca->ijcd', Y_bb, h_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ia,kc,jdak->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ia,kc,kjda->ijcd', Y_bb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ia,kd,jcak->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ia,kd,kjca->ijcd', Y_bb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ia,jkcb,kbda->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ia,jkcb,kadb->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ia,jkdb,kbca->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ia,jkdb,kacb->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ia,ka,jcdk->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ia,ka,kcdj->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ia,kjbc,kbda->ijcd', Y_bb, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ia,kjbd,kbca->ijcd', Y_bb, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('ia,klcd,lakj->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('ja,kc,idak->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ja,kc,kida->ijcd', Y_bb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ja,kd,icak->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ja,kd,kica->ijcd', Y_bb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ja,ikcb,kbda->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ja,ikcb,kadb->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ja,ikdb,kbca->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ja,ikdb,kacb->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ja,ka,icdk->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ja,ka,kcdi->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ja,kibc,kbda->ijcd', Y_bb, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ja,kibd,kbca->ijcd', Y_bb, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('ja,klcd,laki->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('kc,ld,ikjl->ijcd', Y_bb, t1_ce_bb, v_cccc_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kc,ld,iljk->ijcd', Y_bb, t1_ce_bb, v_cccc_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kc,ijab,kbda->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('kc,ilda,lakj->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kc,ilda,kalj->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('kc,ia,jdak->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kc,ia,kjda->ijcd', Y_bb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('kc,jlda,laki->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kc,jlda,kali->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('kc,ja,idak->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kc,ja,kida->ijcd', Y_bb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kc,ka,idaj->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kc,ka,jdai->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('kc,liad,lakj->ijcd', Y_bb, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kc,ljad,laki->ijcd', Y_bb, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('kd,lc,ikjl->ijcd', Y_bb, t1_ce_bb, v_cccc_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kd,lc,iljk->ijcd', Y_bb, t1_ce_bb, v_cccc_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kd,ijab,kbca->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('kd,ilca,lakj->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kd,ilca,kalj->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('kd,ia,jcak->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kd,ia,kjca->ijcd', Y_bb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('kd,jlca,laki->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kd,jlca,kali->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('kd,ja,icak->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kd,ja,kica->ijcd', Y_bb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kd,ka,icaj->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kd,ka,jcai->ijcd', Y_bb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('kd,liac,lakj->ijcd', Y_bb, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kd,ljac,laki->ijcd', Y_bb, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('ka,ijcb,kbda->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ka,ijcb,kadb->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ka,ijdb,kbca->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ka,ijdb,kacb->ijcd', Y_bb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += lib.einsum('ka,ilcd,lakj->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ka,ilcd,kalj->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ka,jlcd,laki->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ka,jlcd,kali->ijcd', Y_bb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_bb += 1/2 * lib.einsum('ia,c,ka,jkcd->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += 1/2 * lib.einsum('ia,d,ka,jkcd->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= 1/2 * lib.einsum('ia,j,ka,jkcd->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += 1/2 * lib.einsum('ia,a,ka,jkcd->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ia,k,ka,jkcd->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= 1/2 * lib.einsum('ja,c,ka,ikcd->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= 1/2 * lib.einsum('ja,d,ka,ikcd->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += 1/2 * lib.einsum('ja,i,ka,ikcd->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= 1/2 * lib.einsum('ja,a,ka,ikcd->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ja,k,ka,ikcd->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += 1/2 * lib.einsum('kc,d,ka,ijda->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= 1/2 * lib.einsum('kc,i,ka,ijda->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= 1/2 * lib.einsum('kc,j,ka,ijda->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= 1/2 * lib.einsum('kc,k,ka,ijda->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('kc,a,ka,ijda->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= 1/2 * lib.einsum('kd,c,ka,ijca->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += 1/2 * lib.einsum('kd,i,ka,ijca->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += 1/2 * lib.einsum('kd,j,ka,ijca->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb += 1/2 * lib.einsum('kd,k,ka,ijca->ijcd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('kd,a,ka,ijca->ijcd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            
            
            M_12Y0_bb += lib.einsum('ka,ijcb,kadb->ijcd', Y_aa, t1_ccee_bbbb, v_ceee_aabb, optimize = einsum_type)
            M_12Y0_bb -= lib.einsum('ka,ijdb,kacb->ijcd', Y_aa, t1_ccee_bbbb, v_ceee_aabb, optimize = einsum_type)
            
            M_12Y0_bb -= lib.einsum('ka,ilcd,kalj->ijcd', Y_aa, t1_ccee_bbbb, v_cecc_aabb, optimize = einsum_type)
            M_12Y0_bb += lib.einsum('ka,jlcd,kali->ijcd', Y_aa, t1_ccee_bbbb, v_cecc_aabb, optimize = einsum_type)
            
            
            M_12Y0_aa += lib.einsum('ia,IJCb,iaDb->IJCD', Y_bb, t1_ccee_aaaa, v_ceee_bbaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('ia,IJDb,iaCb->IJCD', Y_bb, t1_ccee_aaaa, v_ceee_bbaa, optimize = einsum_type)
            M_12Y0_aa -= lib.einsum('ia,IjCD,iajJ->IJCD', Y_bb, t1_ccee_aaaa, v_cecc_bbaa, optimize = einsum_type)
            M_12Y0_aa += lib.einsum('ia,JjCD,iajI->IJCD', Y_bb, t1_ccee_aaaa, v_cecc_bbaa, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('Ia,ia,ijCd->IjCd', Y_aa, h_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('iC,ia,Ijad->IjCd', Y_aa, h_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('Ia,iC,jdai->IjCd', Y_aa, t1_ce_aa, v_ceec_bbaa, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('Ia,id,ijCa->IjCd', Y_aa, t1_ce_bb, v_ccee_bbaa, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('Ia,jidb,ibCa->IjCd', Y_aa, t1_ccee_bbbb, v_ceee_bbaa, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('Ia,ia,iCdj->IjCd', Y_aa, t1_ce_aa, v_ceec_aabb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('Ia,ijCb,iadb->IjCd', Y_aa, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('Ia,ijbd,ibCa->IjCd', Y_aa, t1_ccee_abab, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('Ia,ijbd,iaCb->IjCd', Y_aa, t1_ccee_abab, v_ceee_aaaa, optimize = einsum_type)
            
            M_12Y0_ab += lib.einsum('Ia,ikCd,iakj->IjCd', Y_aa, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('iC,Ijab,iadb->IjCd', Y_aa, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('iC,Ia,jdai->IjCd', Y_aa, t1_ce_aa, v_ceec_bbaa, optimize = einsum_type)
            
            M_12Y0_ab += lib.einsum('iC,Ikad,iakj->IjCd', Y_aa, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_12Y0_ab += lib.einsum('iC,kd,Iijk->IjCd', Y_aa, t1_ce_bb, v_cccc_aabb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('iC,jkda,kaiI->IjCd', Y_aa, t1_ccee_bbbb, v_cecc_bbaa, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('iC,ja,Iiad->IjCd', Y_aa, t1_ce_bb, v_ccee_aabb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('iC,ia,Iadj->IjCd', Y_aa, t1_ce_aa, v_ceec_aabb, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('iC,kjad,kaiI->IjCd', Y_aa, t1_ccee_abab, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('iC,kjad,iakI->IjCd', Y_aa, t1_ccee_abab, v_cecc_aaaa, optimize = einsum_type)
            
            M_12Y0_ab += lib.einsum('ia,IjCb,iadb->IjCd', Y_aa, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ia,Ijbd,ibCa->IjCd', Y_aa, t1_ccee_abab, v_ceee_aaaa, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('ia,Ijbd,iaCb->IjCd', Y_aa, t1_ccee_abab, v_ceee_aaaa, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('ia,IkCd,iakj->IjCd', Y_aa, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_12Y0_ab += lib.einsum('ia,kjCd,kaiI->IjCd', Y_aa, t1_ccee_abab, v_cecc_aaaa, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ia,kjCd,iakI->IjCd', Y_aa, t1_ccee_abab, v_cecc_aaaa, optimize = einsum_type)
            
            M_12Y0_ab -= 1/2 * lib.einsum('Ia,C,ia,ijCd->IjCd', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= 1/2 * lib.einsum('Ia,d,ia,ijCd->IjCd', Y_aa, e_extern_b, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += 1/2 * lib.einsum('Ia,j,ia,ijCd->IjCd', Y_aa, e_core_b, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= 1/2 * lib.einsum('Ia,a,ia,ijCd->IjCd', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('Ia,i,ia,ijCd->IjCd', Y_aa, e_core_a, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += 1/2 * lib.einsum('iC,I,ia,Ijad->IjCd', Y_aa, e_core_a, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= 1/2 * lib.einsum('iC,d,ia,Ijad->IjCd', Y_aa, e_extern_b, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += 1/2 * lib.einsum('iC,j,ia,Ijad->IjCd', Y_aa, e_core_b, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += 1/2 * lib.einsum('iC,i,ia,Ijad->IjCd', Y_aa, e_core_a, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('iC,a,ia,Ijad->IjCd', Y_aa, e_extern_a, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            
            
            M_12Y0_ab -= lib.einsum('ja,ia,IiCd->IjCd', Y_bb, h_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('id,ia,IjCa->IjCd', Y_bb, h_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ja,iC,Iiad->IjCd', Y_bb, t1_ce_aa, v_ccee_aabb, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('ja,IiCb,ibda->IjCd', Y_bb, t1_ccee_aaaa, v_ceee_aabb, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('ja,IiCb,ibda->IjCd', Y_bb, t1_ccee_abab, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ja,IiCb,iadb->IjCd', Y_bb, t1_ccee_abab, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ja,Iibd,iaCb->IjCd', Y_bb, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ja,id,ICai->IjCd', Y_bb, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ja,ia,ICdi->IjCd', Y_bb, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('ja,ikCd,kaiI->IjCd', Y_bb, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('id,kC,Ikji->IjCd', Y_bb, t1_ce_aa, v_cccc_aabb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('id,Ijab,ibCa->IjCd', Y_bb, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('id,IkCa,kaij->IjCd', Y_bb, t1_ccee_aaaa, v_cecc_aabb, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('id,Ia,ijCa->IjCd', Y_bb, t1_ce_aa, v_ccee_bbaa, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('id,IkCa,kaij->IjCd', Y_bb, t1_ccee_abab, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('id,IkCa,iakj->IjCd', Y_bb, t1_ccee_abab, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('id,ja,ICai->IjCd', Y_bb, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('id,ia,ICaj->IjCd', Y_bb, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('id,kjCa,iakI->IjCd', Y_bb, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ia,IjCb,ibda->IjCd', Y_bb, t1_ccee_abab, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('ia,IjCb,iadb->IjCd', Y_bb, t1_ccee_abab, v_ceee_bbbb, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('ia,Ijbd,iaCb->IjCd', Y_bb, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            
            M_12Y0_ab += lib.einsum('ia,IkCd,kaij->IjCd', Y_bb, t1_ccee_abab, v_cecc_bbbb, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('ia,IkCd,iakj->IjCd', Y_bb, t1_ccee_abab, v_cecc_bbbb, optimize = einsum_type)
            
            M_12Y0_ab -= lib.einsum('ia,kjCd,iakI->IjCd', Y_bb, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_12Y0_ab -= 1/2 * lib.einsum('ja,C,ia,IiCd->IjCd', Y_bb, e_extern_a, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += 1/2 * lib.einsum('ja,I,ia,IiCd->IjCd', Y_bb, e_core_a, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= 1/2 * lib.einsum('ja,d,ia,IiCd->IjCd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= 1/2 * lib.einsum('ja,a,ia,IiCd->IjCd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += lib.einsum('ja,i,ia,IiCd->IjCd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= 1/2 * lib.einsum('id,C,ia,IjCa->IjCd', Y_bb, e_extern_a, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += 1/2 * lib.einsum('id,I,ia,IjCa->IjCd', Y_bb, e_core_a, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += 1/2 * lib.einsum('id,j,ia,IjCa->IjCd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab += 1/2 * lib.einsum('id,i,ia,IjCa->IjCd', Y_bb, e_core_b, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_12Y0_ab -= lib.einsum('id,a,ia,IjCa->IjCd', Y_bb, e_extern_b, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            
            M_02Y1_aa -= lib.einsum('IiDa,i,ia->ID', Y_aaaa, e_core_a, t2_ce_aa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,a,ia->ID', Y_aaaa, e_extern_a, t2_ce_aa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,ijab->ID', Y_aaaa, h_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,ijab->ID', Y_aaaa, h_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('Iiab,jD,ijab->ID', Y_aaaa, h_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('ijDa,Ib,ijab->ID', Y_aaaa, h_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,ijbc,jcab->ID', Y_aaaa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,ijbc,jcab->ID', Y_aaaa, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('IiDa,jkab,kbji->ID', Y_aaaa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('IiDa,jb,iabj->ID', Y_aaaa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,jb,ijba->ID', Y_aaaa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,jkab,kbji->ID', Y_aaaa, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,iabj->ID', Y_aaaa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,jb,ibaj->ID', Y_aaaa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,iabj->ID', Y_aaaa, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,iabj->ID', Y_aaaa, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('Iiab,jD,iabj->ID', Y_aaaa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,ijac,jcbD->ID', Y_aaaa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('Iiab,ijac,jDbc->ID', Y_aaaa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,ijac,jcbD->ID', Y_aaaa, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('Iiab,ja,ijDb->ID', Y_aaaa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,ja,jDbi->ID', Y_aaaa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            
            M_02Y1_aa += 1/2 * lib.einsum('Iiab,jkab,jDki->ID', Y_aaaa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('ijDa,Ib,iabj->ID', Y_aaaa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('ijDa,ijbc,Ibac->ID', Y_aaaa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('ijDa,ikab,kbIj->ID', Y_aaaa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('ijDa,ikab,Ibkj->ID', Y_aaaa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('ijDa,ib,Ijab->ID', Y_aaaa, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('ijDa,ib,Ibaj->ID', Y_aaaa, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('ijDa,ikab,kbIj->ID', Y_aaaa, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('ijDa,ka,Iikj->ID', Y_aaaa, t1_ce_aa, v_cccc_aaaa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('ijab,ijac,IDbc->ID', Y_aaaa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('ijab,ijac,IcbD->ID', Y_aaaa, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            
            M_02Y1_aa -= 1/2 * lib.einsum('ijab,ikab,IDkj->ID', Y_aaaa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('ijab,ikab,kDIj->ID', Y_aaaa, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_02Y1_aa -= 1/2 * lib.einsum('IiDa,i,ijab,jb->ID', Y_aaaa, e_core_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('IiDa,i,ijab,jb->ID', Y_aaaa, e_core_a, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('IiDa,a,ijab,jb->ID', Y_aaaa, e_extern_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('IiDa,a,ijab,jb->ID', Y_aaaa, e_extern_a, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,j,ijab,jb->ID', Y_aaaa, e_core_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,b,ijab,jb->ID', Y_aaaa, e_extern_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,j,ijab,jb->ID', Y_aaaa, e_core_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,b,ijab,jb->ID', Y_aaaa, e_extern_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_aa += 1/4 * lib.einsum('Iiab,D,ijab,jD->ID', Y_aaaa, e_extern_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= 1/4 * lib.einsum('Iiab,i,ijab,jD->ID', Y_aaaa, e_core_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('Iiab,b,ijab,jD->ID', Y_aaaa, e_extern_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('Iiab,j,ijab,jD->ID', Y_aaaa, e_core_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= 1/4 * lib.einsum('ijDa,I,ijab,Ib->ID', Y_aaaa, e_core_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('ijDa,j,ijab,Ib->ID', Y_aaaa, e_core_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/4 * lib.einsum('ijDa,a,ijab,Ib->ID', Y_aaaa, e_extern_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('ijDa,b,ijab,Ib->ID', Y_aaaa, e_extern_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
            
            M_02Y1_bb -= lib.einsum('ijda,j,ja->id', Y_bbbb, e_core_b, t2_ce_bb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,a,ja->id', Y_bbbb, e_extern_b, t2_ce_bb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,kb,kjba->id', Y_bbbb, h_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,kb,jkab->id', Y_bbbb, h_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('ijab,kd,jkab->id', Y_bbbb, h_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jkda,ib,jkab->id', Y_bbbb, h_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,jkbc,kcab->id', Y_bbbb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,kjbc,kbac->id', Y_bbbb, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,kb,jabk->id', Y_bbbb, t1_ce_aa, v_ceec_bbaa, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('ijda,klba,kblj->id', Y_bbbb, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_02Y1_bb += lib.einsum('ijda,kb,jabk->id', Y_bbbb, t1_ce_aa, v_ceec_bbaa, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('ijda,klab,lbkj->id', Y_bbbb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_02Y1_bb += lib.einsum('ijda,kb,jabk->id', Y_bbbb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('ijda,kb,jkba->id', Y_bbbb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,kb,jabk->id', Y_bbbb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('ijda,kb,jbak->id', Y_bbbb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijab,kd,jabk->id', Y_bbbb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('ijab,jkac,kcbd->id', Y_bbbb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijab,jkac,kdbc->id', Y_bbbb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijab,ka,jkdb->id', Y_bbbb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('ijab,ka,kdbj->id', Y_bbbb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('ijab,kjca,kcbd->id', Y_bbbb, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            
            M_02Y1_bb += 1/2 * lib.einsum('ijab,klab,kdlj->id', Y_bbbb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_02Y1_bb += lib.einsum('jkda,ib,jabk->id', Y_bbbb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jkda,jkbc,ibac->id', Y_bbbb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            
            M_02Y1_bb += lib.einsum('jkda,jlab,lbik->id', Y_bbbb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jkda,jlab,iblk->id', Y_bbbb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_02Y1_bb += lib.einsum('jkda,jb,ikab->id', Y_bbbb, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jkda,jb,ibak->id', Y_bbbb, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jkda,la,ijlk->id', Y_bbbb, t1_ce_bb, v_cccc_bbbb, optimize = einsum_type)
            
            M_02Y1_bb += lib.einsum('jkda,ljba,lbik->id', Y_bbbb, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_02Y1_bb += 1/2 * lib.einsum('jkab,jkac,idbc->id', Y_bbbb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jkab,jkac,icbd->id', Y_bbbb, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            
            M_02Y1_bb -= 1/2 * lib.einsum('jkab,jlab,idlk->id', Y_bbbb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jkab,jlab,ldik->id', Y_bbbb, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_02Y1_bb -= 1/2 * lib.einsum('ijda,j,jkab,kb->id', Y_bbbb, e_core_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('ijda,j,kjba,kb->id', Y_bbbb, e_core_b, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('ijda,a,jkab,kb->id', Y_bbbb, e_extern_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('ijda,a,kjba,kb->id', Y_bbbb, e_extern_b, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('ijda,k,kjba,kb->id', Y_bbbb, e_core_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,b,kjba,kb->id', Y_bbbb, e_extern_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('ijda,k,jkab,kb->id', Y_bbbb, e_core_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('ijda,b,jkab,kb->id', Y_bbbb, e_extern_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/4 * lib.einsum('ijab,d,jkab,kd->id', Y_bbbb, e_extern_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/4 * lib.einsum('ijab,j,jkab,kd->id', Y_bbbb, e_core_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('ijab,b,jkab,kd->id', Y_bbbb, e_extern_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('ijab,k,jkab,kd->id', Y_bbbb, e_core_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/4 * lib.einsum('jkda,i,jkab,ib->id', Y_bbbb, e_core_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jkda,k,jkab,ib->id', Y_bbbb, e_core_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/4 * lib.einsum('jkda,a,jkab,ib->id', Y_bbbb, e_extern_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jkda,b,jkab,ib->id', Y_bbbb, e_extern_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
            
            M_02Y1_bb += 1/2 * lib.einsum('jkab,jkac,idbc->id', Y_aaaa, t1_ccee_aaaa, v_ceee_bbaa, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jkab,jlab,idlk->id', Y_aaaa, t1_ccee_aaaa, v_cecc_bbaa, optimize = einsum_type)
            
            M_02Y1_aa += 1/2 * lib.einsum('ijab,ijac,IDbc->ID', Y_bbbb, t1_ccee_bbbb, v_ceee_aabb, optimize = einsum_type)
            
            M_02Y1_aa -= 1/2 * lib.einsum('ijab,ikab,IDkj->ID', Y_bbbb, t1_ccee_bbbb, v_cecc_aabb, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('IiDa,i,ia->ID', Y_abab, e_core_b, t2_ce_bb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,a,ia->ID', Y_abab, e_extern_b, t2_ce_bb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,jiba->ID', Y_abab, h_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,ijab->ID', Y_abab, h_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,jD,jiab->ID', Y_abab, h_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('ijDa,Ib,ijba->ID', Y_abab, h_ce_aa, t1_ccee_abab, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,ijbc,jcab->ID', Y_abab, t1_ccee_bbbb, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jibc,jbac->ID', Y_abab, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,iabj->ID', Y_abab, t1_ce_aa, v_ceec_bbaa, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('IiDa,jkba,jbki->ID', Y_abab, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('IiDa,jb,iabj->ID', Y_abab, t1_ce_aa, v_ceec_bbaa, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('IiDa,jkab,kbji->ID', Y_abab, t1_ccee_bbbb, v_cecc_bbbb, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('IiDa,jb,iabj->ID', Y_abab, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,jb,ijba->ID', Y_abab, t1_ce_bb, v_ccee_bbbb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,jb,iabj->ID', Y_abab, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,jb,ibaj->ID', Y_abab, t1_ce_bb, v_ceec_bbbb, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,jD,ibaj->ID', Y_abab, t1_ce_aa, v_ceec_bbaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('Iiab,ijbc,jcaD->ID', Y_abab, t1_ccee_bbbb, v_ceee_bbaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,ja,jDbi->ID', Y_abab, t1_ce_aa, v_ceec_aabb, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,jb,ijDa->ID', Y_abab, t1_ce_bb, v_ccee_bbaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,jiac,jDbc->ID', Y_abab, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('Iiab,jicb,jcaD->ID', Y_abab, t1_ccee_abab, v_ceee_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('Iiab,jicb,jDac->ID', Y_abab, t1_ccee_abab, v_ceee_aaaa, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('Iiab,jkab,jDki->ID', Y_abab, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('ijDa,Ib,ibaj->ID', Y_abab, t1_ce_aa, v_ceec_aabb, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('ijDa,ijbc,Ibac->ID', Y_abab, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('ijDa,ib,Ibaj->ID', Y_abab, t1_ce_aa, v_ceec_aabb, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('ijDa,ikba,Ibkj->ID', Y_abab, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('ijDa,jkab,kbIi->ID', Y_abab, t1_ccee_bbbb, v_cecc_bbaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('ijDa,jb,Iiab->ID', Y_abab, t1_ce_bb, v_ccee_aabb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('ijDa,ka,Iikj->ID', Y_abab, t1_ce_bb, v_cccc_aabb, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('ijDa,kjba,kbIi->ID', Y_abab, t1_ccee_abab, v_cecc_aaaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('ijDa,kjba,Ibki->ID', Y_abab, t1_ccee_abab, v_cecc_aaaa, optimize = einsum_type)
            
            M_02Y1_aa += lib.einsum('ijab,ijac,IDbc->ID', Y_abab, t1_ccee_abab, v_ceee_aabb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('ijab,ijcb,IDac->ID', Y_abab, t1_ccee_abab, v_ceee_aaaa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('ijab,ijcb,IcaD->ID', Y_abab, t1_ccee_abab, v_ceee_aaaa, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('ijab,ikab,IDkj->ID', Y_abab, t1_ccee_abab, v_cecc_aabb, optimize = einsum_type)
            
            M_02Y1_aa -= lib.einsum('ijab,kjab,IDki->ID', Y_abab, t1_ccee_abab, v_cecc_aaaa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('ijab,kjab,kDIi->ID', Y_abab, t1_ccee_abab, v_cecc_aaaa, optimize = einsum_type)
            
            M_02Y1_aa -= 1/2 * lib.einsum('IiDa,i,ijab,jb->ID', Y_abab, e_core_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('IiDa,i,jiba,jb->ID', Y_abab, e_core_b, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('IiDa,a,ijab,jb->ID', Y_abab, e_extern_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('IiDa,a,jiba,jb->ID', Y_abab, e_extern_b, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,j,jiba,jb->ID', Y_abab, e_core_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,b,jiba,jb->ID', Y_abab, e_extern_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('IiDa,j,ijab,jb->ID', Y_abab, e_core_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('IiDa,b,ijab,jb->ID', Y_abab, e_extern_b, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('Iiab,D,jiab,jD->ID', Y_abab, e_extern_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('Iiab,i,jiab,jD->ID', Y_abab, e_core_b, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('Iiab,a,jiab,jD->ID', Y_abab, e_extern_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('Iiab,b,jiab,jD->ID', Y_abab, e_extern_b, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += lib.einsum('Iiab,j,jiab,jD->ID', Y_abab, e_core_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('ijDa,I,ijba,Ib->ID', Y_abab, e_core_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('ijDa,i,ijba,Ib->ID', Y_abab, e_core_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa += 1/2 * lib.einsum('ijDa,j,ijba,Ib->ID', Y_abab, e_core_b, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= 1/2 * lib.einsum('ijDa,a,ijba,Ib->ID', Y_abab, e_extern_b, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            M_02Y1_aa -= lib.einsum('ijDa,b,ijba,Ib->ID', Y_abab, e_extern_a, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('jiad,j,ja->id', Y_abab, e_core_a, t2_ce_aa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,a,ja->id', Y_abab, e_extern_a, t2_ce_aa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,kb,jkab->id', Y_abab, h_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,kb,jkab->id', Y_abab, h_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiab,kd,jkab->id', Y_abab, h_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jkad,ib,jkab->id', Y_abab, h_ce_bb, t1_ccee_abab, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,jkbc,kcab->id', Y_abab, t1_ccee_aaaa, v_ceee_aaaa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,jkbc,kcab->id', Y_abab, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('jiad,klab,lbkj->id', Y_abab, t1_ccee_aaaa, v_cecc_aaaa, optimize = einsum_type)
            
            M_02Y1_bb += lib.einsum('jiad,kb,jabk->id', Y_abab, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiad,kb,jkba->id', Y_abab, t1_ce_aa, v_ccee_aaaa, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiad,klab,lbkj->id', Y_abab, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,kb,jabk->id', Y_abab, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiad,kb,jbak->id', Y_abab, t1_ce_aa, v_ceec_aaaa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,kb,jabk->id', Y_abab, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,kb,jabk->id', Y_abab, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiab,kd,jabk->id', Y_abab, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiab,jkac,kcbd->id', Y_abab, t1_ccee_aaaa, v_ceee_aabb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiab,jkac,kcbd->id', Y_abab, t1_ccee_abab, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiab,jkac,kdbc->id', Y_abab, t1_ccee_abab, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiab,jkcb,kdac->id', Y_abab, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiab,ka,jkdb->id', Y_abab, t1_ce_aa, v_ccee_aabb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiab,kb,kdaj->id', Y_abab, t1_ce_bb, v_ceec_bbaa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiab,klab,ldkj->id', Y_abab, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jkad,ib,jabk->id', Y_abab, t1_ce_bb, v_ceec_aabb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jkad,jkbc,icab->id', Y_abab, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('jkad,jlab,lbik->id', Y_abab, t1_ccee_aaaa, v_cecc_aabb, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('jkad,jb,ikab->id', Y_abab, t1_ce_aa, v_ccee_bbaa, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('jkad,jlab,lbik->id', Y_abab, t1_ccee_abab, v_cecc_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jkad,jlab,iblk->id', Y_abab, t1_ccee_abab, v_cecc_bbbb, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('jkad,kb,ibaj->id', Y_abab, t1_ce_bb, v_ceec_bbaa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jkad,la,ljik->id', Y_abab, t1_ce_aa, v_cccc_aabb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jkad,lkab,iblj->id', Y_abab, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jkab,jkac,idbc->id', Y_abab, t1_ccee_abab, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jkab,jkac,icbd->id', Y_abab, t1_ccee_abab, v_ceee_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jkab,jkcb,idac->id', Y_abab, t1_ccee_abab, v_ceee_bbaa, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('jkab,jlab,idlk->id', Y_abab, t1_ccee_abab, v_cecc_bbbb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jkab,jlab,ldik->id', Y_abab, t1_ccee_abab, v_cecc_bbbb, optimize = einsum_type)
            
            M_02Y1_bb -= lib.einsum('jkab,lkab,idlj->id', Y_abab, t1_ccee_abab, v_cecc_bbaa, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jiad,j,jkab,kb->id', Y_abab, e_core_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jiad,j,jkab,kb->id', Y_abab, e_core_a, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jiad,a,jkab,kb->id', Y_abab, e_extern_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jiad,a,jkab,kb->id', Y_abab, e_extern_a, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiad,k,jkab,kb->id', Y_abab, e_core_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,b,jkab,kb->id', Y_abab, e_extern_a, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jiad,k,jkab,kb->id', Y_abab, e_core_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiad,b,jkab,kb->id', Y_abab, e_extern_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jiab,d,jkab,kd->id', Y_abab, e_extern_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jiab,j,jkab,kd->id', Y_abab, e_core_a, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jiab,a,jkab,kd->id', Y_abab, e_extern_a, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jiab,b,jkab,kd->id', Y_abab, e_extern_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += lib.einsum('jiab,k,jkab,kd->id', Y_abab, e_core_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jkad,i,jkab,ib->id', Y_abab, e_core_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jkad,j,jkab,ib->id', Y_abab, e_core_a, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb += 1/2 * lib.einsum('jkad,k,jkab,ib->id', Y_abab, e_core_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= 1/2 * lib.einsum('jkad,a,jkab,ib->id', Y_abab, e_extern_a, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            M_02Y1_bb -= lib.einsum('jkad,b,jkab,ib->id', Y_abab, e_extern_b, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)

            s[s_a:f_a] += M_02Y1_aa.reshape(-1)
            s[s_b:f_b] += M_02Y1_bb.reshape(-1)

            s[s_abab:f_ab] += M_12Y0_ab.reshape(-1)

            M_12Y0_aa = M_12Y0_aa[:,:,ab_ind_a[0],ab_ind_a[1]]
            s[s_aaaa:f_aaaa] += M_12Y0_aa[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)
           
            M_12Y0_bb = M_12Y0_bb[:,:,ab_ind_b[0],ab_ind_b[1]]
            s[s_bbbb:f_bbbb] += M_12Y0_bb[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
            
            del r1_ab
            del r2_a
            del r2_b
            
        return s

    return sigma_

#@profile
def get_opdm(adc):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    dm_a = adc.dm_a.copy()
    dm_b = adc.dm_b.copy()

    t1 = adc.t1
    t2 = adc.t2
    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b


    vir_list_a = range(nvir_a)
    vir_list_b = range(nvir_b)
    occ_list_a = range(nocc_a)
    occ_list_b = range(nocc_b)


    if adc.f_ov is None:
        f_ov_a = np.zeros((nocc_a, nvir_a))
        f_ov_b = np.zeros((nocc_b, nvir_b))
        t1_ce_aa = np.zeros((nocc_a, nvir_a))
        t1_ce_bb = np.zeros((nocc_b, nvir_b))
    else:
        f_ov_a, f_ov_b = adc.f_ov
        t1_ce_aa = t1[2][0][:]
        t1_ce_bb = t1[2][1][:]

    t2_ce_aa = t1[0][0][:]
    t2_ce_bb = t1[0][1][:]

    t1_ccee_aaaa = t2[0][0][:]
    t1_ccee_abab = t2[0][1][:]
    t1_ccee_bbbb = t2[0][2][:]

    t2_ccee_aaaa = t2[1][0][:]
    t2_ccee_abab = t2[1][1][:]
    t2_ccee_bbbb = t2[1][2][:]

    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    temp_a = np.zeros((nmo_a,nmo_a))
    temp_b = np.zeros((nmo_b,nmo_b))

    temp_2a = np.zeros((nmo_a,nmo_a,nmo_a,nmo_a))
    temp_2b = np.zeros((nmo_b,nmo_b,nmo_b,nmo_b))

    temp_2 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))

    n_singles_a = nocc_a * nvir_a
    n_singles_b = nocc_b * nvir_b
    n_doubles_aaaa = nocc_a * (nocc_a - 1) * nvir_a * (nvir_a -1) // 4
    n_doubles_ab = nocc_a * nocc_b * nvir_a * nvir_b
    n_doubles_bbbb = nocc_b * (nocc_b - 1) * nvir_b * (nvir_b -1) // 4

    dim = n_singles_a + n_singles_b + n_doubles_aaaa + n_doubles_ab + n_doubles_bbbb

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaaa = f_b
    f_aaaa = s_aaaa + n_doubles_aaaa
    s_abab = f_aaaa
    f_ab = s_abab + n_doubles_ab
    s_bbbb = f_ab
    f_bbbb = s_bbbb + n_doubles_bbbb

    U = adc.U.T
    nroots = U.shape[0]

    opdm_a = np.array([])
    opdm_b = np.array([])


    for r in range(U.shape[0]):
        
        Y_aa = U[r][:f_a].reshape(nocc_a, nvir_a)
        Y_bb = U[r][f_a:f_b].reshape(nocc_b, nvir_b)


        Y_abab = U[r][s_abab:f_ab].reshape(nocc_a, nocc_b, nvir_a, nvir_b)

        Y_vv_u_a = np.zeros((int((nocc_a * (nocc_a - 1))/2),nvir_a, nvir_a))
        Y_vv_u_a[:,ab_ind_a[0],ab_ind_a[1]]= U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2)) 
        Y_vv_u_a[:,ab_ind_a[1],ab_ind_a[0]]= -U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2))
        Y_aaaa = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        Y_aaaa[ij_ind_a[0],ij_ind_a[1],:,:]= Y_vv_u_a
        Y_aaaa[ij_ind_a[1],ij_ind_a[0],:,:]= -Y_vv_u_a

        del Y_vv_u_a

        Y_vv_u_b = np.zeros((int((nocc_b * (nocc_b - 1))/2),nvir_b, nvir_b))
        Y_vv_u_b[:,ab_ind_b[0],ab_ind_b[1]]= U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_vv_u_b[:,ab_ind_b[1],ab_ind_b[0]]= -U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_bbbb = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        Y_bbbb[ij_ind_b[0],ij_ind_b[1],:,:]= Y_vv_u_b
        Y_bbbb[ij_ind_b[1],ij_ind_b[0],:,:]= -Y_vv_u_b

        del Y_vv_u_b

# OPDM ADC(2)


        temp_a[:nocc_a,:nocc_a] =- 1/2 * lib.einsum('Ijab,Ljab->IL', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,La->IL', Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ijab,Ljab->IL', Y_abab, Y_abab, optimize = True)

        temp_a[occ_list_a,occ_list_a] += 1/4 * lib.einsum('jkab,jkab->', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('ja,ja->', Y_aa, Y_aa, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('jkab,jkab->', Y_abab, Y_abab, optimize = True)
        temp_a[occ_list_a,occ_list_a] += 1/4 * lib.einsum('jkab,jkab->', Y_bbbb, Y_bbbb, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('ja,ja->', Y_bb, Y_bb, optimize = True)

        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,Ljab,jb->IL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,Ljab,jb->IL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('La,Ijab,jb->IL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('La,Ijab,jb->IL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/4 * lib.einsum('Ia,ja,jkbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ia,ja,jb,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ia,ja,jkbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,kjcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/4 * lib.einsum('La,ja,jkbc,Ikbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('La,ja,jb,Ib->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('La,ja,jkbc,Ikbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,kjcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ib,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ia,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('ja,ka,Ijbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,kb,Ijac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,kb,Ijac,Lkcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ib,Lb->IL', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikca,Lkcb->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,kb,Ijca,Lkbc->IL', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,ka,Ijbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,kb,Ijca,Lkcb->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        
        temp_a[nocc_a:,nocc_a:]  = lib.einsum('iA,iC->AC', Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ijAb,ijCb->AC', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ijAb,ijCb->AC', Y_abab, Y_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iA,ijCb,jb->AC', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iA,ijCb,jb->AC', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iC,ijAb,jb->AC', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iC,ijAb,jb->AC', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iA,ib,jb,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/4 * lib.einsum('iA,ib,jkbd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iA,ib,jkbd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,kjdb->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iC,ib,jb,jA->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/4 * lib.einsum('iC,ib,jkbd,jkAd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iC,ib,jkbd,jkAd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,kjdb->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jA,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ib,ib,jkAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jkAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('ib,id,jkAb,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,iA,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,ikAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,ikAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,jd,ikAb,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jd,ikAb,kjCd->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jd,ikCb,kjAd->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jA,jC->AC', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ib,ib,jkAd,jkCd->AC', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jkAd,jkCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,id,jkAb,jkCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,kiAd,kjCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,jd,kiAb,kjCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        temp_IC  = lib.einsum('IjCa,ja->IC', Y_aaaa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,ja->IC', Y_abab, Y_bb, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t1_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_IC -= lib.einsum('jC,ja,Ia->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_aaaa, Y_abab, Y_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('Ijab,jkab,kC->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ia,ja,jC->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_abab, Y_bbbb, Y_bb, optimize = True)
        temp_IC += lib.einsum('IjCa,kjba,kb->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ijab,kjab,kC->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('jkCa,jkab,Ib->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('jkCa,jkba,Ib->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t2_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_IC -= lib.einsum('jC,ja,Ia->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ia,ja,jC->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,ja,jkCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,ja,jkCb,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('Ia,jb,ka,jkCb->IC', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,jb,ka,kjCb->IC', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ja,Ikab,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ja,Ikab,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('jC,ka,jb,Ikab->IC', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ka,jb,Ikba->IC', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,jb,IkCa,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,ka,IjCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_bb, Y_bb, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,jb,IkCa,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,ka,IjCb,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        
        temp_b[:nocc_b,:nocc_b] =- 1/2 * lib.einsum('ijab,ljab->il', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,la->il', Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('jiab,jlab->il', Y_abab, Y_abab, optimize = True)


        temp_b[occ_list_b, occ_list_b] += 1/4 * lib.einsum('jkab,jkab->', Y_aaaa, Y_aaaa, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('ja,ja->', Y_aa, Y_aa, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('jkab,jkab->', Y_abab, Y_abab, optimize = True)
        temp_b[occ_list_b, occ_list_b] += 1/4 * lib.einsum('jkab,jkab->', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('ja,ja->', Y_bb, Y_bb, optimize = True)

        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,ljab,jb->il', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,jlba,jb->il', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('la,ijab,jb->il', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('la,jiba,jb->il', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,lkac,jkbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,klca,jkbc->il', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/4 * lib.einsum('ia,ja,jkbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ia,ja,jb,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ia,ja,kjbc,klbc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,lkac,jkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,klca,kjcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,ikac,jkbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,kica,jkbc->il', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/4 * lib.einsum('la,ja,jkbc,ikbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('la,ja,jb,ib->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('la,ja,kjbc,kibc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,ikac,jkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,kica,kjcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ja,ja,ikbc,lkbc->il', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,ib,lb->il', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,kibc,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,kiac,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,ka,jibc,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,kb,jiac,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ja,ja,ikbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,ib,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,kibc,klbc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,ia,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,ikac,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,kica,klcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,kb,ijac,klbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,kb,ljac,kibc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ja,ka,ijbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,kb,ijac,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        temp_b[nocc_b:,nocc_b:]  = lib.einsum('ijba,ijbc->ac', Y_abab, Y_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,ic->ac', Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ijab,ijcb->ac', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,ijcb,jb->ac', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,jibc,jb->ac', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ic,ijab,jb->ac', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ic,jiba,jb->ac', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,ja,jc->ac', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,jkda,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ib,ib,jkad,jkcd->ac', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,id,jkba,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ikda,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,jd,ikba,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ia,ib,jb,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ia,ib,jkdb,jkdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/4 * lib.einsum('ia,ib,jkbd,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,ikcd,jkbd->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,kidc,jkbd->ac', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,ikcd,jkbd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,kidc,kjdb->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ic,ib,jb,ja->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ic,ib,jkdb,jkda->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/4 * lib.einsum('ic,ib,jkbd,jkad->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,ikad,jkbd->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,kida,jkbd->ac', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,ikad,jkbd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,kida,kjdb->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,ja,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,jkda,jkdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ib,ib,jkad,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ib,id,jkab,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jd,ikab,jkdc->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jd,ikcb,jkda->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ia,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ikad,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,kida,kjdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,jd,ikab,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        
        temp_ic  = lib.einsum('ijca,ja->ic', Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('jiac,ja->ic', Y_abab, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t1_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jc,ja,ia->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ijca,jkab,kb->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ijca,kjba,kb->ic', t1_ccee_bbbb, Y_abab, Y_aa, optimize = True)
        temp_ic += 1/2 * lib.einsum('ijab,jkab,kc->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('ia,ja,jc->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic += lib.einsum('jiac,jkab,kb->ic', t1_ccee_abab, Y_aaaa, Y_aa, optimize = True)
        temp_ic += lib.einsum('jiac,jkab,kb->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jiab,jkab,kc->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jkac,jkab,ib->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('jkca,jkab,ib->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t2_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jc,ja,ia->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('ia,ja,jc->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,jb,ka,jkbc->ic', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,ja,jkcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,ja,kjbc,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic += 1/2 * lib.einsum('ia,jb,ka,jkcb->ic', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,ikcb,kb->ic', Y_aa, Y_aa, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,kibc,kb->ic', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,jb,ka,kibc->ic', Y_aa, Y_aa, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,ka,jibc,kb->ic', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ja,ikab,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ja,kiba,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ka,jb,kiab->ic', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_ic += 1/2 * lib.einsum('jc,ka,jb,ikab->ic', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,ikcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,kibc,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,jb,ikca,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,ka,ijcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        
            
# OPDM ADC(2)-X
        if (method == "adc(2)-x"):

            temp_IC += 1/4 * lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC += lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_IC += 1/4 * lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_bbbb, Y_bbbb, optimize = True)
            temp_IC -= 1/2 * lib.einsum('jC,jkab,Ikab->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC -= lib.einsum('jC,jkab,Ikab->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_IC -= 1/2 * lib.einsum('Ia,jkab,jkCb->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC -= lib.einsum('Ia,jkab,jkCb->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            
            temp_ic += 1/4 * lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_aaaa, Y_aaaa, optimize = True)
            temp_ic += lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic += 1/4 * lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ic -= 1/2 * lib.einsum('jc,jkab,ikab->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ic -= lib.einsum('jc,kjab,kiab->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic -= lib.einsum('ia,jkba,jkbc->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic -= 1/2 * lib.einsum('ia,jkab,jkcb->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)

        temp_a[:nocc_a,nocc_a:] = temp_IC
        temp_a[nocc_a:,:nocc_a] = temp_IC.T

        temp_b[:nocc_b,nocc_b:] = temp_ic
        temp_b[nocc_b:,:nocc_b] = temp_ic.T
        
        opdm_a = np.append(opdm_a,temp_a)
        opdm_b = np.append(opdm_b,temp_b)

        #pdm = (opdm_a,opdm_b, None ,None ,None)

        norm = np.linalg.norm(temp_a - temp_a.transpose(1,0))
        print("total OPDM_a singles norm for Hermiticity",norm)
        print("opdm trace_a",np.einsum('pp',temp_a))
        
        
        norm = np.linalg.norm(temp_b - temp_b.transpose(1,0))
        print("total OPDM_b singles norm for Hermiticity",norm)
        print("opdm trace_b",np.einsum('pp',temp_b))
        
       # na = np.einsum('pp',temp_a)
       # nb = np.einsum('pp',temp_b)
       # spin = (temp_a,temp_b)
        
       # dip = lib.einsum("rqp,qp->r", dm_a, temp_a, optimize = True)
       # dip += lib.einsum("rqp,qp->r", dm_b, temp_b, optimize = True)
       # 
       # dipole_x = (adc.nucl_dip[0] + dip)*nist.AU2DEBYE
       # dipole_y = (adc.nucl_dip[1] + dip)*nist.AU2DEBYE
       # dipole_z = (adc.nucl_dip[2] + dip)*nist.AU2DEBYE
       # 
       # mag_dipole = np.sqrt((dipole_x[0])**2 + (dipole_y[1])**2 + (dipole_z[2])**2)
       # print("mag_dipole", mag_dipole)
       # print("opdm trace dipole na nb ",mag_dipole, np.einsum('pp',temp_a), np.einsum('pp',temp_b))

    opdm_a = opdm_a.reshape(nroots,nmo_a,nmo_a)
    opdm_b = opdm_b.reshape(nroots,nmo_b,nmo_b)

    opdm = (opdm_a ,opdm_b)

    return opdm



#@profile
def get_tpdm(adc):
    
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    dm_a = adc.dm_a.copy()
    dm_b = adc.dm_b.copy()

    t1 = adc.t1
    t2 = adc.t2
    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b


    vir_list_a = range(nvir_a)
    vir_list_b = range(nvir_b)
    occ_list_a = range(nocc_a)
    occ_list_b = range(nocc_b)


    if adc.f_ov is None:
        f_ov_a = np.zeros((nocc_a, nvir_a))
        f_ov_b = np.zeros((nocc_b, nvir_b))
        t1_ce_aa = np.zeros((nocc_a, nvir_a))
        t1_ce_bb = np.zeros((nocc_b, nvir_b))
    else:
        f_ov_a, f_ov_b = adc.f_ov
        t1_ce_aa = t1[2][0][:]
        t1_ce_bb = t1[2][1][:]

    t2_ce_aa = t1[0][0][:]
    t2_ce_bb = t1[0][1][:]

    t1_ccee_aaaa = t2[0][0][:]
    t1_ccee_abab = t2[0][1][:]
    t1_ccee_bbbb = t2[0][2][:]

    t2_ccee_aaaa = t2[1][0][:]
    t2_ccee_abab = t2[1][1][:]
    t2_ccee_bbbb = t2[1][2][:]

    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    temp_a = np.zeros((nmo_a,nmo_a))
    temp_b = np.zeros((nmo_b,nmo_b))

    temp_2a = np.zeros((nmo_a,nmo_a,nmo_a,nmo_a))
    temp_2b = np.zeros((nmo_b,nmo_b,nmo_b,nmo_b))

    temp_2 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))

    n_singles_a = nocc_a * nvir_a
    n_singles_b = nocc_b * nvir_b
    n_doubles_aaaa = nocc_a * (nocc_a - 1) * nvir_a * (nvir_a -1) // 4
    n_doubles_ab = nocc_a * nocc_b * nvir_a * nvir_b
    n_doubles_bbbb = nocc_b * (nocc_b - 1) * nvir_b * (nvir_b -1) // 4

    dim = n_singles_a + n_singles_b + n_doubles_aaaa + n_doubles_ab + n_doubles_bbbb

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaaa = f_b
    f_aaaa = s_aaaa + n_doubles_aaaa
    s_abab = f_aaaa
    f_ab = s_abab + n_doubles_ab
    s_bbbb = f_ab
    f_bbbb = s_bbbb + n_doubles_bbbb

    U = adc.U.T
    nroots = U.shape[0]


    tpdm_a = np.array([])
    tpdm_b = np.array([])
    mixed_spin = np.array([])

    block_1 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))

    for r in range(U.shape[0]):
        
        Y_aa = U[r][:f_a].reshape(nocc_a, nvir_a)
        Y_bb = U[r][f_a:f_b].reshape(nocc_b, nvir_b)


        Y_abab = U[r][s_abab:f_ab].reshape(nocc_a, nocc_b, nvir_a, nvir_b)

        Y_vv_u_a = np.zeros((int((nocc_a * (nocc_a - 1))/2),nvir_a, nvir_a))
        Y_vv_u_a[:,ab_ind_a[0],ab_ind_a[1]]= U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2)) 
        Y_vv_u_a[:,ab_ind_a[1],ab_ind_a[0]]= -U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2))
        Y_aaaa = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        Y_aaaa[ij_ind_a[0],ij_ind_a[1],:,:]= Y_vv_u_a
        Y_aaaa[ij_ind_a[1],ij_ind_a[0],:,:]= -Y_vv_u_a

        del Y_vv_u_a

        Y_vv_u_b = np.zeros((int((nocc_b * (nocc_b - 1))/2),nvir_b, nvir_b))
        Y_vv_u_b[:,ab_ind_b[0],ab_ind_b[1]]= U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_vv_u_b[:,ab_ind_b[1],ab_ind_b[0]]= -U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_bbbb = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        Y_bbbb[ij_ind_b[0],ij_ind_b[1],:,:]= Y_vv_u_b
        Y_bbbb[ij_ind_b[1],ij_ind_b[0],:,:]= -Y_vv_u_b

        del Y_vv_u_b

#TPDM ADC(2)

        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a]  = 1/2 * lib.einsum('IJab,KLab->IJKL', Y_aaaa, Y_aaaa, optimize = True)
        
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('Jmab,Lmab->JL', Y_aaaa, Y_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('Ja,La->JL', Y_aa, Y_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('Jmab,Lmab->JL', Y_abab, Y_abab, optimize = True)
        
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Jmab,Kmab->JK', Y_aaaa, Y_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('Ja,Ka->JK', Y_aa, Y_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('Jmab,Kmab->JK', Y_abab, Y_abab, optimize = True)
        
        block_1 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))
        block_1[occ_list_a,occ_list_a,:,:] = 1/2 * lib.einsum('Imab,Lmab->IL', Y_aaaa, Y_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('Ia,La->IL', Y_aa, Y_aa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('Imab,Lmab->IL', Y_abab, Y_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += block_1.transpose(2,0,1,3)

        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Imab,Kmab->IK', Y_aaaa, Y_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('Ia,Ka->IK', Y_aa, Y_aa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('Imab,Kmab->IK', Y_abab, Y_abab, optimize = True)

        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ia,KLab,Jb->IJKL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ja,KLab,Ib->IJKL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ka,IJab,Lb->IJKL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('La,IJab,Kb->IJKL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)

        diag_ = 1/4 * lib.einsum('mnab,mnab->', Y_aaaa, Y_aaaa, optimize = True)
        diag_ += lib.einsum('mnab,mnab->', Y_abab, Y_abab, optimize = True)
        diag_ += lib.einsum('ma,ma->', Y_aa, Y_aa, optimize = True)
        diag_ += 1/4 * lib.einsum('mnab,mnab->', Y_bbbb, Y_bbbb, optimize = True)
        diag_ += lib.einsum('ma,ma->', Y_bb, Y_bb, optimize = True)
        
        block_1a = np.zeros((nocc_a**2,nocc_a**2))
        np.fill_diagonal(block_1a, diag_)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += block_1a.reshape(nocc_a,nocc_a,nocc_a,nocc_a)

        off_diag_ = -1/4 * lib.einsum('mnab,mnab->', Y_aaaa, Y_aaaa, optimize = True)
        off_diag_ -= lib.einsum('ma,ma->', Y_aa, Y_aa, optimize = True)
        off_diag_ -= lib.einsum('mnab,mnab->', Y_abab, Y_abab, optimize = True)
        off_diag_ -= 1/4 * lib.einsum('mnab,mnab->', Y_bbbb, Y_bbbb, optimize = True)
        off_diag_ -= lib.einsum('ma,ma->', Y_bb, Y_bb, optimize = True)

        block_1a = np.zeros((nocc_a**2,nocc_a**2))
        np.fill_diagonal(block_1a, off_diag_)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += block_1a.reshape(nocc_a,nocc_a,nocc_a,nocc_a).transpose(0,1,3,2)


        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ia,Ka,Jmbc,Lmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ia,Ka,Jb,Lb->IJKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ia,Ka,Jmbc,Lmbc->IJKL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ia,Kb,La,Jb->IJKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ia,Kb,Lmac,Jmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ia,Kb,Lmac,Jmbc->IJKL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,La,Jmbc,Kmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ia,La,Jb,Kb->IJKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ia,La,Jmbc,Kmbc->IJKL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ia,Lb,Ka,Jb->IJKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ia,Lb,Kmac,Jmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ia,Lb,Kmac,Jmbc->IJKL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ia,ma,Jmbc,KLbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ia,mb,KLac,Jmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ia,mb,KLac,Jmcb->IJKL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ja,Ka,Imbc,Lmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ja,Ka,Ib,Lb->IJKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ja,Ka,Imbc,Lmbc->IJKL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ja,Kb,La,Ib->IJKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ja,Kb,Lmac,Imbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ja,Kb,Lmac,Imbc->IJKL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ja,La,Imbc,Kmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ja,La,Ib,Kb->IJKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ja,La,Imbc,Kmbc->IJKL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ja,Lb,Ka,Ib->IJKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ja,Lb,Kmac,Imbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ja,Lb,Kmac,Imbc->IJKL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ja,ma,Imbc,KLbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ja,mb,KLac,Imbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ja,mb,KLac,Imcb->IJKL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ka,ma,Lmbc,IJbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('Ka,mb,IJac,Lmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('Ka,mb,IJac,Lmcb->IJKL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,ma,Kmbc,IJbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += lib.einsum('La,mb,IJac,Kmbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('La,mb,IJac,Kmcb->IJKL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += 1/2 * lib.einsum('ma,ma,IJbc,KLbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] -= lib.einsum('ma,mb,IJac,KLbc->IJKL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += 1/2 * lib.einsum('ma,ma,IJbc,KLbc->IJKL', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)

        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('Ja,Lmab,mb->JL',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('Ja,Lmab,mb->JL',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('La,Jmab,mb->JL',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('La,Jmab,mb->JL',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('Ja,Kmab,mb->JK',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('Ja,Kmab,mb->JK',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('Ka,Jmab,mb->JK',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('Ka,Jmab,mb->JK',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        
        block_1 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))
        block_1[occ_list_a,occ_list_a,:,:] = lib.einsum('Ia,Lmab,mb->IL',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('Ia,Lmab,mb->IL',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('La,Imab,mb->IL',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('La,Imab,mb->IL',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += block_1.transpose(2,0,1,3)
        
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('Ia,Kmab,mb->IK',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('Ia,Kmab,mb->IK',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('Ka,Imab,mb->IK',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('Ka,Imab,mb->IK',  t1_ce_aa, Y_abab, Y_bb, optimize = True)

        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += 1/4 * lib.einsum('Ja,ma,mnbc,Lnbc->JL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += 1/2 * lib.einsum('Ja,ma,mb,Lb->JL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += 1/2 * lib.einsum('Ja,ma,mnbc,Lnbc->JL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('Ja,mb,Lnac,mnbc->JL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('Ja,mb,Lnac,mnbc->JL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('Ja,mb,Lnac,nmcb->JL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('Ja,mb,Lnac,mnbc->JL',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += 1/4 * lib.einsum('La,ma,mnbc,Jnbc->JL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += 1/2 * lib.einsum('La,ma,mb,Jb->JL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += 1/2 * lib.einsum('La,ma,mnbc,Jnbc->JL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('La,mb,Jnac,mnbc->JL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('La,mb,Jnac,mnbc->JL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('La,mb,Jnac,nmcb->JL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('La,mb,Jnac,mnbc->JL',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('ma,ma,Jnbc,Lnbc->JL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('ma,ma,Jb,Lb->JL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('ma,ma,Jnbc,Lnbc->JL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += lib.einsum('ma,mb,Ja,Lb->JL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += lib.einsum('ma,mb,Jnac,Lnbc->JL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += lib.einsum('ma,mb,Jnac,Lnbc->JL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += 1/2 * lib.einsum('ma,na,Jmbc,Lnbc->JL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('ma,nb,Jmac,Lnbc->JL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += lib.einsum('ma,nb,Jmac,Lncb->JL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('ma,ma,Jnbc,Lnbc->JL',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('ma,ma,Jb,Lb->JL',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('ma,ma,Jnbc,Lnbc->JL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += lib.einsum('ma,mb,Jnca,Lncb->JL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += lib.einsum('ma,nb,Jmca,Lnbc->JL',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] += lib.einsum('ma,na,Jmbc,Lnbc->JL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('ma,nb,Jmca,Lncb->JL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= 1/4 * lib.einsum('Ja,ma,mnbc,Knbc->JK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ja,ma,mb,Kb->JK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ja,ma,mnbc,Knbc->JK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ja,mb,Knac,mnbc->JK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ja,mb,Knac,mnbc->JK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ja,mb,Knac,nmcb->JK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ja,mb,Knac,mnbc->JK',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= 1/4 * lib.einsum('Ka,ma,mnbc,Jnbc->JK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ka,ma,mb,Jb->JK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ka,ma,mnbc,Jnbc->JK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ka,mb,Jnac,mnbc->JK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ka,mb,Jnac,mnbc->JK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ka,mb,Jnac,nmcb->JK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ka,mb,Jnac,mnbc->JK',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('ma,ma,Jnbc,Knbc->JK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('ma,ma,Jb,Kb->JK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('ma,ma,Jnbc,Knbc->JK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= lib.einsum('ma,mb,Ja,Kb->JK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= lib.einsum('ma,mb,Jnac,Knbc->JK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= lib.einsum('ma,mb,Jnac,Knbc->JK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('ma,na,Jmbc,Knbc->JK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('ma,nb,Jmac,Knbc->JK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= lib.einsum('ma,nb,Jmac,Kncb->JK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('ma,ma,Jnbc,Knbc->JK',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('ma,ma,Jb,Kb->JK',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('ma,ma,Jnbc,Knbc->JK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= lib.einsum('ma,mb,Jnca,Kncb->JK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= lib.einsum('ma,nb,Jmca,Knbc->JK',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] -= lib.einsum('ma,na,Jmbc,Knbc->JK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('ma,nb,Jmca,Kncb->JK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        block_1 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))
        block_1[occ_list_a,occ_list_a,:,:] = -1/4 * lib.einsum('Ia,ma,mnbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= 1/2 * lib.einsum('Ia,ma,mb,Lb->IL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= 1/2 * lib.einsum('Ia,ma,mnbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('Ia,mb,Lnac,mnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('Ia,mb,Lnac,mnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('Ia,mb,Lnac,nmcb->IL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('Ia,mb,Lnac,mnbc->IL',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= 1/4 * lib.einsum('La,ma,mnbc,Inbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= 1/2 * lib.einsum('La,ma,mb,Ib->IL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= 1/2 * lib.einsum('La,ma,mnbc,Inbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('La,mb,Inac,mnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('La,mb,Inac,mnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('La,mb,Inac,nmcb->IL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('La,mb,Inac,mnbc->IL',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('ma,ma,Inbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('ma,ma,Ib,Lb->IL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('ma,ma,Inbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= lib.einsum('ma,mb,Ia,Lb->IL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= lib.einsum('ma,mb,Inac,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= lib.einsum('ma,mb,Inac,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= 1/2 * lib.einsum('ma,na,Imbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('ma,nb,Imac,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= lib.einsum('ma,nb,Imac,Lncb->IL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += 1/2 * lib.einsum('ma,ma,Inbc,Lnbc->IL',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('ma,ma,Ib,Lb->IL',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('ma,ma,Inbc,Lnbc->IL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= lib.einsum('ma,mb,Inca,Lncb->IL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= lib.einsum('ma,nb,Imca,Lnbc->IL',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] -= lib.einsum('ma,na,Imbc,Lnbc->IL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_1[occ_list_a,occ_list_a,:,:] += lib.einsum('ma,nb,Imca,Lncb->IL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += block_1.transpose(2,0,1,3)
        
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += 1/4 * lib.einsum('Ia,ma,mnbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ia,ma,mb,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ia,ma,mnbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ia,mb,Knac,nmcb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += 1/4 * lib.einsum('Ka,ma,mnbc,Inbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ka,ma,mb,Ib->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Ka,ma,mnbc,Inbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ka,mb,Inac,nmcb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('ma,ma,Ib,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += lib.einsum('ma,mb,Ia,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += lib.einsum('ma,mb,Inac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += lib.einsum('ma,mb,Inac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('ma,na,Imbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('ma,nb,Imac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += lib.einsum('ma,nb,Imac,Kncb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= 1/2 * lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('ma,ma,Ib,Kb->IK',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += lib.einsum('ma,mb,Inca,Kncb->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += lib.einsum('ma,nb,Imca,Knbc->IK',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] += lib.einsum('ma,na,Imbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[:nocc_a,occ_list_a,:nocc_a,occ_list_a] -= lib.einsum('ma,nb,Imca,Kncb->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:]  = 1/2 * lib.einsum('ijAB,ijCD->ABCD', Y_aaaa, Y_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iA,ijCD,jB->ABCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iB,ijCD,jA->ABCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iC,ijAB,jD->ABCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iD,ijAB,jC->ABCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iA,iC,jB,jD->ABCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,iC,jkBe,jkDe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iA,iC,jkBe,jkDe->ABCD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iA,iD,jB,jC->ABCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iA,iD,jkBe,jkCe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iA,iD,jkBe,jkCe->ABCD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,ie,jkBe,jkCD->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iA,jC,iD,jB->ABCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iA,jC,ikDe,jkBe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iA,jC,ikDe,jkBe->ABCD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iA,jD,iC,jB->ABCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iA,jD,ikCe,jkBe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iA,jD,ikCe,jkBe->ABCD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iA,je,ikCD,jkBe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iA,je,ikCD,kjBe->ABCD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iB,iC,jA,jD->ABCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iB,iC,jkAe,jkDe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iB,iC,jkAe,jkDe->ABCD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iB,iD,jA,jC->ABCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += 1/2 * lib.einsum('iB,iD,jkAe,jkCe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iB,iD,jkAe,jkCe->ABCD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iB,ie,jkAe,jkCD->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iB,jC,iD,jA->ABCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iB,jC,ikDe,jkAe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iB,jC,ikDe,jkAe->ABCD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iB,jD,iC,jA->ABCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iB,jD,ikCe,jkAe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iB,jD,ikCe,jkAe->ABCD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iB,je,ikCD,jkAe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iB,je,ikCD,kjAe->ABCD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,ie,jkDe,jkAB->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iC,je,ikAB,jkDe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iC,je,ikAB,kjDe->ABCD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iD,ie,jkCe,jkAB->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += lib.einsum('iD,je,ikAB,jkCe->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('iD,je,ikAB,kjCe->ABCD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += 1/2 * lib.einsum('ie,ie,jkAB,jkCD->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] -= lib.einsum('ie,je,ikAB,jkCD->ABCD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_2a[nocc_a:,nocc_a:,nocc_a:,nocc_a:] += 1/2 * lib.einsum('ie,ie,jkAB,jkCD->ABCD', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        
        
        temp_IJCD  = lib.einsum('IC,JkDa,ka->IJCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IC,JkDa,ka->IJCD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_IJCD -= lib.einsum('JC,IkDa,ka->IJCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('JC,IkDa,ka->IJCD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_IJCD += lib.einsum('kC,ka,IJDa->IJCD', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
        temp_IJCD -= lib.einsum('ID,JkCa,ka->IJCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('ID,JkCa,ka->IJCD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_IJCD += lib.einsum('JD,IkCa,ka->IJCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('JD,IkCa,ka->IJCD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_IJCD -= lib.einsum('kD,ka,IJCa->IJCD', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
        temp_IJCD += lib.einsum('IJCD,ka,ka->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IJCD,ka,ka->IJCD', t1_ccee_aaaa, Y_bb, Y_bb, optimize = True)
        temp_IJCD -= lib.einsum('IJCa,ka,kD->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IJDa,ka,kC->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('IkCD,ka,Ja->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IkCa,ka,JD->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('IkDa,ka,JC->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('Ia,ka,JkCD->IJCD', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
        temp_IJCD += lib.einsum('IkCa,ka,JD->IJCD', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('IkDa,ka,JC->IJCD', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('JkCD,ka,Ia->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('JkCa,ka,ID->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('JkDa,ka,IC->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('Ja,ka,IkCD->IJCD', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
        temp_IJCD -= lib.einsum('JkCa,ka,ID->IJCD', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('JkDa,ka,IC->IJCD', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IJCD,ka,ka->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IJCD,ka,ka->IJCD', t2_ccee_aaaa, Y_bb, Y_bb, optimize = True)
        temp_IJCD -= lib.einsum('IJCa,ka,kD->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IJDa,ka,kC->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('IkCD,ka,Ja->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IkCa,ka,JD->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('IkDa,ka,JC->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('IkCa,ka,JD->IJCD', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('IkDa,ka,JC->IJCD', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('JkCD,ka,Ia->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('JkCa,ka,ID->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('JkDa,ka,IC->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('JkCa,ka,ID->IJCD', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
        temp_IJCD += lib.einsum('JkDa,ka,IC->IJCD', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
        temp_IJCD -= lib.einsum('IC,ka,kD,Ja->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD += lib.einsum('ID,ka,kC,Ja->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD -= lib.einsum('Ia,ka,kC,JD->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD += lib.einsum('Ia,ka,kD,JC->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD += lib.einsum('JC,ka,kD,Ia->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD -= lib.einsum('JD,ka,kC,Ia->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD += lib.einsum('Ja,ka,kC,ID->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD -= lib.einsum('Ja,ka,kD,IC->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD -= lib.einsum('kC,ka,Ia,JD->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD += lib.einsum('kC,ka,Ja,ID->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD += lib.einsum('kD,ka,Ia,JC->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD -= lib.einsum('kD,ka,Ja,IC->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD += lib.einsum('ka,ka,IC,JD->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD -= lib.einsum('ka,ka,JC,ID->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD += lib.einsum('ka,ka,IC,JD->IJCD', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_IJCD -= lib.einsum('ka,ka,JC,ID->IJCD', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        
        
        temp =- lib.einsum('ID,KA->IAKD', Y_aa, Y_aa, optimize = True)
        temp -= lib.einsum('IjDb,KjAb->IAKD', Y_aaaa, Y_aaaa, optimize = True)
        temp -= lib.einsum('IjDb,KjAb->IAKD', Y_abab, Y_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jA,jD->AD', Y_aa, Y_aa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jlAb,jlDb->AD', Y_aaaa, Y_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jlAb,jlDb->AD', Y_abab, Y_abab, optimize = True)
        temp -= lib.einsum('KA,IjDb,jb->IAKD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp -= lib.einsum('KA,IjDb,jb->IAKD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp += lib.einsum('jA,IjDb,Kb->IAKD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp -= lib.einsum('ID,KjAb,jb->IAKD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp -= lib.einsum('ID,KjAb,jb->IAKD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp += lib.einsum('jD,KjAb,Ib->IAKD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp += lib.einsum('Ib,KjAb,jD->IAKD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp += lib.einsum('Kb,IjDb,jA->IAKD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp += 1/2 * lib.einsum('ID,Kb,jb,jA->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp += 1/4 * lib.einsum('ID,Kb,jlbc,jlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += 1/2 * lib.einsum('ID,Kb,jlbc,jlAc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += 1/4 * lib.einsum('ID,jA,jlbc,Klbc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += 1/2 * lib.einsum('ID,jA,jb,Kb->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp += 1/2 * lib.einsum('ID,jA,jlbc,Klbc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('ID,jb,jlbc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= 1/2 * lib.einsum('ID,jb,jlbc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('ID,jb,jlbc,KlAc->IAKD', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('ID,jb,ljcb,KlAc->IAKD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp += 1/2 * lib.einsum('Ib,KA,jb,jD->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp += 1/4 * lib.einsum('Ib,KA,jlbc,jlDc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += 1/2 * lib.einsum('Ib,KA,jlbc,jlDc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('Ib,Kb,jA,jD->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp -= 1/2 * lib.einsum('Ib,Kb,jlAc,jlDc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('Ib,Kb,jlAc,jlDc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += 1/2 * lib.einsum('Ib,Kc,jlAb,jlDc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('Ib,jA,Kb,jD->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp -= lib.einsum('Ib,jA,Klbc,jlDc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('Ib,jA,Klbc,jlDc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('Ib,jb,jD,KA->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp += lib.einsum('Ib,jb,jlDc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('Ib,jb,jlDc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('Ib,jc,KlAb,jlDc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('Ib,jc,KlAb,ljDc->IAKD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp += 1/4 * lib.einsum('KA,jD,jlbc,Ilbc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += 1/2 * lib.einsum('KA,jD,jb,Ib->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp += 1/2 * lib.einsum('KA,jD,jlbc,Ilbc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('KA,jb,jlbc,IlDc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= 1/2 * lib.einsum('KA,jb,jlbc,IlDc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('KA,jb,jlbc,IlDc->IAKD', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('KA,jb,ljcb,IlDc->IAKD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('Kb,jD,Ib,jA->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp -= lib.einsum('Kb,jD,Ilbc,jlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('Kb,jD,Ilbc,jlAc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('Kb,jb,jA,ID->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp += lib.einsum('Kb,jb,jlAc,IlDc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('Kb,jb,jlAc,IlDc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('Kb,jc,IlDb,jlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('Kb,jc,IlDb,ljAc->IAKD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('jA,jD,Ilbc,Klbc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('jA,jD,Ib,Kb->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp -= lib.einsum('jA,jD,Ilbc,Klbc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jA,jb,Kb,ID->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp += lib.einsum('jA,jb,Klbc,IlDc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('jA,jb,Klbc,IlDc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += 1/2 * lib.einsum('jA,lD,Ijbc,Klbc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('jA,lb,IjDc,Klbc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('jA,lb,IjDc,Klcb->IAKD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jD,jb,Ib,KA->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp += lib.einsum('jD,jb,Ilbc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('jD,jb,Ilbc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('jD,lb,KjAc,Ilbc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('jD,lb,KjAc,Ilcb->IAKD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('jb,jb,KA,ID->IAKD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp -= lib.einsum('jb,jb,IlDc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('jb,jb,IlDc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jb,jc,IlDb,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('jb,lb,IjDc,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('jb,lc,IjDb,KlAc->IAKD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('jb,lc,IjDb,KlAc->IAKD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('jb,jb,KA,ID->IAKD', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp -= lib.einsum('jb,jb,IlDc,KlAc->IAKD', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp -= lib.einsum('jb,jb,IlDc,KlAc->IAKD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jb,jc,IlDb,KlAc->IAKD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('jb,lc,IjDb,KlAc->IAKD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp += lib.einsum('jb,lb,IjDc,KlAc->IAKD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('jb,lc,IjDb,KlAc->IAKD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)

        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jA,jlDb,lb->AD',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jA,jlDb,lb->AD',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jD,jlAb,lb->AD',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jD,jlAb,lb->AD',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jA,jb,lb,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('jA,jb,lmbc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jA,jb,lmbc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jA,lb,jmDc,mlcb->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jD,jb,lb,lA->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('jD,jb,lmbc,lmAc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jD,jb,lmbc,lmAc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jD,lb,jmAc,mlcb->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,lA,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jb,jc,lmAb,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,jA,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,jmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,jmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,lc,jmAb,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lc,jmAb,mlDc->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lc,jmDb,mlAc->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,lA,lD->AD',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,jc,lmAb,lmDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,mjAc,mlDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,lc,mjAb,mlDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        


        temp_2a[:nocc_a,nocc_a:,:nocc_a,nocc_a:] = temp
        temp_2a[:nocc_a,nocc_a:,nocc_a:,:nocc_a] = -temp.transpose(0,1,3,2)
        temp_2a[nocc_a:,:nocc_a,nocc_a:,:nocc_a] = temp.transpose(1,0,3,2)
        temp_2a[nocc_a:,:nocc_a,:nocc_a,nocc_a:] = -temp.transpose(1,0,2,3)




        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b]  = 1/2 * lib.einsum('ijab,klab->ijkl', Y_bbbb, Y_bbbb, optimize = True)

        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('jmab,lmab->jl', Y_bbbb, Y_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ja,la->jl', Y_bb, Y_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('mjab,mlab->jl', Y_abab, Y_abab, optimize = True)

        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('jmab,kmab->jk', Y_bbbb, Y_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ja,ka->jk', Y_bb, Y_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('mjab,mkab->jk', Y_abab, Y_abab, optimize = True)

        block_b = np.zeros((nocc_b,nocc_b,nocc_b,nocc_b))
        block_b[occ_list_b,occ_list_b,:,:] = 1/2 * lib.einsum('imab,lmab->il', Y_bbbb, Y_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('ia,la->il', Y_bb, Y_bb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('miab,mlab->il', Y_abab, Y_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += block_b.transpose(2,0,1,3)

        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('imab,kmab->ik', Y_bbbb, Y_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ia,ka->ik', Y_bb, Y_bb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('miab,mkab->ik', Y_abab, Y_abab, optimize = True)

        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ia,klab,jb->ijkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ja,klab,ib->ijkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ka,ijab,lb->ijkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('la,ijab,kb->ijkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)

        diag_b = 1/4 * lib.einsum('mnab,mnab->', Y_aaaa, Y_aaaa, optimize = True)
        diag_b += lib.einsum('ma,ma->', Y_aa, Y_aa, optimize = True)
        diag_b += lib.einsum('mnab,mnab->', Y_abab, Y_abab, optimize = True)
        diag_b += 1/4 * lib.einsum('mnab,mnab->', Y_bbbb, Y_bbbb, optimize = True)
        diag_b += lib.einsum('ma,ma->', Y_bb, Y_bb, optimize = True)
        
        block_1b = np.zeros((nocc_b**2,nocc_b**2))
        np.fill_diagonal(block_1b, diag_b)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += block_1b.reshape(nocc_b,nocc_b,nocc_b,nocc_b)
        
        off_diag_b = -1/4 * lib.einsum('mnab,mnab->', Y_aaaa, Y_aaaa, optimize = True)
        off_diag_b -= lib.einsum('ma,ma->', Y_aa, Y_aa, optimize = True)
        off_diag_b -= lib.einsum('mnab,mnab->', Y_abab, Y_abab, optimize = True)
        off_diag_b -= 1/4 * lib.einsum('mnab,mnab->', Y_bbbb, Y_bbbb, optimize = True)
        off_diag_b -= lib.einsum('ma,ma->', Y_bb, Y_bb, optimize = True)


        block_1b = np.zeros((nocc_b**2,nocc_b**2))
        np.fill_diagonal(block_1b, off_diag_b)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += block_1b.reshape(nocc_b,nocc_b,nocc_b,nocc_b).transpose(0,1,3,2)


        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ia,ka,jmbc,lmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ia,ka,jb,lb->ijkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ia,ka,mjbc,mlbc->ijkl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ia,kb,la,jb->ijkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ia,kb,lmac,jmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ia,kb,mlca,mjcb->ijkl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,la,jmbc,kmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ia,la,jb,kb->ijkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ia,la,mjbc,mkbc->ijkl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ia,lb,ka,jb->ijkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ia,lb,kmac,jmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ia,lb,mkca,mjcb->ijkl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ia,mb,klac,mjbc->ijkl', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ia,ma,jmbc,klbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ia,mb,klac,jmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ja,ka,imbc,lmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ja,ka,ib,lb->ijkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ja,ka,mibc,mlbc->ijkl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ja,kb,la,ib->ijkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ja,kb,lmac,imbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ja,kb,mlca,micb->ijkl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ja,la,imbc,kmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ja,la,ib,kb->ijkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ja,la,mibc,mkbc->ijkl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ja,lb,ka,ib->ijkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ja,lb,kmac,imbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ja,lb,mkca,micb->ijkl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ja,mb,klac,mibc->ijkl', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ja,ma,imbc,klbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ja,mb,klac,imbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ka,mb,ijac,mlbc->ijkl', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ka,ma,lmbc,ijbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ka,mb,ijac,lmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('la,mb,ijac,mkbc->ijkl', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,ma,kmbc,ijbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('la,mb,ijac,kmbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ma,ma,ijbc,klbc->ijkl', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ma,ma,ijbc,klbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ma,mb,ijac,klbc->ijkl', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ja,lmab,mb->jl',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ja,mlba,mb->jl',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('la,jmab,mb->jl',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('la,mjba,mb->jl',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ja,kmab,mb->jk',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ja,mkba,mb->jk',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ka,jmab,mb->jk',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ka,mjba,mb->jk',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        
        block_b = np.zeros((nocc_b,nocc_b,nocc_b,nocc_b))
        block_b[occ_list_b,occ_list_b,:,:] = lib.einsum('ia,lmab,mb->il',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('ia,mlba,mb->il',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('la,imab,mb->il',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('la,miba,mb->il',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += block_b.transpose(2,0,1,3)
        
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ia,kmab,mb->ik',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ia,mkba,mb->ik',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ka,imab,mb->ik',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ka,miba,mb->ik',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('ja,mb,lnac,mnbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('ja,mb,nlca,mnbc->jl',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += 1/4 * lib.einsum('ja,ma,mnbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += 1/2 * lib.einsum('ja,ma,mb,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += 1/2 * lib.einsum('ja,ma,nmbc,nlbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('ja,mb,lnac,mnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('ja,mb,nlca,nmcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('la,mb,jnac,mnbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('la,mb,njca,mnbc->jl',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += 1/4 * lib.einsum('la,ma,mnbc,jnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += 1/2 * lib.einsum('la,ma,mb,jb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += 1/2 * lib.einsum('la,ma,nmbc,njbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('la,mb,jnac,mnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('la,mb,njca,nmcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('ma,ma,jnbc,lnbc->jl',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ma,ma,jb,lb->jl',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ma,ma,njbc,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += lib.einsum('ma,mb,njac,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += lib.einsum('ma,na,mjbc,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ma,nb,mjac,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= 1/2 * lib.einsum('ma,ma,jnbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ma,ma,jb,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ma,ma,njbc,nlbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += lib.einsum('ma,mb,ja,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += lib.einsum('ma,mb,jnac,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += lib.einsum('ma,mb,njca,nlcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += lib.einsum('ma,nb,jmac,nlbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += lib.einsum('ma,nb,lmac,njbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] += 1/2 * lib.einsum('ma,na,jmbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,occ_list_b,:nocc_b] -= lib.einsum('ma,nb,jmac,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ja,mb,knac,mnbc->jk',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ja,mb,nkca,mnbc->jk',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= 1/4 * lib.einsum('ja,ma,mnbc,knbc->jk',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ja,ma,mb,kb->jk',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ja,ma,nmbc,nkbc->jk',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ja,mb,knac,mnbc->jk',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ja,mb,nkca,nmcb->jk',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ka,mb,jnac,mnbc->jk',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ka,mb,njca,mnbc->jk',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= 1/4 * lib.einsum('ka,ma,mnbc,jnbc->jk',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ka,ma,mb,jb->jk',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ka,ma,nmbc,njbc->jk',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ka,mb,jnac,mnbc->jk',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ka,mb,njca,nmcb->jk',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ma,ma,jnbc,knbc->jk',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ma,ma,jb,kb->jk',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ma,ma,njbc,nkbc->jk',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= lib.einsum('ma,mb,njac,nkbc->jk',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= lib.einsum('ma,na,mjbc,nkbc->jk',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ma,nb,mjac,nkbc->jk',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ma,ma,jnbc,knbc->jk',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ma,ma,jb,kb->jk',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ma,ma,njbc,nkbc->jk',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= lib.einsum('ma,mb,ja,kb->jk',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= lib.einsum('ma,mb,jnac,knbc->jk',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= lib.einsum('ma,mb,njca,nkcb->jk',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= lib.einsum('ma,nb,jmac,nkbc->jk',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= lib.einsum('ma,nb,kmac,njbc->jk',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ma,na,jmbc,knbc->jk',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[occ_list_b,:nocc_b,:nocc_b,occ_list_b] += lib.einsum('ma,nb,jmac,knbc->jk',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        block_b = np.zeros((nocc_b,nocc_b,nocc_b,nocc_b))
        block_b[occ_list_b,occ_list_b,:,:] = 1/2 * lib.einsum('ia,mb,lnac,mnbc->il',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('ia,mb,nlca,mnbc->il',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= 1/4 * lib.einsum('ia,ma,mnbc,lnbc->il',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= 1/2 * lib.einsum('ia,ma,mb,lb->il',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= 1/2 * lib.einsum('ia,ma,nmbc,nlbc->il',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('ia,mb,lnac,mnbc->il',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('ia,mb,nlca,nmcb->il',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('la,mb,inac,mnbc->il',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('la,mb,nica,mnbc->il',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= 1/4 * lib.einsum('la,ma,mnbc,inbc->il',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= 1/2 * lib.einsum('la,ma,mb,ib->il',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= 1/2 * lib.einsum('la,ma,nmbc,nibc->il',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('la,mb,inac,mnbc->il',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('la,mb,nica,nmcb->il',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('ma,ma,inbc,lnbc->il',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('ma,ma,ib,lb->il',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('ma,ma,nibc,nlbc->il',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= lib.einsum('ma,mb,niac,nlbc->il',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= lib.einsum('ma,na,mibc,nlbc->il',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('ma,nb,miac,nlbc->il',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += 1/2 * lib.einsum('ma,ma,inbc,lnbc->il',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('ma,ma,ib,lb->il',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('ma,ma,nibc,nlbc->il',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= lib.einsum('ma,mb,ia,lb->il',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= lib.einsum('ma,mb,inac,lnbc->il',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= lib.einsum('ma,mb,nica,nlcb->il',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= lib.einsum('ma,nb,imac,nlbc->il',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= lib.einsum('ma,nb,lmac,nibc->il',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] -= 1/2 * lib.einsum('ma,na,imbc,lnbc->il',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        block_b[occ_list_b,occ_list_b,:,:] += lib.einsum('ma,nb,imac,lnbc->il',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += block_b.transpose(2,0,1,3)
        
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ia,mb,knac,mnbc->ik',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ia,mb,nkca,mnbc->ik',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += 1/4 * lib.einsum('ia,ma,mnbc,knbc->ik',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ia,ma,mb,kb->ik',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ia,ma,nmbc,nkbc->ik',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ia,mb,knac,mnbc->ik',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ia,mb,nkca,nmcb->ik',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ka,mb,inac,mnbc->ik',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ka,mb,nica,mnbc->ik',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += 1/4 * lib.einsum('ka,ma,mnbc,inbc->ik',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ka,ma,mb,ib->ik',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ka,ma,nmbc,nibc->ik',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ka,mb,inac,mnbc->ik',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ka,mb,nica,nmcb->ik',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ma,ma,inbc,knbc->ik',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ma,ma,ib,kb->ik',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ma,ma,nibc,nkbc->ik',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += lib.einsum('ma,mb,niac,nkbc->ik',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += lib.einsum('ma,na,mibc,nkbc->ik',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ma,nb,miac,nkbc->ik',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= 1/2 * lib.einsum('ma,ma,inbc,knbc->ik',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ma,ma,ib,kb->ik',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ma,ma,nibc,nkbc->ik',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += lib.einsum('ma,mb,ia,kb->ik',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += lib.einsum('ma,mb,inac,knbc->ik',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += lib.einsum('ma,mb,nica,nkcb->ik',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += lib.einsum('ma,nb,imac,nkbc->ik',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += lib.einsum('ma,nb,kmac,nibc->ik',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] += 1/2 * lib.einsum('ma,na,imbc,knbc->ik',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[:nocc_b,occ_list_b,:nocc_b,occ_list_b] -= lib.einsum('ma,nb,imac,knbc->ik',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:]  = 1/2 * lib.einsum('ijab,ijcd->abcd', Y_bbbb, Y_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ia,ijcd,jb->abcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ib,ijcd,ja->abcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ic,ijab,jd->abcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('id,ijab,jc->abcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += 1/2 * lib.einsum('ie,ie,jkab,jkcd->abcd', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ia,ic,jb,jd->abcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ia,ic,jkeb,jked->abcd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,ic,jkbe,jkde->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ia,id,jb,jc->abcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ia,id,jkeb,jkec->abcd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ia,id,jkbe,jkce->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,ie,jkbe,jkcd->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ia,je,ikcd,jkeb->abcd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ia,jc,id,jb->abcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ia,jc,ikde,jkbe->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ia,jc,kied,kjeb->abcd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ia,jd,ic,jb->abcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ia,jd,ikce,jkbe->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ia,jd,kiec,kjeb->abcd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ia,je,ikcd,jkbe->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ib,ic,ja,jd->abcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ib,ic,jkea,jked->abcd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ib,ic,jkae,jkde->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ib,id,ja,jc->abcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ib,id,jkea,jkec->abcd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += 1/2 * lib.einsum('ib,id,jkae,jkce->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ib,ie,jkae,jkcd->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ib,je,ikcd,jkea->abcd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ib,jc,id,ja->abcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ib,jc,ikde,jkae->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ib,jc,kied,kjea->abcd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ib,jd,ic,ja->abcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ib,jd,ikce,jkae->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ib,jd,kiec,kjea->abcd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ib,je,ikcd,jkae->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,ie,jkde,jkab->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('ic,je,ikab,jked->abcd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ic,je,ikab,jkde->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= 1/2 * lib.einsum('id,ie,jkce,jkab->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('id,je,ikab,jkec->abcd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += lib.einsum('id,je,ikab,jkce->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] += 1/2 * lib.einsum('ie,ie,jkab,jkcd->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_2b[nocc_b:,nocc_b:,nocc_b:,nocc_b:] -= lib.einsum('ie,je,ikab,jkcd->abcd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        
        temp_ijcd  = lib.einsum('ic,jkda,ka->ijcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('ic,kjad,ka->ijcd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_ijcd -= lib.einsum('jc,ikda,ka->ijcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('jc,kiad,ka->ijcd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_ijcd += lib.einsum('kc,ka,ijda->ijcd', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
        temp_ijcd -= lib.einsum('id,jkca,ka->ijcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('id,kjac,ka->ijcd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_ijcd += lib.einsum('jd,ikca,ka->ijcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('jd,kiac,ka->ijcd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_ijcd -= lib.einsum('kd,ka,ijca->ijcd', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
        temp_ijcd += lib.einsum('ijcd,ka,ka->ijcd', t1_ccee_bbbb, Y_aa, Y_aa, optimize = True)
        temp_ijcd += lib.einsum('ijcd,ka,ka->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('ijca,ka,kd->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('ijda,ka,kc->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('ikcd,ka,ja->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('ikca,ka,jd->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('ikda,ka,jc->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('ia,ka,jkcd->ijcd', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
        temp_ijcd += lib.einsum('jkcd,ka,ia->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('jkca,ka,id->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('jkda,ka,ic->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('ja,ka,ikcd->ijcd', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
        temp_ijcd += lib.einsum('kiac,ka,jd->ijcd', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('kiad,ka,jc->ijcd', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('kjac,ka,id->ijcd', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('kjad,ka,ic->ijcd', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('ijcd,ka,ka->ijcd', t2_ccee_bbbb, Y_aa, Y_aa, optimize = True)
        temp_ijcd += lib.einsum('ijcd,ka,ka->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('ijca,ka,kd->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('ijda,ka,kc->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('ikcd,ka,ja->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('ikca,ka,jd->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('ikda,ka,jc->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('jkcd,ka,ia->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('jkca,ka,id->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('jkda,ka,ic->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('kiac,ka,jd->ijcd', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('kiad,ka,jc->ijcd', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('kjac,ka,id->ijcd', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
        temp_ijcd += lib.einsum('kjad,ka,ic->ijcd', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
        temp_ijcd -= lib.einsum('ic,ka,kd,ja->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd += lib.einsum('id,ka,kc,ja->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd -= lib.einsum('ia,ka,kc,jd->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd += lib.einsum('ia,ka,kd,jc->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd += lib.einsum('jc,ka,kd,ia->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd -= lib.einsum('jd,ka,kc,ia->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd += lib.einsum('ja,ka,kc,id->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd -= lib.einsum('ja,ka,kd,ic->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd += lib.einsum('ka,ka,ic,jd->ijcd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd -= lib.einsum('ka,ka,jc,id->ijcd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd -= lib.einsum('kc,ka,ia,jd->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd += lib.einsum('kc,ka,ja,id->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd += lib.einsum('kd,ka,ia,jc->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd -= lib.einsum('kd,ka,ja,ic->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd += lib.einsum('ka,ka,ic,jd->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_ijcd -= lib.einsum('ka,ka,jc,id->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            
        temp =- lib.einsum('id,ka->iakd', Y_bb, Y_bb, optimize = True)
        temp -= lib.einsum('ijdb,kjab->iakd', Y_bbbb, Y_bbbb, optimize = True)
        temp -= lib.einsum('jibd,jkba->iakd', Y_abab, Y_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jlba,jlbd->ad', Y_abab, Y_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('ja,jd->ad', Y_bb, Y_bb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jlab,jldb->ad', Y_bbbb, Y_bbbb, optimize = True)
        temp -= lib.einsum('ka,ijdb,jb->iakd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp -= lib.einsum('ka,jibd,jb->iakd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp += lib.einsum('ja,ijdb,kb->iakd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp -= lib.einsum('id,kjab,jb->iakd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp -= lib.einsum('id,jkba,jb->iakd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp += lib.einsum('jd,kjab,ib->iakd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp += lib.einsum('ib,kjab,jd->iakd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp += lib.einsum('kb,ijdb,ja->iakd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp += 1/2 * lib.einsum('id,kb,jb,ja->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp += 1/2 * lib.einsum('id,kb,jlcb,jlca->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += 1/4 * lib.einsum('id,kb,jlbc,jlac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= 1/2 * lib.einsum('id,jb,jlbc,lkca->iakd', Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('id,jb,jlbc,klac->iakd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp += 1/4 * lib.einsum('id,ja,jlbc,klbc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += 1/2 * lib.einsum('id,ja,jb,kb->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp += 1/2 * lib.einsum('id,ja,ljbc,lkbc->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('id,jb,jlbc,klac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= 1/2 * lib.einsum('id,jb,ljcb,lkca->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += 1/2 * lib.einsum('ib,ka,jb,jd->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp += 1/2 * lib.einsum('ib,ka,jlcb,jlcd->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += 1/4 * lib.einsum('ib,ka,jlbc,jldc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('ib,kb,ja,jd->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp -= lib.einsum('ib,kb,jlca,jlcd->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('ib,kb,jlac,jldc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += 1/2 * lib.einsum('ib,kc,jlab,jldc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += lib.einsum('ib,jc,klab,jlcd->iakd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('ib,ja,kb,jd->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp -= lib.einsum('ib,ja,klbc,jldc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('ib,ja,lkcb,ljcd->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('ib,jb,jd,ka->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp += lib.einsum('ib,jb,jldc,klac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += lib.einsum('ib,jb,ljcd,lkca->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('ib,jc,klab,jldc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= 1/2 * lib.einsum('ka,jb,jlbc,licd->iakd', Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('ka,jb,jlbc,ildc->iakd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp += 1/4 * lib.einsum('ka,jd,jlbc,ilbc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += 1/2 * lib.einsum('ka,jd,jb,ib->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp += 1/2 * lib.einsum('ka,jd,ljbc,libc->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('ka,jb,jlbc,ildc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= 1/2 * lib.einsum('ka,jb,ljcb,licd->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('kb,jc,ildb,jlca->iakd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('kb,jd,ib,ja->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp -= lib.einsum('kb,jd,ilbc,jlac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('kb,jd,licb,ljca->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('kb,jb,ja,id->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp += lib.einsum('kb,jb,jlac,ildc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += lib.einsum('kb,jb,ljca,licd->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('kb,jc,ildb,jlac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('jb,jb,ka,id->iakd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp -= lib.einsum('jb,jb,ildc,klac->iakd', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('jb,jb,licd,lkca->iakd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jb,jc,libd,lkca->iakd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jb,lb,jicd,lkca->iakd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('jb,lc,jibd,lkca->iakd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp -= 1/2 * lib.einsum('ja,jd,ilbc,klbc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('ja,jd,ib,kb->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp -= lib.einsum('ja,jd,libc,lkbc->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('ja,jb,kb,id->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp += lib.einsum('ja,jb,klbc,ildc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += lib.einsum('ja,jb,lkcb,licd->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('ja,lb,ijdc,lkbc->iakd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp += 1/2 * lib.einsum('ja,ld,ijbc,klbc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('ja,lb,ijdc,klbc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += lib.einsum('jd,jb,ib,ka->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp += lib.einsum('jd,jb,ilbc,klac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp += lib.einsum('jd,jb,licb,lkca->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jd,lb,kjac,libc->iakd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('jd,lb,kjac,ilbc->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('jb,jb,ka,id->iakd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp -= lib.einsum('jb,jb,ildc,klac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('jb,jb,licd,lkca->iakd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jb,jc,ildb,klac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('jb,lc,ijdb,lkca->iakd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp -= lib.einsum('jb,lc,kjab,licd->iakd', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp += lib.einsum('jb,lb,ijdc,klac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp -= lib.einsum('jb,lc,ijdb,klac->iakd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)

        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('ja,jldb,lb->ad',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('ja,ljbd,lb->ad',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jd,jlab,lb->ad',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jd,ljba,lb->ad',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,la,ld->ad',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lmca,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jb,jb,lmac,lmdc->ad',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,jc,lmba,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,jmca,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,lc,jmba,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('ja,jb,lb,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('ja,jb,lmcb,lmcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('ja,jb,lmbc,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('ja,lb,jmdc,lmbc->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('ja,lb,mjcd,lmbc->ad',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('ja,lb,jmdc,lmbc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('ja,lb,mjcd,mlcb->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jd,jb,lb,la->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jd,jb,lmcb,lmca->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('jd,jb,lmbc,lmac->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jd,lb,jmac,lmbc->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jd,lb,mjca,lmbc->ad',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jd,lb,jmac,lmbc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jd,lb,mjca,mlcb->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,la,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lmca,lmcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jb,jb,lmac,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jb,jc,lmab,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lc,jmab,lmcd->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lc,jmdb,lmca->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,ja,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,jmac,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,mjca,mlcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,lc,jmab,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            
        temp_2b[:nocc_b,nocc_b:,:nocc_b,nocc_b:] = temp
        temp_2b[:nocc_b,nocc_b:,nocc_b:,:nocc_b] = -temp.transpose(0,1,3,2)
        temp_2b[nocc_b:,:nocc_b,nocc_b:,:nocc_b] = temp.transpose(1,0,3,2)
        temp_2b[nocc_b:,:nocc_b,:nocc_b,nocc_b:] = -temp.transpose(1,0,2,3)
    
#TPDM AD-X
        if (method == "adc(2)-x"):

            temp_IJCD += 1/4 * lib.einsum('IJCD,klab,klab->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('IJCD,klab,klab->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD += 1/4 * lib.einsum('IJCD,klab,klab->IJCD', t1_ccee_aaaa, Y_bbbb, Y_bbbb, optimize = True)
            temp_IJCD -= 1/2 * lib.einsum('IJCa,klab,klDb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD -= lib.einsum('IJCa,klab,klDb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD += 1/2 * lib.einsum('IJDa,klab,klCb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('IJDa,klab,klCb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD += 1/4 * lib.einsum('IJab,klab,klCD->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD -= 1/2 * lib.einsum('IkCD,klab,Jlab->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD -= lib.einsum('IkCD,klab,Jlab->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD += lib.einsum('IkCa,klab,JlDb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('IkCa,klab,JlDb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD -= lib.einsum('IkDa,klab,JlCb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD -= lib.einsum('IkDa,klab,JlCb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD -= 1/2 * lib.einsum('Ikab,klab,JlCD->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('IkCa,klab,JlDb->IJCD', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
            temp_IJCD += lib.einsum('IkCa,lkba,JlDb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
            temp_IJCD -= lib.einsum('IkDa,klab,JlCb->IJCD', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
            temp_IJCD -= lib.einsum('IkDa,lkba,JlCb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('Ikab,lkab,JlCD->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
            temp_IJCD += 1/2 * lib.einsum('JkCD,klab,Ilab->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('JkCD,klab,Ilab->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD -= lib.einsum('JkCa,klab,IlDb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD -= lib.einsum('JkCa,klab,IlDb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD += lib.einsum('JkDa,klab,IlCb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('JkDa,klab,IlCb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
            temp_IJCD += 1/2 * lib.einsum('Jkab,klab,IlCD->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD -= lib.einsum('JkCa,klab,IlDb->IJCD', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
            temp_IJCD -= lib.einsum('JkCa,lkba,IlDb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('JkDa,klab,IlCb->IJCD', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
            temp_IJCD += lib.einsum('JkDa,lkba,IlCb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
            temp_IJCD -= lib.einsum('Jkab,lkab,IlCD->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
            temp_IJCD += 1/4 * lib.einsum('klCD,klab,IJab->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD -= 1/2 * lib.einsum('klCa,klab,IJDb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD += 1/2 * lib.einsum('klDa,klab,IJCb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IJCD += lib.einsum('klCa,klba,IJDb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
            temp_IJCD -= lib.einsum('klDa,klba,IJCb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
            

            temp_ijcd += 1/4 * lib.einsum('ijcd,klab,klab->ijcd', t1_ccee_bbbb, Y_aaaa, Y_aaaa, optimize = True)
            temp_ijcd += lib.einsum('ijcd,klab,klab->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd += 1/4 * lib.einsum('ijcd,klab,klab->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd -= lib.einsum('ijca,klba,klbd->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd -= 1/2 * lib.einsum('ijca,klab,kldb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd += lib.einsum('ijda,klba,klbc->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd += 1/2 * lib.einsum('ijda,klab,klcb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd += 1/4 * lib.einsum('ijab,klab,klcd->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd -= 1/2 * lib.einsum('ikcd,klab,jlab->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd -= lib.einsum('ikcd,lkab,ljab->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd += lib.einsum('ikca,klab,jldb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd += lib.einsum('ikca,lkba,ljbd->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd -= lib.einsum('ikda,klab,jlcb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd -= lib.einsum('ikda,lkba,ljbc->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd -= 1/2 * lib.einsum('ikab,klab,jlcd->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd += 1/2 * lib.einsum('jkcd,klab,ilab->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd += lib.einsum('jkcd,lkab,liab->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd -= lib.einsum('jkca,klab,ildb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd -= lib.einsum('jkca,lkba,libd->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd += lib.einsum('jkda,klab,ilcb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd += lib.einsum('jkda,lkba,libc->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
            temp_ijcd += 1/2 * lib.einsum('jkab,klab,ilcd->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd += lib.einsum('kiac,klab,ljbd->ijcd', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
            temp_ijcd += lib.einsum('kiac,klab,jldb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
            temp_ijcd -= lib.einsum('kiad,klab,ljbc->ijcd', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
            temp_ijcd -= lib.einsum('kiad,klab,jlcb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
            temp_ijcd += lib.einsum('kiab,klab,jlcd->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
            temp_ijcd -= lib.einsum('kjac,klab,libd->ijcd', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
            temp_ijcd -= lib.einsum('kjac,klab,ildb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
            temp_ijcd += lib.einsum('kjad,klab,libc->ijcd', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
            temp_ijcd += lib.einsum('kjad,klab,ilcb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
            temp_ijcd -= lib.einsum('kjab,klab,ilcd->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
            temp_ijcd += lib.einsum('klac,klab,ijdb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
            temp_ijcd -= lib.einsum('klad,klab,ijcb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
            temp_ijcd += 1/4 * lib.einsum('klcd,klab,ijab->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd -= 1/2 * lib.einsum('klca,klab,ijdb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ijcd += 1/2 * lib.einsum('klda,klab,ijcb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
        
        temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] = temp_IJCD
        temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] = temp_IJCD.transpose(2,3,0,1)

        temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] = temp_ijcd
        temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] = temp_ijcd.transpose(2,3,0,1)
        







        IjKl  = lib.einsum('Ijab,Klab->IjKl', Y_abab, Y_abab, optimize = True)

        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jmab,lmab->jl', Y_bbbb, Y_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ja,la->jl', Y_bb, Y_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('mjab,mlab->jl', Y_abab, Y_abab, optimize = True)
        
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Imab,Kmab->IK', Y_aaaa, Y_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ia,Ka->IK', Y_aa, Y_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Imab,Kmab->IK', Y_abab, Y_abab, optimize = True)

        IjKl += lib.einsum('Ia,Klab,jb->IjKl', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IjKl += lib.einsum('Ka,Ijab,lb->IjKl', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IjKl += lib.einsum('ja,Klba,Ib->IjKl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IjKl += lib.einsum('la,Ijba,Kb->IjKl', t1_ce_bb, Y_abab, Y_aa, optimize = True)

        diag_sc = 1/4 * lib.einsum('mnab,mnab->', Y_aaaa, Y_aaaa, optimize = True)
        diag_sc += lib.einsum('ma,ma->', Y_aa, Y_aa, optimize = True)
        diag_sc += lib.einsum('mnab,mnab->', Y_abab, Y_abab, optimize = True)
        diag_sc += 1/4 * lib.einsum('mnab,mnab->', Y_bbbb, Y_bbbb, optimize = True)
        diag_sc += lib.einsum('ma,ma->', Y_bb, Y_bb, optimize = True)
        
        block_sc = np.zeros((nocc_a*nocc_b,nocc_a*nocc_b))
        np.fill_diagonal(block_sc, diag_sc)
        IjKl += block_sc.reshape(nocc_a,nocc_b,nocc_a,nocc_b)
        
        
        IjKl += 1/2 * lib.einsum('Ia,Ka,jmbc,lmbc->IjKl', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl += lib.einsum('Ia,Ka,jb,lb->IjKl', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl += lib.einsum('Ia,Ka,mjbc,mlbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('Ia,Kb,mlac,mjbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ia,lb,Ka,jb->IjKl', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IjKl += lib.einsum('Ia,lb,Kmac,mjcb->IjKl', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ia,lb,Kmac,jmbc->IjKl', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl -= lib.einsum('Ia,ma,mjbc,Klbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ia,mb,Klac,mjbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('Ia,mb,Klac,jmbc->IjKl', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl += lib.einsum('Ka,jb,Ia,lb->IjKl', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IjKl += lib.einsum('Ka,jb,Imac,mlcb->IjKl', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ka,jb,Imac,lmbc->IjKl', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl -= lib.einsum('Ka,ma,mlbc,Ijbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ka,mb,Ijac,mlbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('Ka,mb,Ijac,lmbc->IjKl', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl += 1/2 * lib.einsum('ja,la,Imbc,Kmbc->IjKl', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl += lib.einsum('ja,la,Ib,Kb->IjKl', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl += lib.einsum('ja,la,Imbc,Kmbc->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ja,lb,Kmca,Imcb->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ja,mb,Klca,Imbc->IjKl', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl -= lib.einsum('ja,ma,Imbc,Klbc->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ja,mb,Klca,Imcb->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('la,mb,Ijca,Kmbc->IjKl', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl -= lib.einsum('la,ma,Kmbc,Ijbc->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('la,mb,Ijca,Kmcb->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ma,ma,Ijbc,Klbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ma,mb,Ijac,Klbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ma,ma,Ijbc,Klbc->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ma,mb,Ijca,Klcb->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ja,lmab,mb->jl',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ja,mlba,mb->jl',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('la,jmab,mb->jl',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('la,mjba,mb->jl',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ia,Kmab,mb->IK', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ia,Kmab,mb->IK', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ka,Imab,mb->IK', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ka,Imab,mb->IK', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,mb,lnac,mnbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,mb,nlca,mnbc->jl',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/4 * lib.einsum('ja,ma,mnbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,ma,mb,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,ma,nmbc,nlbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,mb,lnac,mnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,mb,nlca,nmcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('la,mb,jnac,mnbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('la,mb,njca,mnbc->jl',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/4 * lib.einsum('la,ma,mnbc,jnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('la,ma,mb,jb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('la,ma,nmbc,njbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('la,mb,jnac,mnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('la,mb,njca,nmcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ma,ma,jnbc,lnbc->jl',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,ma,jb,lb->jl',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,ma,njbc,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,mb,njac,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,na,mjbc,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,nb,mjac,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ma,ma,jnbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,ma,jb,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,ma,njbc,nlbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,mb,ja,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,mb,jnac,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,mb,njca,nlcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,nb,jmac,nlbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,nb,lmac,njbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ma,na,jmbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,nb,jmac,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        IjKl[:,occ_list_b,:,occ_list_b] += 1/4 * lib.einsum('Ia,ma,mnbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('Ia,ma,mb,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('Ia,ma,mnbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ia,mb,Knac,nmcb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/4 * lib.einsum('Ka,ma,mnbc,Inbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('Ka,ma,mb,Ib->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('Ka,ma,mnbc,Inbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ka,mb,Inac,nmcb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,ma,Ib,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,mb,Ia,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,mb,Inac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,mb,Inac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('ma,na,Imbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,nb,Imac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,nb,Imac,Kncb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,ma,Ib,Kb->IK',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,mb,Inca,Kncb->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,nb,Imca,Knbc->IK',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,na,Imbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,nb,Imca,Kncb->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            
            
        AbCd  = lib.einsum('ijAb,ijCd->AbCd', Y_abab, Y_abab, optimize = True)
        AbCd += lib.einsum('iA,ijCd,jb->AbCd', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        AbCd += lib.einsum('iC,ijAb,jd->AbCd', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        AbCd += lib.einsum('ib,jiCd,jA->AbCd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        AbCd += lib.einsum('id,jiAb,jC->AbCd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        AbCd += lib.einsum('iA,iC,jb,jd->AbCd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        AbCd += lib.einsum('iA,iC,jkeb,jked->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += 1/2 * lib.einsum('iA,iC,jkbe,jkde->AbCd', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('iA,ie,jkeb,jkCd->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('iA,jC,iked,jkeb->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iA,je,ikCd,jkeb->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iA,jd,iC,jb->AbCd', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        AbCd += lib.einsum('iA,jd,ikCe,kjeb->AbCd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iA,jd,ikCe,jkbe->AbCd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('iA,je,ikCd,jkbe->AbCd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('iC,ie,jked,jkAb->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iC,je,ikAb,jked->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iC,jb,iA,jd->AbCd', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        AbCd += lib.einsum('iC,jb,ikAe,kjed->AbCd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iC,jb,ikAe,jkde->AbCd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('iC,je,ikAb,jkde->AbCd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd += lib.einsum('ie,ie,jkAb,jkCd->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ie,je,ikAb,jkCd->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ib,id,jA,jC->AbCd', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        AbCd += 1/2 * lib.einsum('ib,id,jkAe,jkCe->AbCd', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AbCd += lib.einsum('ib,id,jkAe,jkCe->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ib,ie,jkAe,jkCd->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ib,je,kiCd,jkAe->AbCd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        AbCd -= lib.einsum('ib,jd,kiCe,kjAe->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ib,je,kiCd,kjAe->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('id,ie,jkCe,jkAb->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('id,je,kiAb,jkCe->AbCd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        AbCd += lib.einsum('id,je,kiAb,kjCe->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ie,ie,jkAb,jkCd->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ie,je,kiAb,kjCd->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            
            
            
        IjCd  = lib.einsum('IC,jkda,ka->IjCd', t1_ce_aa, Y_bbbb, Y_bb, optimize = True)
        IjCd += lib.einsum('IC,kjad,ka->IjCd', t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IjCd -= lib.einsum('kC,ka,Ijad->IjCd', t1_ce_aa, Y_aa, Y_abab, optimize = True)
        IjCd += lib.einsum('IjCd,ka,ka->IjCd', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IjCd,ka,ka->IjCd', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('IjCa,ka,kd->IjCd', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('Ijad,ka,kC->IjCd', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IkCa,ka,jd->IjCd', t1_ccee_aaaa, Y_aa, Y_bb, optimize = True)
        IjCd -= lib.einsum('Ia,ka,kjCd->IjCd', t1_ce_aa, Y_aa, Y_abab, optimize = True)
        IjCd -= lib.einsum('IkCd,ka,ja->IjCd', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('IkCa,ka,jd->IjCd', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('jd,IkCa,ka->IjCd', t1_ce_bb, Y_aaaa, Y_aa, optimize = True)
        IjCd += lib.einsum('jd,IkCa,ka->IjCd', t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IjCd -= lib.einsum('kd,ka,IjCa->IjCd', t1_ce_bb, Y_bb, Y_abab, optimize = True)
        IjCd += lib.einsum('jkda,ka,IC->IjCd', t1_ccee_bbbb, Y_bb, Y_aa, optimize = True)
        IjCd -= lib.einsum('ja,ka,IkCd->IjCd', t1_ce_bb, Y_bb, Y_abab, optimize = True)
        IjCd -= lib.einsum('kjCd,ka,Ia->IjCd', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('kjad,ka,IC->IjCd', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IjCd,ka,ka->IjCd', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IjCd,ka,ka->IjCd', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('IjCa,ka,kd->IjCd', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('Ijad,ka,kC->IjCd', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IkCa,ka,jd->IjCd', t2_ccee_aaaa, Y_aa, Y_bb, optimize = True)
        IjCd -= lib.einsum('IkCd,ka,ja->IjCd', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('IkCa,ka,jd->IjCd', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('jkda,ka,IC->IjCd', t2_ccee_bbbb, Y_bb, Y_aa, optimize = True)
        IjCd -= lib.einsum('kjCd,ka,Ia->IjCd', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('kjad,ka,IC->IjCd', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('IC,ka,kd,ja->IjCd', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjCd -= lib.einsum('Ia,ka,kC,jd->IjCd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd -= lib.einsum('jd,ka,kC,Ia->IjCd', Y_bb, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjCd -= lib.einsum('ja,ka,kd,IC->IjCd', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IjCd -= lib.einsum('kC,ka,Ia,jd->IjCd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd += lib.einsum('ka,ka,IC,jd->IjCd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd -= lib.einsum('kd,ka,ja,IC->IjCd', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IjCd += lib.einsum('ka,ka,IC,jd->IjCd', Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
            
            
            
            
        IaKd =- lib.einsum('Ijbd,Kjba->IaKd', Y_abab, Y_abab, optimize = True)

        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jlba,jlbd->ad', Y_abab, Y_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('ja,jd->ad', Y_bb, Y_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jlab,jldb->ad', Y_bbbb, Y_bbbb, optimize = True)
        
        IaKd -= lib.einsum('Ib,Kjba,jd->IaKd', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IaKd -= lib.einsum('Kb,Ijbd,ja->IaKd', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IaKd -= lib.einsum('ja,Ijbd,Kb->IaKd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IaKd -= lib.einsum('jd,Kjba,Ib->IaKd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IaKd -= lib.einsum('Ib,Kb,ja,jd->IaKd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd -= lib.einsum('Ib,Kb,jlca,jlcd->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 1/2 * lib.einsum('Ib,Kb,jlac,jldc->IaKd', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('Ib,Kc,jlba,jlcd->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('Ib,jb,jlcd,Klca->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Ib,jc,Klba,jlcd->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Ib,ja,Kb,jd->IaKd', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IaKd -= lib.einsum('Ib,ja,Klbc,ljcd->IaKd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Ib,ja,Klbc,jldc->IaKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('Ib,jc,Klba,jldc->IaKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('Kb,jb,jlca,Ilcd->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Kb,jc,Ilbd,jlca->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Kb,jd,Ib,ja->IaKd', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IaKd -= lib.einsum('Kb,jd,Ilbc,ljca->IaKd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Kb,jd,Ilbc,jlac->IaKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('Kb,jc,Ilbd,jlac->IaKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaKd -= lib.einsum('jb,jb,Ilcd,Klca->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('jb,jc,Ilbd,Klca->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 1/2 * lib.einsum('ja,jd,Ilbc,Klbc->IaKd', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IaKd -= lib.einsum('ja,jd,Ib,Kb->IaKd', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IaKd -= lib.einsum('ja,jd,Ilbc,Klbc->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ja,jb,Klcb,Ilcd->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ja,lb,Ijcd,Klbc->IaKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd += lib.einsum('ja,ld,Ijbc,Klbc->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ja,lb,Ijcd,Klcb->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('jd,jb,Ilcb,Klca->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('jd,lb,Kjca,Ilbc->IaKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd -= lib.einsum('jd,lb,Kjca,Ilcb->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('jb,jb,Ilcd,Klca->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('jb,lb,Ijcd,Klca->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('ja,jldb,lb->ad',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('ja,ljbd,lb->ad',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jd,jlab,lb->ad',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jd,ljba,lb->ad',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,la,ld->ad',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,lmca,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jb,jb,lmac,lmdc->ad',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,jc,lmba,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,jmca,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,lc,jmba,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,jb,lb,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,jb,lmcb,lmcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('ja,jb,lmbc,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,lb,jmdc,lmbc->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,lb,mjcd,lmbc->ad',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,lb,jmdc,lmbc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,lb,mjcd,mlcb->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jd,jb,lb,la->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jd,jb,lmcb,lmca->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('jd,jb,lmbc,lmac->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jd,lb,jmac,lmbc->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jd,lb,mjca,lmbc->ad',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jd,lb,jmac,lmbc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jd,lb,mjca,mlcb->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,la,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,lmca,lmcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jb,jb,lmac,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jb,jc,lmab,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lc,jmab,lmcd->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lc,jmdb,lmca->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,ja,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,jmac,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,mjca,mlcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,lc,jmab,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            
            
        IakD =- lib.einsum('ID,ka->IakD', Y_aa, Y_bb, optimize = True)
        IakD -= lib.einsum('IjDb,jkba->IakD', Y_aaaa, Y_abab, optimize = True)
        IakD -= lib.einsum('IjDb,kjab->IakD', Y_abab, Y_bbbb, optimize = True)
        IakD -= lib.einsum('ID,kjab,jb->IakD', t1_ce_aa, Y_bbbb, Y_bb, optimize = True)
        IakD -= lib.einsum('ID,jkba,jb->IakD', t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IakD += lib.einsum('jD,jkba,Ib->IakD', t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IakD += lib.einsum('Ib,jkba,jD->IakD', t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IakD -= lib.einsum('ka,IjDb,jb->IakD', t1_ce_bb, Y_aaaa, Y_aa, optimize = True)
        IakD -= lib.einsum('ka,IjDb,jb->IakD', t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IakD += lib.einsum('ja,IjDb,kb->IakD', t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IakD += lib.einsum('kb,IjDb,ja->IakD', t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IakD += 1/2 * lib.einsum('ID,kb,jb,ja->IakD', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IakD += 1/2 * lib.einsum('ID,kb,jlcb,jlca->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += 1/4 * lib.einsum('ID,kb,jlbc,jlac->IakD', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IakD -= 1/2 * lib.einsum('ID,jb,jlbc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ID,jb,jlbc,klac->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD += 1/4 * lib.einsum('ID,ja,jlbc,klbc->IakD', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IakD += 1/2 * lib.einsum('ID,ja,jb,kb->IakD', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IakD += 1/2 * lib.einsum('ID,ja,ljbc,lkbc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ID,jb,jlbc,klac->IakD', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IakD -= 1/2 * lib.einsum('ID,jb,ljcb,lkca->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += 1/2 * lib.einsum('Ib,ka,jb,jD->IakD', Y_aa, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IakD += 1/4 * lib.einsum('Ib,ka,jlbc,jlDc->IakD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IakD += 1/2 * lib.einsum('Ib,ka,jlbc,jlDc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('Ib,kc,jlba,jlDc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('Ib,jb,jD,ka->IakD', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IakD += lib.einsum('Ib,jb,jlDc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('Ib,jb,jlDc,klac->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('Ib,jc,lkba,jlDc->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD -= lib.einsum('Ib,ja,lkbc,ljDc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('Ib,jc,lkba,ljDc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += 1/4 * lib.einsum('ka,jD,jlbc,Ilbc->IakD', Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IakD += 1/2 * lib.einsum('ka,jD,jb,Ib->IakD', Y_bb, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IakD += 1/2 * lib.einsum('ka,jD,jlbc,Ilbc->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ka,jb,jlbc,IlDc->IakD', Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IakD -= 1/2 * lib.einsum('ka,jb,jlbc,IlDc->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ka,jb,jlbc,IlDc->IakD', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ka,jb,ljcb,IlDc->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD -= lib.einsum('kb,jD,Ilcb,jlca->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('kb,jc,IlDb,jlca->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('kb,jb,ja,ID->IakD', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IakD += lib.einsum('kb,jb,jlac,IlDc->IakD', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('kb,jb,ljca,IlDc->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD -= lib.einsum('kb,jc,IlDb,jlac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD += lib.einsum('jD,jb,Ib,ka->IakD', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IakD += lib.einsum('jD,jb,Ilbc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('jD,jb,Ilbc,klac->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('jD,lb,jkca,Ilbc->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD -= lib.einsum('jD,la,jkbc,Ilbc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('jD,lb,jkca,Ilcb->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,jb,ID,ka->IakD', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IakD -= lib.einsum('jb,jb,IlDc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,jb,IlDc,klac->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD += lib.einsum('jb,jc,IlDb,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('jb,lb,IjDc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,lc,IjDb,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,lc,IjDb,klac->IakD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_bbbb, optimize = True)
        IakD += lib.einsum('ja,jb,kb,ID->IakD', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IakD += lib.einsum('ja,jb,klbc,IlDc->IakD', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('ja,jb,lkcb,IlDc->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD += lib.einsum('ja,lb,IjDc,lkbc->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('ja,lb,IjDc,klbc->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('jb,jb,ID,ka->IakD', Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IakD -= lib.einsum('jb,jb,IlDc,lkca->IakD', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,jb,IlDc,klac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD += lib.einsum('jb,jc,IlDb,klac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('jb,lc,IjDb,lkca->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('jb,lb,IjDc,klac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('jb,lc,IjDb,klac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        
        
        
        
        
        iAkD =- lib.einsum('jiDb,jkAb->iAkD', Y_abab, Y_abab, optimize = True)
        
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jA,jD->AD', Y_aa, Y_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jlAb,jlDb->AD', Y_aaaa, Y_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jlAb,jlDb->AD', Y_abab, Y_abab, optimize = True)
        
        iAkD -= lib.einsum('jA,jiDb,kb->iAkD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        iAkD -= lib.einsum('jD,jkAb,ib->iAkD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        iAkD -= lib.einsum('ib,jkAb,jD->iAkD', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        iAkD -= lib.einsum('kb,jiDb,jA->iAkD', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        iAkD -= lib.einsum('ib,kb,jA,jD->iAkD', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD -= 1/2 * lib.einsum('ib,kb,jlAc,jlDc->iAkD', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD -= lib.einsum('ib,kb,jlAc,jlDc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('ib,kc,jlAb,jlDc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('ib,jA,kb,jD->iAkD', Y_bb, Y_aa, t1_ce_bb, t1_ce_aa, optimize = True)
        iAkD -= lib.einsum('ib,jA,klbc,jlDc->iAkD', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('ib,jA,lkcb,jlDc->iAkD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        iAkD += lib.einsum('ib,jc,lkAb,jlDc->iAkD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        iAkD += lib.einsum('ib,jb,ljDc,lkAc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('ib,jc,lkAb,ljDc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('kb,jD,ib,jA->iAkD', Y_bb, Y_aa, t1_ce_bb, t1_ce_aa, optimize = True)
        iAkD -= lib.einsum('kb,jD,ilbc,jlAc->iAkD', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('kb,jD,licb,jlAc->iAkD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        iAkD += lib.einsum('kb,jc,liDb,jlAc->iAkD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        iAkD += lib.einsum('kb,jb,ljAc,liDc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('kb,jc,liDb,ljAc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= 1/2 * lib.einsum('jA,jD,ilbc,klbc->iAkD', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        iAkD -= lib.einsum('jA,jD,ib,kb->iAkD', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        iAkD -= lib.einsum('jA,jD,libc,lkbc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jA,jb,lkbc,liDc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jA,lD,jibc,lkbc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('jA,lb,jiDc,lkbc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jA,lb,jiDc,klbc->iAkD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        iAkD += lib.einsum('jD,jb,libc,lkAc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('jD,lb,jkAc,libc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jD,lb,jkAc,ilbc->iAkD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        iAkD -= lib.einsum('jb,jb,liDc,lkAc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jb,lb,jiDc,lkAc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('jb,jb,liDc,lkAc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jb,jc,liDb,lkAc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jA,jlDb,lb->AD',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jA,jlDb,lb->AD',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jD,jlAb,lb->AD',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jD,jlAb,lb->AD',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jA,jb,lb,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('jA,jb,lmbc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jA,jb,lmbc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jA,lb,jmDc,mlcb->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jD,jb,lb,lA->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('jD,jb,lmbc,lmAc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jD,jb,lmbc,lmAc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jD,lb,jmAc,mlcb->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lA,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jb,jc,lmAb,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,jA,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,jmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,jmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,lc,jmAb,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lc,jmAb,mlDc->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lc,jmDb,mlAc->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lA,lD->AD',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,jc,lmAb,lmDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,mjAc,mlDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,lc,mjAb,mlDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
     
        if (method == "adc(2)-x"):

            IjCd += 1/4 * lib.einsum('IjCd,klab,klab->IjCd', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd += lib.einsum('IjCd,klab,klab->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += 1/4 * lib.einsum('IjCd,klab,klab->IjCd', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd -= lib.einsum('IjCa,klba,klbd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= 1/2 * lib.einsum('IjCa,klab,kldb->IjCd', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd -= 1/2 * lib.einsum('Ijad,klab,klCb->IjCd', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd -= lib.einsum('Ijad,klab,klCb->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('Ijab,klab,klCd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('IkCa,klab,ljbd->IjCd', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
            IjCd += lib.einsum('IkCa,klab,jldb->IjCd', t1_ccee_aaaa, Y_abab, Y_bbbb, optimize = True)
            IjCd += 1/2 * lib.einsum('Ikab,klab,ljCd->IjCd', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
            IjCd -= 1/2 * lib.einsum('IkCd,klab,jlab->IjCd', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd -= lib.einsum('IkCd,lkab,ljab->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('IkCa,klab,jldb->IjCd', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd += lib.einsum('IkCa,lkba,ljbd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('Ikad,lkab,ljCb->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('Ikab,lkab,ljCd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('jkda,klab,IlCb->IjCd', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)
            IjCd += lib.einsum('jkda,lkba,IlCb->IjCd', t1_ccee_bbbb, Y_abab, Y_aaaa, optimize = True)
            IjCd += 1/2 * lib.einsum('jkab,klab,IlCd->IjCd', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)
            IjCd -= 1/2 * lib.einsum('kjCd,klab,Ilab->IjCd', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd -= lib.einsum('kjCd,klab,Ilab->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('kjCa,klba,Ilbd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('kjad,klab,IlCb->IjCd', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd += lib.einsum('kjad,klab,IlCb->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('kjab,klab,IlCd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += 1/2 * lib.einsum('klCa,klab,Ijbd->IjCd', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
            IjCd += lib.einsum('klCd,klab,Ijab->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('klCa,klba,Ijbd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('klad,klab,IjCb->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += 1/2 * lib.einsum('klda,klab,IjCb->IjCd', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)


        abab = (IjKl,AbCd,IjCd,IaKd,IakD,iAkD)
        mixed_spin = np.append(mixed_spin,abab)
    

        norm = np.linalg.norm(temp_2a - temp_2a.transpose(2,3,0,1))
        print("Total TPDM_a  Hermiticity",norm)
        norm = np.linalg.norm(temp_2a + temp_2a.transpose(1,0,2,3))
        print("Total TPDM_a bra symmetry",norm)
        norm = np.linalg.norm(temp_2a + temp_2a.transpose(0,1,3,2))
        print("Total TPDM_a ket symmetry",norm)
        print("TPDM_a trace",np.einsum('pqpq',temp_2a))
        print("") 

        norm = np.linalg.norm(temp_2b - temp_2b.transpose(2,3,0,1))
        print("Total TPDM_b  Hermiticity",norm)
        norm = np.linalg.norm(temp_2b + temp_2b.transpose(1,0,2,3))
        print("Total TPDM_b bra symmetry",norm)
        norm = np.linalg.norm(temp_2b + temp_2b.transpose(0,1,3,2))
        print("Total TPDM_b ket symmetry",norm)
        print("TPDM trace",np.einsum('pqpq',temp_2b))
        print("")

        tpdm_a = np.append(tpdm_a,temp_2a)
        tpdm_b = np.append(tpdm_b,temp_2b)

    tpdm_a = tpdm_a.reshape(nroots,nmo_a, nmo_a, nmo_a, nmo_a)
    tpdm_b = tpdm_b.reshape(nroots,nmo_b, nmo_b, nmo_b, nmo_b)

    tpdm = (tpdm_a,tpdm_b ,mixed_spin)

    return tpdm
#@profile
def get_old_spin_contamination(adc):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)
    method = adc.method
    dm_a = adc.dm_a.copy()
    dm_b = adc.dm_b.copy()
    print("this is the old spin c")
    exit()

    t1 = adc.t1
    t2 = adc.t2
    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b


    vir_list_a = range(nvir_a)
    vir_list_b = range(nvir_b)
    occ_list_a = range(nocc_a)
    occ_list_b = range(nocc_b)


    ovlp = adc._scf.get_ovlp(adc._scf.mol).copy()
 
    delta = np.dot(adc.mo_coeff[0].transpose(),np.dot(ovlp,adc.mo_coeff[1]))

    if adc.f_ov is None:
        f_ov_a = np.zeros((nocc_a, nvir_a))
        f_ov_b = np.zeros((nocc_b, nvir_b))
        t1_ce_aa = np.zeros((nocc_a, nvir_a))
        t1_ce_bb = np.zeros((nocc_b, nvir_b))
    else:
        f_ov_a, f_ov_b = adc.f_ov
        t1_ce_aa = t1[2][0][:]
        t1_ce_bb = t1[2][1][:]

    t2_ce_aa = t1[0][0][:]
    t2_ce_bb = t1[0][1][:]

    t1_ccee_aaaa = t2[0][0][:]
    t1_ccee_abab = t2[0][1][:]
    t1_ccee_bbbb = t2[0][2][:]

    t2_ccee_aaaa = t2[1][0][:]
    t2_ccee_abab = t2[1][1][:]
    t2_ccee_bbbb = t2[1][2][:]

    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    temp_a = np.zeros((nmo_a,nmo_a))
    temp_b = np.zeros((nmo_b,nmo_b))

    temp_2a = np.zeros((nmo_a,nmo_a,nmo_a,nmo_a))
    temp_2b = np.zeros((nmo_b,nmo_b,nmo_b,nmo_b))

    temp_2 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))

    n_singles_a = nocc_a * nvir_a
    n_singles_b = nocc_b * nvir_b
    n_doubles_aaaa = nocc_a * (nocc_a - 1) * nvir_a * (nvir_a -1) // 4
    n_doubles_ab = nocc_a * nocc_b * nvir_a * nvir_b
    n_doubles_bbbb = nocc_b * (nocc_b - 1) * nvir_b * (nvir_b -1) // 4

    dim = n_singles_a + n_singles_b + n_doubles_aaaa + n_doubles_ab + n_doubles_bbbb

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaaa = f_b
    f_aaaa = s_aaaa + n_doubles_aaaa
    s_abab = f_aaaa
    f_ab = s_abab + n_doubles_ab
    s_bbbb = f_ab
    f_bbbb = s_bbbb + n_doubles_bbbb

    U = adc.U.T
    nroots = U.shape[0]

    spin = np.array([])
    trace_a = np.array([])
    trace_b = np.array([])

    for r in range(U.shape[0]):
        
        Y_aa = U[r][:f_a].reshape(nocc_a, nvir_a)
        Y_bb = U[r][f_a:f_b].reshape(nocc_b, nvir_b)


        Y_abab = U[r][s_abab:f_ab].reshape(nocc_a, nocc_b, nvir_a, nvir_b)

        Y_vv_u_a = np.zeros((int((nocc_a * (nocc_a - 1))/2),nvir_a, nvir_a))
        Y_vv_u_a[:,ab_ind_a[0],ab_ind_a[1]]= U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2)) 
        Y_vv_u_a[:,ab_ind_a[1],ab_ind_a[0]]= -U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2))
        Y_aaaa = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        Y_aaaa[ij_ind_a[0],ij_ind_a[1],:,:]= Y_vv_u_a
        Y_aaaa[ij_ind_a[1],ij_ind_a[0],:,:]= -Y_vv_u_a

        del Y_vv_u_a

        Y_vv_u_b = np.zeros((int((nocc_b * (nocc_b - 1))/2),nvir_b, nvir_b))
        Y_vv_u_b[:,ab_ind_b[0],ab_ind_b[1]]= U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_vv_u_b[:,ab_ind_b[1],ab_ind_b[0]]= -U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_bbbb = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        Y_bbbb[ij_ind_b[0],ij_ind_b[1],:,:]= Y_vv_u_b
        Y_bbbb[ij_ind_b[1],ij_ind_b[0],:,:]= -Y_vv_u_b

        del Y_vv_u_b

# OPDM ADC(2)
        temp_a[:nocc_a,:nocc_a] =- 1/2 * lib.einsum('Ijab,Ljab->IL', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,La->IL', Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ijab,Ljab->IL', Y_abab, Y_abab, optimize = True)

        temp_a[occ_list_a,occ_list_a] += 1/4 * lib.einsum('jkab,jkab->', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('ja,ja->', Y_aa, Y_aa, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('jkab,jkab->', Y_abab, Y_abab, optimize = True)
        temp_a[occ_list_a,occ_list_a] += 1/4 * lib.einsum('jkab,jkab->', Y_bbbb, Y_bbbb, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('ja,ja->', Y_bb, Y_bb, optimize = True)

        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,Ljab,jb->IL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,Ljab,jb->IL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('La,Ijab,jb->IL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('La,Ijab,jb->IL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/4 * lib.einsum('Ia,ja,jkbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ia,ja,jb,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ia,ja,jkbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,kjcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/4 * lib.einsum('La,ja,jkbc,Ikbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('La,ja,jb,Ib->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('La,ja,jkbc,Ikbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,kjcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ib,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ia,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('ja,ka,Ijbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,kb,Ijac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,kb,Ijac,Lkcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ib,Lb->IL', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikca,Lkcb->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,kb,Ijca,Lkbc->IL', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,ka,Ijbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,kb,Ijca,Lkcb->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        
        temp_a[nocc_a:,nocc_a:]  = lib.einsum('iA,iC->AC', Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ijAb,ijCb->AC', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ijAb,ijCb->AC', Y_abab, Y_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iA,ijCb,jb->AC', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iA,ijCb,jb->AC', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iC,ijAb,jb->AC', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iC,ijAb,jb->AC', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iA,ib,jb,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/4 * lib.einsum('iA,ib,jkbd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iA,ib,jkbd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,kjdb->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iC,ib,jb,jA->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/4 * lib.einsum('iC,ib,jkbd,jkAd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iC,ib,jkbd,jkAd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,kjdb->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jA,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ib,ib,jkAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jkAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('ib,id,jkAb,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,iA,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,ikAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,ikAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,jd,ikAb,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jd,ikAb,kjCd->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jd,ikCb,kjAd->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jA,jC->AC', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ib,ib,jkAd,jkCd->AC', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jkAd,jkCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,id,jkAb,jkCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,kiAd,kjCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,jd,kiAb,kjCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        temp_IC  = lib.einsum('IjCa,ja->IC', Y_aaaa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,ja->IC', Y_abab, Y_bb, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t1_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_IC -= lib.einsum('jC,ja,Ia->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_aaaa, Y_abab, Y_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('Ijab,jkab,kC->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ia,ja,jC->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_abab, Y_bbbb, Y_bb, optimize = True)
        temp_IC += lib.einsum('IjCa,kjba,kb->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ijab,kjab,kC->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('jkCa,jkab,Ib->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('jkCa,jkba,Ib->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t2_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_IC -= lib.einsum('jC,ja,Ia->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ia,ja,jC->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,ja,jkCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,ja,jkCb,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('Ia,jb,ka,jkCb->IC', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,jb,ka,kjCb->IC', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ja,Ikab,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ja,Ikab,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('jC,ka,jb,Ikab->IC', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ka,jb,Ikba->IC', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,jb,IkCa,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,ka,IjCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_bb, Y_bb, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,jb,IkCa,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,ka,IjCb,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        
        temp_b[:nocc_b,:nocc_b] =- 1/2 * lib.einsum('ijab,ljab->il', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,la->il', Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('jiab,jlab->il', Y_abab, Y_abab, optimize = True)


        temp_b[occ_list_b, occ_list_b] += 1/4 * lib.einsum('jkab,jkab->', Y_aaaa, Y_aaaa, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('ja,ja->', Y_aa, Y_aa, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('jkab,jkab->', Y_abab, Y_abab, optimize = True)
        temp_b[occ_list_b, occ_list_b] += 1/4 * lib.einsum('jkab,jkab->', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('ja,ja->', Y_bb, Y_bb, optimize = True)

        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,ljab,jb->il', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,jlba,jb->il', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('la,ijab,jb->il', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('la,jiba,jb->il', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,lkac,jkbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,klca,jkbc->il', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/4 * lib.einsum('ia,ja,jkbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ia,ja,jb,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ia,ja,kjbc,klbc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,lkac,jkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,klca,kjcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,ikac,jkbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,kica,jkbc->il', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/4 * lib.einsum('la,ja,jkbc,ikbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('la,ja,jb,ib->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('la,ja,kjbc,kibc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,ikac,jkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,kica,kjcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ja,ja,ikbc,lkbc->il', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,ib,lb->il', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,kibc,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,kiac,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,ka,jibc,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,kb,jiac,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ja,ja,ikbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,ib,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,kibc,klbc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,ia,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,ikac,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,kica,klcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,kb,ijac,klbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,kb,ljac,kibc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ja,ka,ijbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,kb,ijac,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        temp_b[nocc_b:,nocc_b:]  = lib.einsum('ijba,ijbc->ac', Y_abab, Y_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,ic->ac', Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ijab,ijcb->ac', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,ijcb,jb->ac', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,jibc,jb->ac', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ic,ijab,jb->ac', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ic,jiba,jb->ac', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,ja,jc->ac', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,jkda,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ib,ib,jkad,jkcd->ac', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,id,jkba,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ikda,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,jd,ikba,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ia,ib,jb,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ia,ib,jkdb,jkdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/4 * lib.einsum('ia,ib,jkbd,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,ikcd,jkbd->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,kidc,jkbd->ac', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,ikcd,jkbd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,kidc,kjdb->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ic,ib,jb,ja->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ic,ib,jkdb,jkda->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/4 * lib.einsum('ic,ib,jkbd,jkad->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,ikad,jkbd->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,kida,jkbd->ac', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,ikad,jkbd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,kida,kjdb->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,ja,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,jkda,jkdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ib,ib,jkad,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ib,id,jkab,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jd,ikab,jkdc->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jd,ikcb,jkda->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ia,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ikad,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,kida,kjdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,jd,ikab,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        
        temp_ic  = lib.einsum('ijca,ja->ic', Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('jiac,ja->ic', Y_abab, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t1_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jc,ja,ia->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ijca,jkab,kb->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ijca,kjba,kb->ic', t1_ccee_bbbb, Y_abab, Y_aa, optimize = True)
        temp_ic += 1/2 * lib.einsum('ijab,jkab,kc->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('ia,ja,jc->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic += lib.einsum('jiac,jkab,kb->ic', t1_ccee_abab, Y_aaaa, Y_aa, optimize = True)
        temp_ic += lib.einsum('jiac,jkab,kb->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jiab,jkab,kc->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jkac,jkab,ib->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('jkca,jkab,ib->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t2_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jc,ja,ia->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('ia,ja,jc->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,jb,ka,jkbc->ic', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,ja,jkcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,ja,kjbc,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic += 1/2 * lib.einsum('ia,jb,ka,jkcb->ic', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,ikcb,kb->ic', Y_aa, Y_aa, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,kibc,kb->ic', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,jb,ka,kibc->ic', Y_aa, Y_aa, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,ka,jibc,kb->ic', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ja,ikab,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ja,kiba,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ka,jb,kiab->ic', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_ic += 1/2 * lib.einsum('jc,ka,jb,ikab->ic', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,ikcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,kibc,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,jb,ikca,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,ka,ijcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        
            
# OPDM ADC(2)-X
        if (method == "adc(2)-x"):

            temp_IC += 1/4 * lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC += lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_IC += 1/4 * lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_bbbb, Y_bbbb, optimize = True)
            temp_IC -= 1/2 * lib.einsum('jC,jkab,Ikab->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC -= lib.einsum('jC,jkab,Ikab->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_IC -= 1/2 * lib.einsum('Ia,jkab,jkCb->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC -= lib.einsum('Ia,jkab,jkCb->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            
            temp_ic += 1/4 * lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_aaaa, Y_aaaa, optimize = True)
            temp_ic += lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic += 1/4 * lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ic -= 1/2 * lib.einsum('jc,jkab,ikab->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ic -= lib.einsum('jc,kjab,kiab->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic -= lib.einsum('ia,jkba,jkbc->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic -= 1/2 * lib.einsum('ia,jkab,jkcb->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)

        temp_a[:nocc_a,nocc_a:] = temp_IC
        temp_a[nocc_a:,:nocc_a] = temp_IC.T

        temp_b[:nocc_b,nocc_b:] = temp_ic
        temp_b[nocc_b:,:nocc_b] = temp_ic.T
        

      #  norm = np.linalg.norm(temp_a - temp_a.transpose(1,0))
      #  print("total OPDM_a singles norm for Hermiticity",norm)
      #  print("opdm trace_a",np.einsum('pp',temp_a))
      #  
      #  
      #  norm = np.linalg.norm(temp_b - temp_b.transpose(1,0))
      #  print("total OPDM_b singles norm for Hermiticity",norm)
      #  print("opdm trace_b",np.einsum('pp',temp_b))

        IjKl  = lib.einsum('Ijab,Klab->IjKl', Y_abab, Y_abab, optimize = True)
    
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jmab,lmab->jl', Y_bbbb, Y_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ja,la->jl', Y_bb, Y_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('mjab,mlab->jl', Y_abab, Y_abab, optimize = True)
        
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Imab,Kmab->IK', Y_aaaa, Y_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ia,Ka->IK', Y_aa, Y_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Imab,Kmab->IK', Y_abab, Y_abab, optimize = True)
    
        IjKl += lib.einsum('Ia,Klab,jb->IjKl', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IjKl += lib.einsum('Ka,Ijab,lb->IjKl', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IjKl += lib.einsum('ja,Klba,Ib->IjKl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IjKl += lib.einsum('la,Ijba,Kb->IjKl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
    
        diag_sc = 1/4 * lib.einsum('mnab,mnab->', Y_aaaa, Y_aaaa, optimize = True)
        diag_sc += lib.einsum('ma,ma->', Y_aa, Y_aa, optimize = True)
        diag_sc += lib.einsum('mnab,mnab->', Y_abab, Y_abab, optimize = True)
        diag_sc += 1/4 * lib.einsum('mnab,mnab->', Y_bbbb, Y_bbbb, optimize = True)
        diag_sc += lib.einsum('ma,ma->', Y_bb, Y_bb, optimize = True)
        
        block_sc = np.zeros((nocc_a*nocc_b,nocc_a*nocc_b))
        np.fill_diagonal(block_sc, diag_sc)
        IjKl += block_sc.reshape(nocc_a,nocc_b,nocc_a,nocc_b)
        
        
        IjKl += 1/2 * lib.einsum('Ia,Ka,jmbc,lmbc->IjKl', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl += lib.einsum('Ia,Ka,jb,lb->IjKl', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl += lib.einsum('Ia,Ka,mjbc,mlbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('Ia,Kb,mlac,mjbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ia,lb,Ka,jb->IjKl', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IjKl += lib.einsum('Ia,lb,Kmac,mjcb->IjKl', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ia,lb,Kmac,jmbc->IjKl', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl -= lib.einsum('Ia,ma,mjbc,Klbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ia,mb,Klac,mjbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('Ia,mb,Klac,jmbc->IjKl', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl += lib.einsum('Ka,jb,Ia,lb->IjKl', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IjKl += lib.einsum('Ka,jb,Imac,mlcb->IjKl', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ka,jb,Imac,lmbc->IjKl', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl -= lib.einsum('Ka,ma,mlbc,Ijbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('Ka,mb,Ijac,mlbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('Ka,mb,Ijac,lmbc->IjKl', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl += 1/2 * lib.einsum('ja,la,Imbc,Kmbc->IjKl', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl += lib.einsum('ja,la,Ib,Kb->IjKl', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl += lib.einsum('ja,la,Imbc,Kmbc->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ja,lb,Kmca,Imcb->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ja,mb,Klca,Imbc->IjKl', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl -= lib.einsum('ja,ma,Imbc,Klbc->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ja,mb,Klca,Imcb->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('la,mb,Ijca,Kmbc->IjKl', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl -= lib.einsum('la,ma,Kmbc,Ijbc->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('la,mb,Ijca,Kmcb->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ma,ma,Ijbc,Klbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ma,mb,Ijac,Klbc->IjKl', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ma,ma,Ijbc,Klbc->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ma,mb,Ijca,Klcb->IjKl', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ja,lmab,mb->jl',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ja,mlba,mb->jl',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('la,jmab,mb->jl',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('la,mjba,mb->jl',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ia,Kmab,mb->IK', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ia,Kmab,mb->IK', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ka,Imab,mb->IK', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('Ka,Imab,mb->IK', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,mb,lnac,mnbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,mb,nlca,mnbc->jl',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/4 * lib.einsum('ja,ma,mnbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,ma,mb,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,ma,nmbc,nlbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,mb,lnac,mnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,mb,nlca,nmcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('la,mb,jnac,mnbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('la,mb,njca,mnbc->jl',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/4 * lib.einsum('la,ma,mnbc,jnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('la,ma,mb,jb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('la,ma,nmbc,njbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('la,mb,jnac,mnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('la,mb,njca,nmcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ma,ma,jnbc,lnbc->jl',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,ma,jb,lb->jl',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,ma,njbc,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,mb,njac,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,na,mjbc,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,nb,mjac,nlbc->jl',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ma,ma,jnbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,ma,jb,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,ma,njbc,nlbc->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,mb,ja,lb->jl',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,mb,jnac,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,mb,njca,nlcb->jl',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,nb,jmac,nlbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += lib.einsum('ma,nb,lmac,njbc->jl',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ma,na,jmbc,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl[occ_list_a,:,occ_list_a,:] -= lib.einsum('ma,nb,jmac,lnbc->jl',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        IjKl[:,occ_list_b,:,occ_list_b] += 1/4 * lib.einsum('Ia,ma,mnbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('Ia,ma,mb,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('Ia,ma,mnbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ia,mb,Knac,nmcb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ia,mb,Knac,mnbc->IK',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/4 * lib.einsum('Ka,ma,mnbc,Inbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('Ka,ma,mb,Ib->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('Ka,ma,mnbc,Inbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ka,mb,Inac,nmcb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('Ka,mb,Inac,mnbc->IK',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,ma,Ib,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,mb,Ia,Kb->IK',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,mb,Inac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,mb,Inac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += 1/2 * lib.einsum('ma,na,Imbc,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,nb,Imac,Knbc->IK',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,nb,Imac,Kncb->IK',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= 1/2 * lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,ma,Ib,Kb->IK',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,ma,Inbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,mb,Inca,Kncb->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,nb,Imca,Knbc->IK',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] += lib.einsum('ma,na,Imbc,Knbc->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl[:,occ_list_b,:,occ_list_b] -= lib.einsum('ma,nb,Imca,Kncb->IK',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            
            
        AbCd  = lib.einsum('ijAb,ijCd->AbCd', Y_abab, Y_abab, optimize = True)
        AbCd += lib.einsum('iA,ijCd,jb->AbCd', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        AbCd += lib.einsum('iC,ijAb,jd->AbCd', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        AbCd += lib.einsum('ib,jiCd,jA->AbCd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        AbCd += lib.einsum('id,jiAb,jC->AbCd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        AbCd += lib.einsum('iA,iC,jb,jd->AbCd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        AbCd += lib.einsum('iA,iC,jkeb,jked->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += 1/2 * lib.einsum('iA,iC,jkbe,jkde->AbCd', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('iA,ie,jkeb,jkCd->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('iA,jC,iked,jkeb->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iA,je,ikCd,jkeb->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iA,jd,iC,jb->AbCd', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        AbCd += lib.einsum('iA,jd,ikCe,kjeb->AbCd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iA,jd,ikCe,jkbe->AbCd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('iA,je,ikCd,jkbe->AbCd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('iC,ie,jked,jkAb->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iC,je,ikAb,jked->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iC,jb,iA,jd->AbCd', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        AbCd += lib.einsum('iC,jb,ikAe,kjed->AbCd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('iC,jb,ikAe,jkde->AbCd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('iC,je,ikAb,jkde->AbCd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd += lib.einsum('ie,ie,jkAb,jkCd->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ie,je,ikAb,jkCd->AbCd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ib,id,jA,jC->AbCd', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        AbCd += 1/2 * lib.einsum('ib,id,jkAe,jkCe->AbCd', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AbCd += lib.einsum('ib,id,jkAe,jkCe->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ib,ie,jkAe,jkCd->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ib,je,kiCd,jkAe->AbCd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        AbCd -= lib.einsum('ib,jd,kiCe,kjAe->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ib,je,kiCd,kjAe->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('id,ie,jkCe,jkAb->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('id,je,kiAb,jkCe->AbCd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        AbCd += lib.einsum('id,je,kiAb,kjCe->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ie,ie,jkAb,jkCd->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ie,je,kiAb,kjCd->AbCd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            
            
            
        IjCd  = lib.einsum('IC,jkda,ka->IjCd', t1_ce_aa, Y_bbbb, Y_bb, optimize = True)
        IjCd += lib.einsum('IC,kjad,ka->IjCd', t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IjCd -= lib.einsum('kC,ka,Ijad->IjCd', t1_ce_aa, Y_aa, Y_abab, optimize = True)
        IjCd += lib.einsum('IjCd,ka,ka->IjCd', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IjCd,ka,ka->IjCd', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('IjCa,ka,kd->IjCd', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('Ijad,ka,kC->IjCd', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IkCa,ka,jd->IjCd', t1_ccee_aaaa, Y_aa, Y_bb, optimize = True)
        IjCd -= lib.einsum('Ia,ka,kjCd->IjCd', t1_ce_aa, Y_aa, Y_abab, optimize = True)
        IjCd -= lib.einsum('IkCd,ka,ja->IjCd', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('IkCa,ka,jd->IjCd', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('jd,IkCa,ka->IjCd', t1_ce_bb, Y_aaaa, Y_aa, optimize = True)
        IjCd += lib.einsum('jd,IkCa,ka->IjCd', t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IjCd -= lib.einsum('kd,ka,IjCa->IjCd', t1_ce_bb, Y_bb, Y_abab, optimize = True)
        IjCd += lib.einsum('jkda,ka,IC->IjCd', t1_ccee_bbbb, Y_bb, Y_aa, optimize = True)
        IjCd -= lib.einsum('ja,ka,IkCd->IjCd', t1_ce_bb, Y_bb, Y_abab, optimize = True)
        IjCd -= lib.einsum('kjCd,ka,Ia->IjCd', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('kjad,ka,IC->IjCd', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IjCd,ka,ka->IjCd', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IjCd,ka,ka->IjCd', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('IjCa,ka,kd->IjCd', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('Ijad,ka,kC->IjCd', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('IkCa,ka,jd->IjCd', t2_ccee_aaaa, Y_aa, Y_bb, optimize = True)
        IjCd -= lib.einsum('IkCd,ka,ja->IjCd', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('IkCa,ka,jd->IjCd', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('jkda,ka,IC->IjCd', t2_ccee_bbbb, Y_bb, Y_aa, optimize = True)
        IjCd -= lib.einsum('kjCd,ka,Ia->IjCd', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('kjad,ka,IC->IjCd', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('IC,ka,kd,ja->IjCd', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjCd -= lib.einsum('Ia,ka,kC,jd->IjCd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd -= lib.einsum('jd,ka,kC,Ia->IjCd', Y_bb, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjCd -= lib.einsum('ja,ka,kd,IC->IjCd', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IjCd -= lib.einsum('kC,ka,Ia,jd->IjCd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd += lib.einsum('ka,ka,IC,jd->IjCd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd -= lib.einsum('kd,ka,ja,IC->IjCd', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IjCd += lib.einsum('ka,ka,IC,jd->IjCd', Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
            
            
            
            
        IaKd =- lib.einsum('Ijbd,Kjba->IaKd', Y_abab, Y_abab, optimize = True)
    
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jlba,jlbd->ad', Y_abab, Y_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('ja,jd->ad', Y_bb, Y_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jlab,jldb->ad', Y_bbbb, Y_bbbb, optimize = True)
        
        IaKd -= lib.einsum('Ib,Kjba,jd->IaKd', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IaKd -= lib.einsum('Kb,Ijbd,ja->IaKd', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        IaKd -= lib.einsum('ja,Ijbd,Kb->IaKd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IaKd -= lib.einsum('jd,Kjba,Ib->IaKd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IaKd -= lib.einsum('Ib,Kb,ja,jd->IaKd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd -= lib.einsum('Ib,Kb,jlca,jlcd->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 1/2 * lib.einsum('Ib,Kb,jlac,jldc->IaKd', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('Ib,Kc,jlba,jlcd->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('Ib,jb,jlcd,Klca->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Ib,jc,Klba,jlcd->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Ib,ja,Kb,jd->IaKd', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IaKd -= lib.einsum('Ib,ja,Klbc,ljcd->IaKd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Ib,ja,Klbc,jldc->IaKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('Ib,jc,Klba,jldc->IaKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('Kb,jb,jlca,Ilcd->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Kb,jc,Ilbd,jlca->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Kb,jd,Ib,ja->IaKd', Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IaKd -= lib.einsum('Kb,jd,Ilbc,ljca->IaKd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('Kb,jd,Ilbc,jlac->IaKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('Kb,jc,Ilbd,jlac->IaKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaKd -= lib.einsum('jb,jb,Ilcd,Klca->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('jb,jc,Ilbd,Klca->IaKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 1/2 * lib.einsum('ja,jd,Ilbc,Klbc->IaKd', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IaKd -= lib.einsum('ja,jd,Ib,Kb->IaKd', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IaKd -= lib.einsum('ja,jd,Ilbc,Klbc->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ja,jb,Klcb,Ilcd->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ja,lb,Ijcd,Klbc->IaKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd += lib.einsum('ja,ld,Ijbc,Klbc->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ja,lb,Ijcd,Klcb->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('jd,jb,Ilcb,Klca->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('jd,lb,Kjca,Ilbc->IaKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd -= lib.einsum('jd,lb,Kjca,Ilcb->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('jb,jb,Ilcd,Klca->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('jb,lb,Ijcd,Klca->IaKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('ja,jldb,lb->ad',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('ja,ljbd,lb->ad',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jd,jlab,lb->ad',  t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jd,ljba,lb->ad',  t1_ce_bb, Y_abab, Y_aa, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,la,ld->ad',  Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,lmca,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jb,jb,lmac,lmdc->ad',  Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,jc,lmba,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,jmca,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,lc,jmba,lmcd->ad',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,jb,lb,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('ja,jb,lmcb,lmcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('ja,jb,lmbc,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,lb,jmdc,lmbc->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,lb,mjcd,lmbc->ad',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,lb,jmdc,lmbc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('ja,lb,mjcd,mlcb->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jd,jb,lb,la->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jd,jb,lmcb,lmca->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/4 * lib.einsum('jd,jb,lmbc,lmac->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jd,lb,jmac,lmbc->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jd,lb,mjca,lmbc->ad',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jd,lb,jmac,lmbc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jd,lb,mjca,mlcb->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,la,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,jb,lmca,lmcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += 1/2 * lib.einsum('jb,jb,lmac,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= 1/2 * lib.einsum('jb,jc,lmab,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lc,jmab,lmcd->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lc,jmdb,lmca->ad',  Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,ja,ld->ad',  Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,jmac,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] -= lib.einsum('jb,lb,mjca,mlcd->ad',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd[occ_list_a,:,occ_list_a,:] += lib.einsum('jb,lc,jmab,lmdc->ad',  Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            
            
        IakD =- lib.einsum('ID,ka->IakD', Y_aa, Y_bb, optimize = True)
        IakD -= lib.einsum('IjDb,jkba->IakD', Y_aaaa, Y_abab, optimize = True)
        IakD -= lib.einsum('IjDb,kjab->IakD', Y_abab, Y_bbbb, optimize = True)
        IakD -= lib.einsum('ID,kjab,jb->IakD', t1_ce_aa, Y_bbbb, Y_bb, optimize = True)
        IakD -= lib.einsum('ID,jkba,jb->IakD', t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IakD += lib.einsum('jD,jkba,Ib->IakD', t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IakD += lib.einsum('Ib,jkba,jD->IakD', t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IakD -= lib.einsum('ka,IjDb,jb->IakD', t1_ce_bb, Y_aaaa, Y_aa, optimize = True)
        IakD -= lib.einsum('ka,IjDb,jb->IakD', t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IakD += lib.einsum('ja,IjDb,kb->IakD', t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IakD += lib.einsum('kb,IjDb,ja->IakD', t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IakD += 1/2 * lib.einsum('ID,kb,jb,ja->IakD', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IakD += 1/2 * lib.einsum('ID,kb,jlcb,jlca->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += 1/4 * lib.einsum('ID,kb,jlbc,jlac->IakD', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IakD -= 1/2 * lib.einsum('ID,jb,jlbc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ID,jb,jlbc,klac->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD += 1/4 * lib.einsum('ID,ja,jlbc,klbc->IakD', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IakD += 1/2 * lib.einsum('ID,ja,jb,kb->IakD', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IakD += 1/2 * lib.einsum('ID,ja,ljbc,lkbc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ID,jb,jlbc,klac->IakD', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IakD -= 1/2 * lib.einsum('ID,jb,ljcb,lkca->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += 1/2 * lib.einsum('Ib,ka,jb,jD->IakD', Y_aa, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IakD += 1/4 * lib.einsum('Ib,ka,jlbc,jlDc->IakD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IakD += 1/2 * lib.einsum('Ib,ka,jlbc,jlDc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('Ib,kc,jlba,jlDc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('Ib,jb,jD,ka->IakD', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IakD += lib.einsum('Ib,jb,jlDc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('Ib,jb,jlDc,klac->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('Ib,jc,lkba,jlDc->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD -= lib.einsum('Ib,ja,lkbc,ljDc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('Ib,jc,lkba,ljDc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += 1/4 * lib.einsum('ka,jD,jlbc,Ilbc->IakD', Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IakD += 1/2 * lib.einsum('ka,jD,jb,Ib->IakD', Y_bb, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IakD += 1/2 * lib.einsum('ka,jD,jlbc,Ilbc->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ka,jb,jlbc,IlDc->IakD', Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IakD -= 1/2 * lib.einsum('ka,jb,jlbc,IlDc->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ka,jb,jlbc,IlDc->IakD', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IakD -= 1/2 * lib.einsum('ka,jb,ljcb,IlDc->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD -= lib.einsum('kb,jD,Ilcb,jlca->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('kb,jc,IlDb,jlca->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('kb,jb,ja,ID->IakD', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IakD += lib.einsum('kb,jb,jlac,IlDc->IakD', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('kb,jb,ljca,IlDc->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD -= lib.einsum('kb,jc,IlDb,jlac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD += lib.einsum('jD,jb,Ib,ka->IakD', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IakD += lib.einsum('jD,jb,Ilbc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('jD,jb,Ilbc,klac->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('jD,lb,jkca,Ilbc->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD -= lib.einsum('jD,la,jkbc,Ilbc->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('jD,lb,jkca,Ilcb->IakD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,jb,ID,ka->IakD', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IakD -= lib.einsum('jb,jb,IlDc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,jb,IlDc,klac->IakD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD += lib.einsum('jb,jc,IlDb,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('jb,lb,IjDc,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,lc,IjDb,lkca->IakD', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,lc,IjDb,klac->IakD', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_bbbb, optimize = True)
        IakD += lib.einsum('ja,jb,kb,ID->IakD', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IakD += lib.einsum('ja,jb,klbc,IlDc->IakD', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('ja,jb,lkcb,IlDc->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IakD += lib.einsum('ja,lb,IjDc,lkbc->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('ja,lb,IjDc,klbc->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('jb,jb,ID,ka->IakD', Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IakD -= lib.einsum('jb,jb,IlDc,lkca->IakD', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IakD -= lib.einsum('jb,jb,IlDc,klac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD += lib.einsum('jb,jc,IlDb,klac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('jb,lc,IjDb,lkca->IakD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IakD += lib.einsum('jb,lb,IjDc,klac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IakD -= lib.einsum('jb,lc,IjDb,klac->IakD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        
        
        
        
        
        iAkD =- lib.einsum('jiDb,jkAb->iAkD', Y_abab, Y_abab, optimize = True)
        
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jA,jD->AD', Y_aa, Y_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jlAb,jlDb->AD', Y_aaaa, Y_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jlAb,jlDb->AD', Y_abab, Y_abab, optimize = True)
        
        iAkD -= lib.einsum('jA,jiDb,kb->iAkD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        iAkD -= lib.einsum('jD,jkAb,ib->iAkD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        iAkD -= lib.einsum('ib,jkAb,jD->iAkD', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        iAkD -= lib.einsum('kb,jiDb,jA->iAkD', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        iAkD -= lib.einsum('ib,kb,jA,jD->iAkD', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD -= 1/2 * lib.einsum('ib,kb,jlAc,jlDc->iAkD', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD -= lib.einsum('ib,kb,jlAc,jlDc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('ib,kc,jlAb,jlDc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('ib,jA,kb,jD->iAkD', Y_bb, Y_aa, t1_ce_bb, t1_ce_aa, optimize = True)
        iAkD -= lib.einsum('ib,jA,klbc,jlDc->iAkD', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('ib,jA,lkcb,jlDc->iAkD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        iAkD += lib.einsum('ib,jc,lkAb,jlDc->iAkD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        iAkD += lib.einsum('ib,jb,ljDc,lkAc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('ib,jc,lkAb,ljDc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('kb,jD,ib,jA->iAkD', Y_bb, Y_aa, t1_ce_bb, t1_ce_aa, optimize = True)
        iAkD -= lib.einsum('kb,jD,ilbc,jlAc->iAkD', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('kb,jD,licb,jlAc->iAkD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        iAkD += lib.einsum('kb,jc,liDb,jlAc->iAkD', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        iAkD += lib.einsum('kb,jb,ljAc,liDc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('kb,jc,liDb,ljAc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= 1/2 * lib.einsum('jA,jD,ilbc,klbc->iAkD', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        iAkD -= lib.einsum('jA,jD,ib,kb->iAkD', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        iAkD -= lib.einsum('jA,jD,libc,lkbc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jA,jb,lkbc,liDc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jA,lD,jibc,lkbc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('jA,lb,jiDc,lkbc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jA,lb,jiDc,klbc->iAkD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        iAkD += lib.einsum('jD,jb,libc,lkAc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('jD,lb,jkAc,libc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jD,lb,jkAc,ilbc->iAkD', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        iAkD -= lib.einsum('jb,jb,liDc,lkAc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jb,lb,jiDc,lkAc->iAkD', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD -= lib.einsum('jb,jb,liDc,lkAc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD += lib.einsum('jb,jc,liDb,lkAc->iAkD', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jA,jlDb,lb->AD',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jA,jlDb,lb->AD',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jD,jlAb,lb->AD',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jD,jlAb,lb->AD',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jA,jb,lb,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('jA,jb,lmbc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jA,jb,lmbc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jA,lb,jmDc,mlcb->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jA,lb,jmDc,lmbc->AD',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jD,jb,lb,lA->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/4 * lib.einsum('jD,jb,lmbc,lmAc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jD,jb,lmbc,lmAc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jD,lb,jmAc,mlcb->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jD,lb,jmAc,lmbc->AD',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lA,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= 1/2 * lib.einsum('jb,jc,lmAb,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,jA,lD->AD',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,jmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,jmAc,lmDc->AD',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,lc,jmAb,lmDc->AD',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lc,jmAb,mlDc->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lc,jmDb,mlAc->AD',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lA,lD->AD',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += 1/2 * lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,jb,lmAc,lmDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,jc,lmAb,lmDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] -= lib.einsum('jb,lb,mjAc,mlDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        iAkD[occ_list_b,:,occ_list_b,:] += lib.einsum('jb,lc,mjAb,mlDc->AD',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        if (method == "adc(2)-x"):
    
    
            IjCd += 1/4 * lib.einsum('IjCd,klab,klab->IjCd', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd += lib.einsum('IjCd,klab,klab->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += 1/4 * lib.einsum('IjCd,klab,klab->IjCd', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd -= lib.einsum('IjCa,klba,klbd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= 1/2 * lib.einsum('IjCa,klab,kldb->IjCd', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd -= 1/2 * lib.einsum('Ijad,klab,klCb->IjCd', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd -= lib.einsum('Ijad,klab,klCb->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('Ijab,klab,klCd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('IkCa,klab,ljbd->IjCd', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
            IjCd += lib.einsum('IkCa,klab,jldb->IjCd', t1_ccee_aaaa, Y_abab, Y_bbbb, optimize = True)
            IjCd += 1/2 * lib.einsum('Ikab,klab,ljCd->IjCd', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
            IjCd -= 1/2 * lib.einsum('IkCd,klab,jlab->IjCd', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd -= lib.einsum('IkCd,lkab,ljab->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('IkCa,klab,jldb->IjCd', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd += lib.einsum('IkCa,lkba,ljbd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('Ikad,lkab,ljCb->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('Ikab,lkab,ljCd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('jkda,klab,IlCb->IjCd', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)
            IjCd += lib.einsum('jkda,lkba,IlCb->IjCd', t1_ccee_bbbb, Y_abab, Y_aaaa, optimize = True)
            IjCd += 1/2 * lib.einsum('jkab,klab,IlCd->IjCd', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)
            IjCd -= 1/2 * lib.einsum('kjCd,klab,Ilab->IjCd', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd -= lib.einsum('kjCd,klab,Ilab->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('kjCa,klba,Ilbd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('kjad,klab,IlCb->IjCd', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd += lib.einsum('kjad,klab,IlCb->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('kjab,klab,IlCd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += 1/2 * lib.einsum('klCa,klab,Ijbd->IjCd', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
            IjCd += lib.einsum('klCd,klab,Ijab->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('klCa,klba,Ijbd->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('klad,klab,IjCb->IjCd', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += 1/2 * lib.einsum('klda,klab,IjCb->IjCd', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)
    
        IjlK = -IjKl.transpose(0,1,3,2)
        AbdC = -AbCd.transpose(0,1,3,2)
        IjdC = -IjCd.transpose(0,1,3,2)
        IadK = -IaKd.transpose(0,1,3,2)
        AikD = -iAkD.transpose(1,0,2,3)
    
        #################there are 6 unique abba density rtpdms for spin contamination############################
        #abba
        #oooo
        #vvvv
        #oovv
        #ovvo
        #ovov
    
        na = lib.einsum('pp',temp_a)
        nb = lib.einsum('pp',temp_b)
    
        spin_c = 0.25*((na - nb)**2) + 0.5*(na + nb) + lib.einsum('pr,qs,pqrs->',delta[:nocc_a,:nocc_b], delta[:nocc_a,:nocc_b].T, IjlK, optimize = True)
        spin_c += lib.einsum('pr,qs,pqrs->', delta[nocc_a:,nocc_b:], delta[nocc_a:,nocc_b:].T, AbdC, optimize = True)
        spin_c += 2*lib.einsum('pr,qs,pqrs->', delta[:nocc_a,nocc_b:], delta[nocc_a:,:nocc_b].T, IjdC, optimize = True)
        spin_c += lib.einsum('pr,qs,pqrs->', delta[:nocc_a,nocc_b:], delta[:nocc_a,nocc_b:].T,IadK, optimize = True)
        spin_c += 2*lib.einsum('pr,qs,pqrs->', delta[:nocc_a,:nocc_b], delta[nocc_a:,nocc_b:].T,IakD, optimize = True)
        spin_c += lib.einsum('pr,qs,pqrs->', delta[nocc_a:,:nocc_b], delta[nocc_a:,:nocc_b].T,AikD, optimize = True)
        
        spin = np.append(spin,spin_c)
        trace_a = np.append(trace_a,na)
        trace_b = np.append(trace_b,nb)
    
    return spin, (trace_a, trace_b)

#@profile
def get_spin_contamination(adc):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    dm_a = adc.dm_a.copy()
    dm_b = adc.dm_b.copy()

    t1 = adc.t1
    t2 = adc.t2
    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b


    vir_list_a = range(nvir_a)
    vir_list_b = range(nvir_b)
    occ_list_a = range(nocc_a)
    occ_list_b = range(nocc_b)


    ovlp = adc._scf.get_ovlp(adc._scf.mol).copy()
 
    delta = np.dot(adc.mo_coeff[0].transpose(),np.dot(ovlp,adc.mo_coeff[1]))
    S_oc_ab = delta[:nocc_a,:nocc_b]
    S_vir_ab = delta[nocc_a:,nocc_b:]
    S_ov_ab = delta[:nocc_a,nocc_b:]
    S_vo_ab = delta[nocc_a:,:nocc_b]

    if adc.f_ov is None:
        f_ov_a = np.zeros((nocc_a, nvir_a))
        f_ov_b = np.zeros((nocc_b, nvir_b))
        t1_ce_aa = np.zeros((nocc_a, nvir_a))
        t1_ce_bb = np.zeros((nocc_b, nvir_b))
    else:
        f_ov_a, f_ov_b = adc.f_ov
        t1_ce_aa = t1[2][0][:]
        t1_ce_bb = t1[2][1][:]

    t2_ce_aa = t1[0][0][:]
    t2_ce_bb = t1[0][1][:]

    t1_ccee_aaaa = t2[0][0][:]
    t1_ccee_abab = t2[0][1][:]
    t1_ccee_bbbb = t2[0][2][:]

    t2_ccee_aaaa = t2[1][0][:]
    t2_ccee_abab = t2[1][1][:]
    t2_ccee_bbbb = t2[1][2][:]

    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    temp_a = np.zeros((nmo_a,nmo_a))
    temp_b = np.zeros((nmo_b,nmo_b))

    temp_2a = np.zeros((nmo_a,nmo_a,nmo_a,nmo_a))
    temp_2b = np.zeros((nmo_b,nmo_b,nmo_b,nmo_b))

    temp_2 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))

    n_singles_a = nocc_a * nvir_a
    n_singles_b = nocc_b * nvir_b
    n_doubles_aaaa = nocc_a * (nocc_a - 1) * nvir_a * (nvir_a -1) // 4
    n_doubles_ab = nocc_a * nocc_b * nvir_a * nvir_b
    n_doubles_bbbb = nocc_b * (nocc_b - 1) * nvir_b * (nvir_b -1) // 4

    dim = n_singles_a + n_singles_b + n_doubles_aaaa + n_doubles_ab + n_doubles_bbbb

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaaa = f_b
    f_aaaa = s_aaaa + n_doubles_aaaa
    s_abab = f_aaaa
    f_ab = s_abab + n_doubles_ab
    s_bbbb = f_ab
    f_bbbb = s_bbbb + n_doubles_bbbb

    U = adc.U.T
    nroots = U.shape[0]

    spin = np.array([])
    trace_a = np.array([])
    trace_b = np.array([])

    for r in range(U.shape[0]):
        
        Y_aa = U[r][:f_a].reshape(nocc_a, nvir_a)
        Y_bb = U[r][f_a:f_b].reshape(nocc_b, nvir_b)


        Y_abab = U[r][s_abab:f_ab].reshape(nocc_a, nocc_b, nvir_a, nvir_b)

        Y_vv_u_a = np.zeros((int((nocc_a * (nocc_a - 1))/2),nvir_a, nvir_a))
        Y_vv_u_a[:,ab_ind_a[0],ab_ind_a[1]]= U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2)) 
        Y_vv_u_a[:,ab_ind_a[1],ab_ind_a[0]]= -U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2))
        Y_aaaa = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        Y_aaaa[ij_ind_a[0],ij_ind_a[1],:,:]= Y_vv_u_a
        Y_aaaa[ij_ind_a[1],ij_ind_a[0],:,:]= -Y_vv_u_a

        del Y_vv_u_a

        Y_vv_u_b = np.zeros((int((nocc_b * (nocc_b - 1))/2),nvir_b, nvir_b))
        Y_vv_u_b[:,ab_ind_b[0],ab_ind_b[1]]= U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_vv_u_b[:,ab_ind_b[1],ab_ind_b[0]]= -U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_bbbb = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        Y_bbbb[ij_ind_b[0],ij_ind_b[1],:,:]= Y_vv_u_b
        Y_bbbb[ij_ind_b[1],ij_ind_b[0],:,:]= -Y_vv_u_b

        del Y_vv_u_b

# OPDM ADC(2)
        temp_a[:nocc_a,:nocc_a] =- 1/2 * lib.einsum('Ijab,Ljab->IL', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,La->IL', Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ijab,Ljab->IL', Y_abab, Y_abab, optimize = True)

        temp_a[occ_list_a,occ_list_a] += 1/4 * lib.einsum('jkab,jkab->', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('ja,ja->', Y_aa, Y_aa, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('jkab,jkab->', Y_abab, Y_abab, optimize = True)
        temp_a[occ_list_a,occ_list_a] += 1/4 * lib.einsum('jkab,jkab->', Y_bbbb, Y_bbbb, optimize = True)
        temp_a[occ_list_a,occ_list_a] += lib.einsum('ja,ja->', Y_bb, Y_bb, optimize = True)

        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,Ljab,jb->IL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,Ljab,jb->IL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('La,Ijab,jb->IL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('La,Ijab,jb->IL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/4 * lib.einsum('Ia,ja,jkbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ia,ja,jb,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('Ia,ja,jkbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,kjcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,jb,Lkac,jkbc->IL', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/4 * lib.einsum('La,ja,jkbc,Ikbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('La,ja,jb,Ib->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('La,ja,jkbc,Ikbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,kjcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('La,jb,Ikac,jkbc->IL', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ib,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ia,Lb->IL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += 1/2 * lib.einsum('ja,ka,Ijbc,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,kb,Ijac,Lkbc->IL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,kb,Ijac,Lkcb->IL', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ib,Lb->IL', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,ja,Ikbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,jb,Ikca,Lkcb->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,kb,Ijca,Lkbc->IL', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,:nocc_a] += lib.einsum('ja,ka,Ijbc,Lkbc->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,:nocc_a] -= lib.einsum('ja,kb,Ijca,Lkcb->IL', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        
        temp_a[nocc_a:,nocc_a:]  = lib.einsum('iA,iC->AC', Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ijAb,ijCb->AC', Y_aaaa, Y_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ijAb,ijCb->AC', Y_abab, Y_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iA,ijCb,jb->AC', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iA,ijCb,jb->AC', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iC,ijAb,jb->AC', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('iC,ijAb,jb->AC', t1_ce_aa, Y_abab, Y_bb, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iA,ib,jb,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/4 * lib.einsum('iA,ib,jkbd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iA,ib,jkbd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,kjdb->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iA,jb,ikCd,jkbd->AC', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iC,ib,jb,jA->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/4 * lib.einsum('iC,ib,jkbd,jkAd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('iC,ib,jkbd,jkAd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,kjdb->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,jb,ikAd,jkbd->AC', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jA,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ib,ib,jkAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jkAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= 1/2 * lib.einsum('ib,id,jkAb,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,iA,jC->AC', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,ikAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,ikAd,jkCd->AC', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,jd,ikAb,jkCd->AC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jd,ikAb,kjCd->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jd,ikCb,kjAd->AC', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jA,jC->AC', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('ib,ib,jkAd,jkCd->AC', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,ib,jkAd,jkCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,id,jkAb,jkCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] -= lib.einsum('ib,jb,kiAd,kjCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,nocc_a:] += lib.einsum('ib,jd,kiAb,kjCd->AC', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        
        temp_IC  = lib.einsum('IjCa,ja->IC', Y_aaaa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,ja->IC', Y_abab, Y_bb, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t1_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_IC -= lib.einsum('jC,ja,Ia->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_aaaa, Y_abab, Y_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('Ijab,jkab,kC->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ia,ja,jC->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_abab, Y_bbbb, Y_bb, optimize = True)
        temp_IC += lib.einsum('IjCa,kjba,kb->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ijab,kjab,kC->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('jkCa,jkab,Ib->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('jkCa,jkba,Ib->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC += lib.einsum('IC,ja,ja->IC', t2_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_IC -= lib.einsum('jC,ja,Ia->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC -= lib.einsum('Ia,ja,jC->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,ja,jkCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,ja,jkCb,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('Ia,jb,ka,jkCb->IC', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('Ia,jb,ka,kjCb->IC', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ja,Ikab,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ja,Ikab,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC += 1/2 * lib.einsum('jC,ka,jb,Ikab->IC', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('jC,ka,jb,Ikba->IC', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,jb,IkCa,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,ka,IjCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_bb, Y_bb, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_IC += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,jb,IkCa,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_IC -= 1/2 * lib.einsum('ja,ka,IjCb,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        
        temp_b[:nocc_b,:nocc_b] =- 1/2 * lib.einsum('ijab,ljab->il', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,la->il', Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('jiab,jlab->il', Y_abab, Y_abab, optimize = True)


        temp_b[occ_list_b, occ_list_b] += 1/4 * lib.einsum('jkab,jkab->', Y_aaaa, Y_aaaa, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('ja,ja->', Y_aa, Y_aa, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('jkab,jkab->', Y_abab, Y_abab, optimize = True)
        temp_b[occ_list_b, occ_list_b] += 1/4 * lib.einsum('jkab,jkab->', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[occ_list_b, occ_list_b] += lib.einsum('ja,ja->', Y_bb, Y_bb, optimize = True)

        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,ljab,jb->il', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ia,jlba,jb->il', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('la,ijab,jb->il', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('la,jiba,jb->il', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,lkac,jkbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,klca,jkbc->il', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/4 * lib.einsum('ia,ja,jkbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ia,ja,jb,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ia,ja,kjbc,klbc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,lkac,jkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,klca,kjcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,ikac,jkbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,kica,jkbc->il', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/4 * lib.einsum('la,ja,jkbc,ikbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('la,ja,jb,ib->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('la,ja,kjbc,kibc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,ikac,jkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('la,jb,kica,kjcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ja,ja,ikbc,lkbc->il', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,ib,lb->il', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,kibc,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,kiac,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,ka,jibc,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,kb,jiac,klbc->il', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ja,ja,ikbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,ib,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ja,kibc,klbc->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,ia,lb->il', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,ikac,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,jb,kica,klcb->il', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,kb,ijac,klbc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += lib.einsum('ja,kb,ljac,kibc->il', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,:nocc_b] += 1/2 * lib.einsum('ja,ka,ijbc,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,:nocc_b] -= lib.einsum('ja,kb,ijac,lkbc->il', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        temp_b[nocc_b:,nocc_b:]  = lib.einsum('ijba,ijbc->ac', Y_abab, Y_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,ic->ac', Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ijab,ijcb->ac', Y_bbbb, Y_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,ijcb,jb->ac', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ia,jibc,jb->ac', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ic,ijab,jb->ac', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ic,jiba,jb->ac', t1_ce_bb, Y_abab, Y_aa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,ja,jc->ac', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,jkda,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ib,ib,jkad,jkcd->ac', Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,id,jkba,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ikda,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,jd,ikba,jkdc->ac', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ia,ib,jb,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ia,ib,jkdb,jkdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/4 * lib.einsum('ia,ib,jkbd,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,ikcd,jkbd->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,kidc,jkbd->ac', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,ikcd,jkbd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ia,jb,kidc,kjdb->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ic,ib,jb,ja->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ic,ib,jkdb,jkda->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/4 * lib.einsum('ic,ib,jkbd,jkad->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,ikad,jkbd->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,kida,jkbd->ac', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,ikad,jkbd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,kida,kjdb->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,ja,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,ib,jkda,jkdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ib,ib,jkad,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ib,id,jkab,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jd,ikab,jkdc->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jd,ikcb,jkda->ac', Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ia,jc->ac', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,ikad,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jb,kida,kjdc->ac', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,nocc_b:] += lib.einsum('ib,jd,ikab,jkcd->ac', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        
        
        temp_ic  = lib.einsum('ijca,ja->ic', Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('jiac,ja->ic', Y_abab, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t1_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jc,ja,ia->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ijca,jkab,kb->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ijca,kjba,kb->ic', t1_ccee_bbbb, Y_abab, Y_aa, optimize = True)
        temp_ic += 1/2 * lib.einsum('ijab,jkab,kc->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('ia,ja,jc->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic += lib.einsum('jiac,jkab,kb->ic', t1_ccee_abab, Y_aaaa, Y_aa, optimize = True)
        temp_ic += lib.einsum('jiac,jkab,kb->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jiab,jkab,kc->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jkac,jkab,ib->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('jkca,jkab,ib->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t2_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_ic += lib.einsum('ic,ja,ja->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('jc,ja,ia->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= lib.einsum('ia,ja,jc->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,jb,ka,jkbc->ic', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,ja,jkcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ia,ja,kjbc,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic += 1/2 * lib.einsum('ia,jb,ka,jkcb->ic', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,ikcb,kb->ic', Y_aa, Y_aa, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,kibc,kb->ic', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,jb,ka,kibc->ic', Y_aa, Y_aa, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,ka,jibc,kb->ic', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ja,ikab,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ja,kiba,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('jc,ka,jb,kiab->ic', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_ic += 1/2 * lib.einsum('jc,ka,jb,ikab->ic', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,ikcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic += 1/2 * lib.einsum('ja,ja,kibc,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,jb,ikca,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_ic -= 1/2 * lib.einsum('ja,ka,ijcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        
            
# OPDM ADC(2)-X
        if (method == "adc(2)-x"):

            temp_IC += 1/4 * lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC += lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_IC += 1/4 * lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_bbbb, Y_bbbb, optimize = True)
            temp_IC -= 1/2 * lib.einsum('jC,jkab,Ikab->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC -= lib.einsum('jC,jkab,Ikab->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_IC -= 1/2 * lib.einsum('Ia,jkab,jkCb->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_IC -= lib.einsum('Ia,jkab,jkCb->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            
            temp_ic += 1/4 * lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_aaaa, Y_aaaa, optimize = True)
            temp_ic += lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic += 1/4 * lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ic -= 1/2 * lib.einsum('jc,jkab,ikab->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_ic -= lib.einsum('jc,kjab,kiab->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic -= lib.einsum('ia,jkba,jkbc->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_ic -= 1/2 * lib.einsum('ia,jkab,jkcb->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)

        temp_a[:nocc_a,nocc_a:] = temp_IC
        temp_a[nocc_a:,:nocc_a] = temp_IC.T

        temp_b[:nocc_b,nocc_b:] = temp_ic
        temp_b[nocc_b:,:nocc_b] = temp_ic.T
        

    #    norm = np.linalg.norm(temp_a - temp_a.transpose(1,0))
    #    print("total OPDM_a singles norm for Hermiticity",norm)
    #    print("opdm trace_a",np.einsum('pp',temp_a))
    #    
    #    
    #    norm = np.linalg.norm(temp_b - temp_b.transpose(1,0))
    #    print("total OPDM_b singles norm for Hermiticity",norm)
    #    print("opdm trace_b",np.einsum('pp',temp_b))

        IjKl =- 1/4 * lib.einsum('ij,ij,klab,klab', S_oc_ab, S_oc_ab, Y_aaaa, Y_aaaa, optimize = True)
        IjKl -= lib.einsum('ij,ij,ka,ka', S_oc_ab, S_oc_ab, Y_aa, Y_aa, optimize = True)
        IjKl -= lib.einsum('ij,ij,klab,klab', S_oc_ab, S_oc_ab, Y_abab, Y_abab, optimize = True)
        IjKl -= 1/4 * lib.einsum('ij,ij,klab,klab', S_oc_ab, S_oc_ab, Y_bbbb, Y_bbbb, optimize = True)
        IjKl -= lib.einsum('ij,ij,ka,ka', S_oc_ab, S_oc_ab, Y_bb, Y_bb, optimize = True)
        IjKl += 1/2 * lib.einsum('ij,ik,jlab,klab', S_oc_ab, S_oc_ab, Y_bbbb, Y_bbbb, optimize = True)
        IjKl += lib.einsum('ij,ik,ja,ka', S_oc_ab, S_oc_ab, Y_bb, Y_bb, optimize = True)
        IjKl += lib.einsum('ij,ik,ljab,lkab', S_oc_ab, S_oc_ab, Y_abab, Y_abab, optimize = True)
        IjKl += 1/2 * lib.einsum('ij,kj,ilab,klab', S_oc_ab, S_oc_ab, Y_aaaa, Y_aaaa, optimize = True)
        IjKl += lib.einsum('ij,kj,ia,ka', S_oc_ab, S_oc_ab, Y_aa, Y_aa, optimize = True)
        IjKl += lib.einsum('ij,kj,ilab,klab', S_oc_ab, S_oc_ab, Y_abab, Y_abab, optimize = True)
        IjKl -= lib.einsum('ij,kl,ilab,kjab', S_oc_ab, S_oc_ab, Y_abab, Y_abab, optimize = True)
        IjKl += 2 * lib.einsum('ia,ij,kj,klab,lb', t1_ce_aa, S_oc_ab, S_oc_ab, Y_aaaa, Y_aa, optimize = True)
        IjKl += 2 * lib.einsum('ia,ij,kj,klab,lb', t1_ce_aa, S_oc_ab, S_oc_ab, Y_abab, Y_bb, optimize = True)
        IjKl -= 2 * lib.einsum('ia,ij,kl,kjab,lb', t1_ce_aa, S_oc_ab, S_oc_ab, Y_abab, Y_bb, optimize = True)
        IjKl += 2 * lib.einsum('ia,ji,jk,klab,lb', t1_ce_bb, S_oc_ab, S_oc_ab, Y_bbbb, Y_bb, optimize = True)
        IjKl += 2 * lib.einsum('ia,ji,jk,lkba,lb', t1_ce_bb, S_oc_ab, S_oc_ab, Y_abab, Y_aa, optimize = True)
        IjKl -= 2 * lib.einsum('ia,ji,kl,jlba,kb', t1_ce_bb, S_oc_ab, S_oc_ab, Y_abab, Y_aa, optimize = True)
        IjKl += lib.einsum('ij,ik,ja,lb,kmac,lmbc', S_oc_ab, S_oc_ab, Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,ik,ja,lb,mkca,lmbc', S_oc_ab, S_oc_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl -= 1/2 * lib.einsum('ij,ik,ja,la,kmbc,lmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl -= lib.einsum('ij,ik,ja,la,kb,lb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl -= lib.einsum('ij,ik,ja,la,mkbc,mlbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,ik,ja,lb,kmac,lmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl += lib.einsum('ij,ik,ja,lb,mkca,mlcb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += 1/2 * lib.einsum('ij,ik,la,la,jmbc,kmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl += lib.einsum('ij,ik,la,la,jb,kb', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl += lib.einsum('ij,ik,la,la,mjbc,mkbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ij,ik,la,lb,mjac,mkbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ij,ik,la,ma,ljbc,mkbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,ik,la,mb,ljac,mkbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += 1/2 * lib.einsum('ij,ik,la,la,jmbc,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl += lib.einsum('ij,ik,la,la,jb,kb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl += lib.einsum('ij,ik,la,la,mjbc,mkbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ij,ik,la,lb,ja,kb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl -= lib.einsum('ij,ik,la,lb,jmac,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl -= lib.einsum('ij,ik,la,lb,mjca,mkcb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= 2 * lib.einsum('ij,ik,la,mb,jlac,mkbc', S_oc_ab, S_oc_ab, Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IjKl -= 1/2 * lib.einsum('ij,ik,la,ma,jlbc,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl += lib.einsum('ij,ik,la,mb,jlac,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl -= 1/2 * lib.einsum('ij,kj,ia,la,kmbc,lmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl -= lib.einsum('ij,kj,ia,la,kb,lb', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl -= lib.einsum('ij,kj,ia,la,kmbc,lmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,kj,ia,lb,kmac,lmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl += lib.einsum('ij,kj,ia,lb,kmac,lmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,kj,ia,lb,kmac,mlcb', S_oc_ab, S_oc_ab, Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,kj,ia,lb,kmac,lmbc', S_oc_ab, S_oc_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl += 1/2 * lib.einsum('ij,kj,la,la,imbc,kmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl += lib.einsum('ij,kj,la,la,ib,kb', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl += lib.einsum('ij,kj,la,la,imbc,kmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ij,kj,la,lb,ia,kb', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl -= lib.einsum('ij,kj,la,lb,imac,kmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl -= lib.einsum('ij,kj,la,lb,imac,kmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= 1/2 * lib.einsum('ij,kj,la,ma,ilbc,kmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl += lib.einsum('ij,kj,la,mb,ilac,kmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl -= 2 * lib.einsum('ij,kj,la,mb,ilac,kmcb', S_oc_ab, S_oc_ab, Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl += 1/2 * lib.einsum('ij,kj,la,la,imbc,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl += lib.einsum('ij,kj,la,la,ib,kb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl += lib.einsum('ij,kj,la,la,imbc,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ij,kj,la,lb,imca,kmcb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ij,kj,la,ma,ilbc,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,kj,la,mb,ilca,kmcb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= 2 * lib.einsum('ij,kl,ia,jb,ka,lb', S_oc_ab, S_oc_ab, Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IjKl -= 2 * lib.einsum('ij,kl,ia,jb,kmac,mlcb', S_oc_ab, S_oc_ab, Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IjKl -= 2 * lib.einsum('ij,kl,ia,jb,kmac,lmbc', S_oc_ab, S_oc_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl -= 1/2 * lib.einsum('ij,kl,ia,ka,jmbc,lmbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IjKl -= lib.einsum('ij,kl,ia,ka,jb,lb', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IjKl -= lib.einsum('ij,kl,ia,ka,mjbc,mlbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,kl,ia,kb,mjac,mlbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += 2 * lib.einsum('ij,kl,ia,ma,kjbc,mlbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= 2 * lib.einsum('ij,kl,ia,mb,kjac,mlbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += 2 * lib.einsum('ij,kl,ia,mb,kjac,lmbc', S_oc_ab, S_oc_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IjKl -= 1/2 * lib.einsum('ij,kl,ja,la,imbc,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IjKl -= lib.einsum('ij,kl,ja,la,ib,kb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IjKl -= lib.einsum('ij,kl,ja,la,imbc,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,kl,ja,lb,imca,kmcb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += 2 * lib.einsum('ij,kl,ja,mb,ilca,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IjKl += 2 * lib.einsum('ij,kl,ja,ma,ilbc,kmbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= 2 * lib.einsum('ij,kl,ja,mb,ilca,kmcb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ij,kl,ma,ma,ilbc,kjbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,kl,ma,mb,ilac,kjbc', S_oc_ab, S_oc_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl -= lib.einsum('ij,kl,ma,ma,ilbc,kjbc', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IjKl += lib.einsum('ij,kl,ma,mb,ilca,kjcb', S_oc_ab, S_oc_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)

        AbCd =- lib.einsum('ab,cd,ijad,ijcb', S_vir_ab, S_vir_ab, Y_abab, Y_abab, optimize = True)
        AbCd -= 2 * lib.einsum('ia,ab,cd,ijcb,jd', t1_ce_aa, S_vir_ab, S_vir_ab, Y_abab, Y_bb, optimize = True)
        AbCd -= 2 * lib.einsum('ia,ba,cd,jibd,jc', t1_ce_bb, S_vir_ab, S_vir_ab, Y_abab, Y_aa, optimize = True)
        AbCd -= lib.einsum('ab,cd,ia,ic,jb,jd', S_vir_ab, S_vir_ab, Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        AbCd -= lib.einsum('ab,cd,ia,ic,jkeb,jked', S_vir_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= 1/2 * lib.einsum('ab,cd,ia,ic,jkbe,jkde', S_vir_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        AbCd += 2 * lib.einsum('ab,cd,ia,ie,jkcb,jked', S_vir_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ab,cd,ia,jc,ikeb,jked', S_vir_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= 2 * lib.einsum('ab,cd,ia,je,ikcb,jked', S_vir_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= 2 * lib.einsum('ab,cd,ia,jb,ic,jd', S_vir_ab, S_vir_ab, Y_aa, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        AbCd -= 2 * lib.einsum('ab,cd,ia,jb,ikce,kjed', S_vir_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AbCd -= 2 * lib.einsum('ab,cd,ia,jb,ikce,jkde', S_vir_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd += 2 * lib.einsum('ab,cd,ia,je,ikcb,jkde', S_vir_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AbCd -= lib.einsum('ab,cd,ie,ie,jkad,jkcb', S_vir_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ab,cd,ie,je,ikad,jkcb', S_vir_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ab,cd,ib,id,ja,jc', S_vir_ab, S_vir_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        AbCd -= 1/2 * lib.einsum('ab,cd,ib,id,jkae,jkce', S_vir_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AbCd -= lib.einsum('ab,cd,ib,id,jkae,jkce', S_vir_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += 2 * lib.einsum('ab,cd,ib,ie,jkad,jkce', S_vir_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += 2 * lib.einsum('ab,cd,ib,je,kiad,jkce', S_vir_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        AbCd += lib.einsum('ab,cd,ib,jd,kiae,kjce', S_vir_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= 2 * lib.einsum('ab,cd,ib,je,kiad,kjce', S_vir_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd -= lib.einsum('ab,cd,ie,ie,jkad,jkcb', S_vir_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AbCd += lib.einsum('ab,cd,ie,je,kiad,kjcb', S_vir_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)


        IjCd =- lib.einsum('ia,bj,ib,jkac,kc', S_ov_ab, S_vo_ab, t1_ce_aa, Y_bbbb, Y_bb, optimize = True)
        IjCd -= lib.einsum('ia,bj,ib,kjca,kc', S_ov_ab, S_vo_ab, t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,ijba,kc,kc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,ijba,kc,kc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,ijbc,ka,kc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,ijca,kb,kc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,ikbc,ja,kc', S_ov_ab, S_vo_ab, t1_ccee_aaaa, Y_bb, Y_aa, optimize = True)
        IjCd += lib.einsum('ia,bj,ic,kjba,kc', S_ov_ab, S_vo_ab, t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IjCd += lib.einsum('ia,bj,ikba,jc,kc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('ia,bj,ikbc,ja,kc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('ia,bj,ja,ikbc,kc', S_ov_ab, S_vo_ab, t1_ce_bb, Y_aaaa, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,ja,ikbc,kc', S_ov_ab, S_vo_ab, t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,ka,ijbc,kc', S_ov_ab, S_vo_ab, t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,kb,ijca,kc', S_ov_ab, S_vo_ab, t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,jkac,ib,kc', S_ov_ab, S_vo_ab, t1_ccee_bbbb, Y_aa, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,jc,ikba,kc', S_ov_ab, S_vo_ab, t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,kjba,ic,kc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,kjca,ib,kc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,ijba,kc,kc', S_ov_ab, S_vo_ab, t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,ijba,kc,kc', S_ov_ab, S_vo_ab, t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,ijbc,ka,kc', S_ov_ab, S_vo_ab, t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,ijca,kb,kc', S_ov_ab, S_vo_ab, t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,ikbc,ja,kc', S_ov_ab, S_vo_ab, t2_ccee_aaaa, Y_bb, Y_aa, optimize = True)
        IjCd += lib.einsum('ia,bj,ikba,jc,kc', S_ov_ab, S_vo_ab, t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('ia,bj,ikbc,ja,kc', S_ov_ab, S_vo_ab, t2_ccee_abab, Y_bb, Y_bb, optimize = True)
        IjCd -= lib.einsum('ia,bj,jkac,ib,kc', S_ov_ab, S_vo_ab, t2_ccee_bbbb, Y_aa, Y_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,kjba,ic,kc', S_ov_ab, S_vo_ab, t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd -= lib.einsum('ia,bj,kjca,ib,kc', S_ov_ab, S_vo_ab, t2_ccee_abab, Y_aa, Y_aa, optimize = True)
        IjCd += lib.einsum('ia,bj,ib,kc,ka,jc', S_ov_ab, S_vo_ab, Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,ic,kc,ja,kb', S_ov_ab, S_vo_ab, Y_aa, Y_aa, t1_ce_bb, t1_ce_aa, optimize = True)
        IjCd += lib.einsum('ia,bj,ja,kc,ic,kb', S_ov_ab, S_vo_ab, Y_bb, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IjCd += lib.einsum('ia,bj,jc,kc,ib,ka', S_ov_ab, S_vo_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,kb,kc,ic,ja', S_ov_ab, S_vo_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd -= lib.einsum('ia,bj,kc,kc,ib,ja', S_ov_ab, S_vo_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd += lib.einsum('ia,bj,ka,kc,ib,jc', S_ov_ab, S_vo_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IjCd -= lib.einsum('ia,bj,kc,kc,ib,ja', S_ov_ab, S_vo_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)

        IaKd =- lib.einsum('ia,ib,jkca,jkcb', S_ov_ab, S_ov_ab, Y_abab, Y_abab, optimize = True)
        IaKd -= lib.einsum('ia,ib,ja,jb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, optimize = True)
        IaKd -= 1/2 * lib.einsum('ia,ib,jkac,jkbc', S_ov_ab, S_ov_ab, Y_bbbb, Y_bbbb, optimize = True)
        IaKd += lib.einsum('ia,jb,ikca,jkcb', S_ov_ab, S_ov_ab, Y_abab, Y_abab, optimize = True)
        IaKd += 2 * lib.einsum('ia,ib,jc,jkac,kb', t1_ce_aa, S_ov_ab, S_ov_ab, Y_abab, Y_bb, optimize = True)
        IaKd -= 2 * lib.einsum('ia,ja,jb,ikbc,kc', t1_ce_bb, S_ov_ab, S_ov_ab, Y_bbbb, Y_bb, optimize = True)
        IaKd -= 2 * lib.einsum('ia,ja,jb,kicb,kc', t1_ce_bb, S_ov_ab, S_ov_ab, Y_abab, Y_aa, optimize = True)
        IaKd += 2 * lib.einsum('ia,ja,kb,kicb,jc', t1_ce_bb, S_ov_ab, S_ov_ab, Y_abab, Y_aa, optimize = True)
        IaKd -= lib.einsum('ia,ib,jc,jc,ka,kb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd -= lib.einsum('ia,ib,jc,jc,klda,kldb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 1/2 * lib.einsum('ia,ib,jc,jc,klad,klbd', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('ia,ib,jc,jd,klca,kldb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ia,ib,jc,kc,jlda,kldb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ia,ib,jc,kd,jlca,kldb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ia,ib,ja,jc,kb,kc', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd += lib.einsum('ia,ib,ja,jc,kldb,kldc', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += 1/2 * lib.einsum('ia,ib,ja,jc,klbd,klcd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd -= lib.einsum('ia,ib,ja,kc,jlbd,klcd', S_ov_ab, S_ov_ab, Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ia,ib,ja,kc,ljdb,klcd', S_ov_ab, S_ov_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd -= lib.einsum('ia,ib,ja,kc,jlbd,klcd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd -= lib.einsum('ia,ib,ja,kc,ljdb,lkdc', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ia,ib,jc,jc,ka,kb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd -= lib.einsum('ia,ib,jc,jc,klda,kldb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 1/2 * lib.einsum('ia,ib,jc,jc,klad,klbd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd += 1/2 * lib.einsum('ia,ib,jc,jd,klac,klbd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd += 2 * lib.einsum('ia,ib,jc,kd,jlac,kldb', S_ov_ab, S_ov_ab, Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ia,ib,jc,kc,ja,kb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd += lib.einsum('ia,ib,jc,kc,jlad,klbd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('ia,ib,jc,kc,ljda,lkdb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ia,ib,jc,kd,jlac,klbd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd += lib.einsum('ia,jb,ic,jc,ka,kb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        IaKd += lib.einsum('ia,jb,ic,jc,klda,kldb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += 1/2 * lib.einsum('ia,jb,ic,jc,klad,klbd', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaKd -= lib.einsum('ia,jb,ic,jd,klda,klcb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 2 * lib.einsum('ia,jb,ic,kc,klda,jldb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += 2 * lib.einsum('ia,jb,ic,kd,klda,jlcb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += 2 * lib.einsum('ia,jb,ic,kb,ka,jc', S_ov_ab, S_ov_ab, Y_aa, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
        IaKd += 2 * lib.einsum('ia,jb,ic,kb,klad,jlcd', S_ov_ab, S_ov_ab, Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd += 2 * lib.einsum('ia,jb,ic,kb,lkda,jlcd', S_ov_ab, S_ov_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaKd -= 2 * lib.einsum('ia,jb,ic,kd,klad,jlcb', S_ov_ab, S_ov_ab, Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ia,jb,kc,kc,ilda,jldb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ia,jb,kc,kd,ilca,jldb', S_ov_ab, S_ov_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += 1/2 * lib.einsum('ia,jb,ka,kb,ilcd,jlcd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IaKd += lib.einsum('ia,jb,ka,kb,ic,jc', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IaKd += lib.einsum('ia,jb,ka,kb,ilcd,jlcd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 2 * lib.einsum('ia,jb,ka,kc,ildc,jldb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= 2 * lib.einsum('ia,jb,ka,lc,ilcd,jkdb', S_ov_ab, S_ov_ab, Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ia,jb,ka,lb,ilcd,jkcd', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += 2 * lib.einsum('ia,jb,ka,lc,ildc,jkdb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd += lib.einsum('ia,jb,kc,kc,ilda,jldb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaKd -= lib.einsum('ia,jb,kc,lc,ikda,jldb', S_ov_ab, S_ov_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)

        AiDk =- lib.einsum('ai,bi,ja,jb', S_vo_ab, S_vo_ab, Y_aa, Y_aa, optimize = True)
        AiDk -= 1/2 * lib.einsum('ai,bi,jkac,jkbc', S_vo_ab, S_vo_ab, Y_aaaa, Y_aaaa, optimize = True)
        AiDk -= lib.einsum('ai,bi,jkac,jkbc', S_vo_ab, S_vo_ab, Y_abab, Y_abab, optimize = True)
        AiDk += lib.einsum('ai,bj,kiac,kjbc', S_vo_ab, S_vo_ab, Y_abab, Y_abab, optimize = True)
        AiDk -= 2 * lib.einsum('ia,aj,bj,ikbc,kc', t1_ce_aa, S_vo_ab, S_vo_ab, Y_aaaa, Y_aa, optimize = True)
        AiDk -= 2 * lib.einsum('ia,aj,bj,ikbc,kc', t1_ce_aa, S_vo_ab, S_vo_ab, Y_abab, Y_bb, optimize = True)
        AiDk += 2 * lib.einsum('ia,aj,bk,ikbc,jc', t1_ce_aa, S_vo_ab, S_vo_ab, Y_abab, Y_bb, optimize = True)
        AiDk += 2 * lib.einsum('ia,bi,cj,kjca,kb', t1_ce_bb, S_vo_ab, S_vo_ab, Y_abab, Y_aa, optimize = True)
        AiDk += lib.einsum('ai,bi,ja,jc,kb,kc', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        AiDk += 1/2 * lib.einsum('ai,bi,ja,jc,klbd,klcd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AiDk += lib.einsum('ai,bi,ja,jc,klbd,klcd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bi,ja,kc,jlbd,klcd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AiDk -= lib.einsum('ai,bi,ja,kc,jlbd,klcd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bi,ja,kc,jlbd,lkdc', S_vo_ab, S_vo_ab, Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bi,ja,kc,jlbd,klcd', S_vo_ab, S_vo_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AiDk -= lib.einsum('ai,bi,jc,jc,ka,kb', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        AiDk -= 1/2 * lib.einsum('ai,bi,jc,jc,klad,klbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AiDk -= lib.einsum('ai,bi,jc,jc,klad,klbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += 1/2 * lib.einsum('ai,bi,jc,jd,klac,klbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AiDk += lib.einsum('ai,bi,jc,kc,ja,kb', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        AiDk += lib.einsum('ai,bi,jc,kc,jlad,klbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AiDk += lib.einsum('ai,bi,jc,kc,jlad,klbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bi,jc,kd,jlac,klbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AiDk += 2 * lib.einsum('ai,bi,jc,kd,jlac,lkbd', S_vo_ab, S_vo_ab, Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bi,jc,jc,ka,kb', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        AiDk -= 1/2 * lib.einsum('ai,bi,jc,jc,klad,klbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AiDk -= lib.einsum('ai,bi,jc,jc,klad,klbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += lib.einsum('ai,bi,jc,jd,klac,klbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += lib.einsum('ai,bi,jc,kc,ljad,lkbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bi,jc,kd,ljac,lkbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += lib.einsum('ai,bj,ic,jc,ka,kb', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        AiDk += 1/2 * lib.einsum('ai,bj,ic,jc,klad,klbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        AiDk += lib.einsum('ai,bj,ic,jc,klad,klbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bj,ic,jd,klad,klbc', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += 2 * lib.einsum('ai,bj,ic,kb,ka,jc', S_vo_ab, S_vo_ab, Y_bb, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        AiDk += 2 * lib.einsum('ai,bj,ic,kb,klad,ljdc', S_vo_ab, S_vo_ab, Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AiDk += 2 * lib.einsum('ai,bj,ic,kb,klad,jlcd', S_vo_ab, S_vo_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        AiDk -= 2 * lib.einsum('ai,bj,ic,kd,klad,ljbc', S_vo_ab, S_vo_ab, Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        AiDk -= 2 * lib.einsum('ai,bj,ic,kc,lkad,ljbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += 2 * lib.einsum('ai,bj,ic,kd,lkad,ljbc', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += 1/2 * lib.einsum('ai,bj,ka,kb,ilcd,jlcd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        AiDk += lib.einsum('ai,bj,ka,kb,ic,jc', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
        AiDk += lib.einsum('ai,bj,ka,kb,licd,ljcd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= 2 * lib.einsum('ai,bj,ka,kc,licd,ljbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bj,ka,lb,licd,kjcd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += 2 * lib.einsum('ai,bj,ka,lc,licd,kjbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= 2 * lib.einsum('ai,bj,ka,lc,ilcd,kjbd', S_vo_ab, S_vo_ab, Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        AiDk += lib.einsum('ai,bj,kc,kc,liad,ljbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bj,kc,lc,kiad,ljbd', S_vo_ab, S_vo_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk += lib.einsum('ai,bj,kc,kc,liad,ljbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        AiDk -= lib.einsum('ai,bj,kc,kd,liac,ljbd', S_vo_ab, S_vo_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)

        IaDk =- lib.einsum('ij,ab,ia,jb', S_oc_ab, S_vir_ab, Y_aa, Y_bb, optimize = True)
        IaDk -= lib.einsum('ij,ab,ikac,kjcb', S_oc_ab, S_vir_ab, Y_aaaa, Y_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,ikac,jkbc', S_oc_ab, S_vir_ab, Y_abab, Y_bbbb, optimize = True)
        IaDk -= lib.einsum('ij,ab,ia,jkbc,kc', S_oc_ab, S_vir_ab, t1_ce_aa, Y_bbbb, Y_bb, optimize = True)
        IaDk -= lib.einsum('ij,ab,ia,kjcb,kc', S_oc_ab, S_vir_ab, t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IaDk += lib.einsum('ij,ab,ic,kjcb,ka', S_oc_ab, S_vir_ab, t1_ce_aa, Y_abab, Y_aa, optimize = True)
        IaDk -= lib.einsum('ij,ab,jb,ikac,kc', S_oc_ab, S_vir_ab, t1_ce_bb, Y_aaaa, Y_aa, optimize = True)
        IaDk -= lib.einsum('ij,ab,jb,ikac,kc', S_oc_ab, S_vir_ab, t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IaDk += lib.einsum('ij,ab,jc,ikac,kb', S_oc_ab, S_vir_ab, t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IaDk += lib.einsum('ij,ab,ka,ic,kjcb', S_oc_ab, S_vir_ab, t1_ce_aa, Y_aa, Y_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,kb,ikac,jc', S_oc_ab, S_vir_ab, t1_ce_bb, Y_abab, Y_bb, optimize = True)
        IaDk += 1/2 * lib.einsum('ij,ab,ia,jc,kb,kc', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaDk += 1/2 * lib.einsum('ij,ab,ia,jc,kldb,kldc', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += 1/4 * lib.einsum('ij,ab,ia,jc,klbd,klcd', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaDk -= 1/2 * lib.einsum('ij,ab,ia,kc,jlbd,klcd', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaDk -= 1/2 * lib.einsum('ij,ab,ia,kc,ljdb,klcd', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaDk += 1/4 * lib.einsum('ij,ab,ia,kb,jlcd,klcd', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaDk += 1/2 * lib.einsum('ij,ab,ia,kb,jc,kc', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
        IaDk += 1/2 * lib.einsum('ij,ab,ia,kb,ljcd,lkcd', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk -= 1/2 * lib.einsum('ij,ab,ia,kc,jlbd,klcd', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
        IaDk -= 1/2 * lib.einsum('ij,ab,ia,kc,ljdb,lkdc', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += 1/2 * lib.einsum('ij,ab,ic,jb,ka,kc', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
        IaDk += 1/4 * lib.einsum('ij,ab,ic,jb,klad,klcd', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IaDk += 1/2 * lib.einsum('ij,ab,ic,jb,klad,klcd', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,ic,jd,klad,klcb', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,ic,kc,jb,ka', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ce_bb, t1_ce_aa, optimize = True)
        IaDk += lib.einsum('ij,ab,ic,kc,jlbd,klad', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,ic,kc,ljdb,klad', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaDk -= lib.einsum('ij,ab,ic,kd,ljcb,klad', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
        IaDk -= lib.einsum('ij,ab,ic,kb,ljcd,lkad', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,ic,kd,ljcb,lkad', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += 1/4 * lib.einsum('ij,ab,jb,ka,ilcd,klcd', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IaDk += 1/2 * lib.einsum('ij,ab,jb,ka,ic,kc', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
        IaDk += 1/2 * lib.einsum('ij,ab,jb,ka,ilcd,klcd', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk -= 1/2 * lib.einsum('ij,ab,jb,kc,ilad,klcd', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
        IaDk -= 1/2 * lib.einsum('ij,ab,jb,kc,ilad,klcd', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk -= 1/2 * lib.einsum('ij,ab,jb,kc,ilad,lkdc', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk -= 1/2 * lib.einsum('ij,ab,jb,kc,ilad,klcd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk -= lib.einsum('ij,ab,jc,ka,ildc,kldb', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,jc,kd,ilac,kldb', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,jc,kc,ia,kb', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IaDk += lib.einsum('ij,ab,jc,kc,ilad,lkdb', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,jc,kc,ilad,klbd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk -= lib.einsum('ij,ab,jc,kd,ilac,klbd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk += lib.einsum('ij,ab,ka,kc,ic,jb', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IaDk += lib.einsum('ij,ab,ka,kc,ilcd,ljdb', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,ka,kc,ilcd,jlbd', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk -= lib.einsum('ij,ab,ka,lc,ilcd,kjdb', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,ka,lb,ilcd,kjcd', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,ka,lc,ildc,kjdb', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,kc,ia,jb', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,kc,ilad,ljdb', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,kc,ilad,jlbd', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk += lib.einsum('ij,ab,kc,kd,ilac,ljdb', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,kc,lc,ikad,ljdb', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,ld,ikac,ljdb', S_oc_ab, S_vir_ab, Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,ld,ikac,jlbd', S_oc_ab, S_vir_ab, Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_bbbb, optimize = True)
        IaDk += lib.einsum('ij,ab,kb,kc,ia,jc', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IaDk += lib.einsum('ij,ab,kb,kc,ilad,ljdc', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,kb,kc,ilad,jlcd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk += lib.einsum('ij,ab,kb,lc,ikad,ljcd', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,kb,lc,ikad,jlcd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,kc,ia,jb', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,kc,ilad,ljdb', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,kc,ilad,jlbd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk += lib.einsum('ij,ab,kc,kd,ilac,jlbd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,ld,ikac,ljdb', S_oc_ab, S_vir_ab, Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
        IaDk += lib.einsum('ij,ab,kc,lc,ikad,jlbd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
        IaDk -= lib.einsum('ij,ab,kc,ld,ikac,jlbd', S_oc_ab, S_vir_ab, Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)

        if (method == "adc(2)-x") or (method == "adc(3)"):

            IjCd -= 1/4 * lib.einsum('ia,bj,ijba,klcd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd -= lib.einsum('ia,bj,ijba,klcd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= 1/4 * lib.einsum('ia,bj,ijba,klcd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd += lib.einsum('ia,bj,ijbc,klda,kldc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += 1/2 * lib.einsum('ia,bj,ijbc,klad,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd += 1/2 * lib.einsum('ia,bj,ijca,klbd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd += lib.einsum('ia,bj,ijca,klbd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,ijcd,klba,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,ikbc,jlad,klcd', S_ov_ab, S_vo_ab, t1_ccee_aaaa, Y_bbbb, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,ikbc,ljda,klcd', S_ov_ab, S_vo_ab, t1_ccee_aaaa, Y_abab, Y_aaaa, optimize = True)
            IjCd -= 1/2 * lib.einsum('ia,bj,ikcd,ljba,klcd', S_ov_ab, S_vo_ab, t1_ccee_aaaa, Y_abab, Y_aaaa, optimize = True)
            IjCd += 1/2 * lib.einsum('ia,bj,ikba,jlcd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd += lib.einsum('ia,bj,ikba,ljcd,lkcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,ikbc,jlad,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
            IjCd -= lib.einsum('ia,bj,ikbc,ljda,lkdc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,ikca,ljbd,lkcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('ia,bj,ikcd,ljba,lkcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,jkac,ilbd,lkdc', S_ov_ab, S_vo_ab, t1_ccee_bbbb, Y_aaaa, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,jkac,ilbd,klcd', S_ov_ab, S_vo_ab, t1_ccee_bbbb, Y_abab, Y_bbbb, optimize = True)
            IjCd -= 1/2 * lib.einsum('ia,bj,jkcd,ilba,klcd', S_ov_ab, S_vo_ab, t1_ccee_bbbb, Y_abab, Y_bbbb, optimize = True)
            IjCd += 1/2 * lib.einsum('ia,bj,kjba,ilcd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd += lib.einsum('ia,bj,kjba,ilcd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,kjbc,ilda,kldc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= lib.einsum('ia,bj,kjca,ilbd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
            IjCd -= lib.einsum('ia,bj,kjca,ilbd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('ia,bj,kjcd,ilba,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= 1/2 * lib.einsum('ia,bj,klbc,ijda,klcd', S_ov_ab, S_vo_ab, t1_ccee_aaaa, Y_abab, Y_aaaa, optimize = True)
            IjCd -= lib.einsum('ia,bj,klba,ijcd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('ia,bj,klbc,ijda,kldc', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd += lib.einsum('ia,bj,klca,ijbd,klcd', S_ov_ab, S_vo_ab, t1_ccee_abab, Y_abab, Y_abab, optimize = True)
            IjCd -= 1/2 * lib.einsum('ia,bj,klac,ijbd,klcd', S_ov_ab, S_vo_ab, t1_ccee_bbbb, Y_abab, Y_bbbb, optimize = True)
    
        na = lib.einsum('pp',temp_a)
        nb = lib.einsum('pp',temp_b)
        
        spin_c = 0.25*((na - nb)**2) + 0.5*(na + nb) + IjKl + AbCd + 2*IjCd + IaKd + 2*IaDk + AiDk
        spin = np.append(spin,spin_c)
        trace_a = np.append(trace_a,na)
        trace_b = np.append(trace_b,nb)
    
    return spin, (trace_a, trace_b)


#@profile
def get_X(adc):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2
    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b


    if adc.f_ov is None:
        f_ov_a = np.zeros((nocc_a, nvir_a))
        f_ov_b = np.zeros((nocc_b, nvir_b))
        t1_1_a = np.zeros((nocc_a, nvir_a))
        t1_1_b = np.zeros((nocc_b, nvir_b))
    else:
        f_ov_a, f_ov_b = adc.f_ov
        t1_1_a = t1[2][0][:]
        t1_1_b = t1[2][1][:]

    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b


    TY_a = np.zeros((nmo_a,nmo_a))
    TY_b = np.zeros((nmo_b,nmo_b))

    dm_a = adc.dm_a.copy()
    dm_b = adc.dm_b.copy()



    n_singles_a = nocc_a * nvir_a
    n_singles_b = nocc_b * nvir_b
    n_doubles_aaaa = nocc_a * (nocc_a - 1) * nvir_a * (nvir_a -1) // 4
    n_doubles_ab = nocc_a * nocc_b * nvir_a * nvir_b
    n_doubles_bbbb = nocc_b * (nocc_b - 1) * nvir_b * (nvir_b -1) // 4

    dim = n_singles_a + n_singles_b + n_doubles_aaaa + n_doubles_ab + n_doubles_bbbb

    
    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaaa = f_b
    f_aaaa = s_aaaa + n_doubles_aaaa
    s_abab = f_aaaa
    f_ab = s_abab + n_doubles_ab
    s_bbbb = f_ab
    f_bbbb = s_bbbb + n_doubles_bbbb

    U = adc.U.T
    nroots = U.shape[0]

    x = np.array([])
    TY_aa = []
    TY_bb = []
    for r in range(U.shape[0]):
        
        Y_a = U[r][:f_a].reshape(nocc_a, nvir_a)
        Y_b = U[r][f_a:f_b].reshape(nocc_b, nvir_b)


        Y1_abab = U[r][s_abab:f_ab].reshape(nocc_a, nocc_b, nvir_a, nvir_b)

        Y_vv_u_a = np.zeros((int((nocc_a * (nocc_a - 1))/2),nvir_a, nvir_a))
        Y_vv_u_a[:,ab_ind_a[0],ab_ind_a[1]]= U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2)) 
        Y_vv_u_a[:,ab_ind_a[1],ab_ind_a[0]]= -U[r][s_aaaa:f_aaaa].reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2))
        Y1_oovv_u_a = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        Y1_oovv_u_a[ij_ind_a[0],ij_ind_a[1],:,:]= Y_vv_u_a
        Y1_oovv_u_a[ij_ind_a[1],ij_ind_a[0],:,:]= -Y_vv_u_a

        del Y_vv_u_a

        Y_vv_u_b = np.zeros((int((nocc_b * (nocc_b - 1))/2),nvir_b, nvir_b))
        Y_vv_u_b[:,ab_ind_b[0],ab_ind_b[1]]= U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y_vv_u_b[:,ab_ind_b[1],ab_ind_b[0]]= -U[r][s_bbbb:f_bbbb].reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2))
        Y1_oovv_u_b = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        Y1_oovv_u_b[ij_ind_b[0],ij_ind_b[1],:,:]= Y_vv_u_b
        Y1_oovv_u_b[ij_ind_b[1],ij_ind_b[0],:,:]= -Y_vv_u_b

        del Y_vv_u_b


        # T_U2_Y0 qp block
        t1_2_a = t1[0][0][:]
        TY_a[:nocc_a,:nocc_a] = -lib.einsum('pg,qg->pq',Y_a,t1_2_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] = lib.einsum('xr,xv->vr',Y_a,t1_2_a,optimize=True)
        del t1_2_a


        TY_a[:nocc_a,nocc_a:] = Y_a
        TY_b[:nocc_b,nocc_b:] = Y_b



        t1_2_b = t1[0][1][:]
        TY_b[:nocc_b,:nocc_b] = -lib.einsum('pg,qg->pq',Y_b,t1_2_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] = lib.einsum('xr,xv->vr',Y_b,t1_2_b,optimize=True)
        del t1_2_b

        t2_1_a = t2[0][0][:]
        t2_1_ab = t2[0][1][:]
        TY_a[:nocc_a,:nocc_a] += lib.einsum('xg,ph,qxgh->pq',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('pg,xh,qxgh->pq',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= 0.5*lib.einsum('pxgh,qxgh->pq',Y1_oovv_u_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] += 0.5*lib.einsum('xyrg,xyvg->vr',Y1_oovv_u_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] -= lib.einsum('xg,yr,xyvg->vr',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] += 0.5*np.einsum('xr,yg,xyvg->vr',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,nocc_a:] += 0.5*lib.einsum('xg,xygh,pyvh->pv',Y_a,t2_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,nocc_a:] -= 0.25*lib.einsum('pg,xygh,xyvh->pv',Y_a,t2_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,:nocc_a] = lib.einsum('xg,pxvg->vp',Y_a,t2_1_a,optimize=True)
        TY_b[:nocc_b,nocc_b:] += 0.5*lib.einsum('xg,xygh,yphv->pv',Y_a,t2_1_a,t2_1_ab,optimize=True)
        TY_a[:nocc_a,nocc_a:] += 0.5*lib.einsum('xg,yxhg,pyvh->pv',Y_b,t2_1_ab,t2_1_a,optimize=True)
        TY_a[:nocc_a,nocc_a:] -= 0.25*lib.einsum('xv,xygh,pygh->pv',Y_a,t2_1_a,t2_1_a,optimize=True)
        del t2_1_a
        del t2_1_ab


        t2_1_b = t2[0][2][:]
        t2_1_ab = t2[0][1][:]
        TY_b[:nocc_b,:nocc_b] += lib.einsum('xg,ph,qxgh->pq',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('pg,xh,qxgh->pq',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= 0.5*lib.einsum('pxgh,qxgh->pq',Y1_oovv_u_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] += 0.5*lib.einsum('xyrg,xyvg->vr',Y1_oovv_u_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] -= lib.einsum('xg,yr,xyvg->vr',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] += 0.5*np.einsum('xr,yg,xyvg->vr',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,nocc_b:] += 0.5*lib.einsum('xg,xygh,pyvh->pv',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,nocc_b:] -= 0.25*lib.einsum('xv,xygh,pygh->pv',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,:nocc_b] = lib.einsum('xg,pxvg->vp',Y_b,t2_1_b,optimize=True)
        TY_a[:nocc_a,nocc_a:] += 0.5*lib.einsum('xg,xygh,pyvh->pv',Y_b,t2_1_b,t2_1_ab,optimize=True)
        TY_b[:nocc_b,nocc_b:] -= 0.25*lib.einsum('pg,xygh,xyvh->pv',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,nocc_b:] += 0.5*lib.einsum('xg,xygh,pyvh->pv',Y_a,t2_1_ab,t2_1_b,optimize=True)
        del t2_1_b
        del t2_1_ab

        t2_1_ab = t2[0][1][:]
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('xg,ph,qxhg->pq',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('xg,ph,xqgh->pq',Y_a,t1_1_b,t2_1_ab,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('pg,xh,qxgh->pq',Y_a,t1_1_b,t2_1_ab,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('pg,xh,xqhg->pq',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('pxgh,qxgh->pq',Y1_abab,t2_1_ab,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('xpgh,xqgh->pq',Y1_abab,t2_1_ab,optimize=True)
        TY_a[nocc_a:,nocc_a:] += lib.einsum('xyrg,xyvg->vr',Y1_abab,t2_1_ab,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xygr,xygv->vr',Y1_abab,t2_1_ab,optimize=True)
        TY_a[nocc_a:,nocc_a:] += lib.einsum('xg,yr,yxvg->vr',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xg,yr,xygv->vr',Y_a,t1_1_b,t2_1_ab,optimize=True)
        TY_a[nocc_a:,nocc_a:] += 0.5*np.einsum('xr,yg,xyvg->vr',Y_a,t1_1_b,t2_1_ab,optimize=True)
        TY_b[nocc_b:,nocc_b:] += 0.5*np.einsum('xr,yg,yxgv->vr',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_a[:nocc_a,nocc_a:] += 0.5*lib.einsum('xg,xygh,pyvh->pv',Y_a,t2_1_ab,t2_1_ab,optimize=True)
        TY_b[:nocc_b,nocc_b:] += 0.5*lib.einsum('xg,yxhg,yphv->pv',Y_b,t2_1_ab,t2_1_ab,optimize=True)
        TY_a[:nocc_a,nocc_a:] -= 0.5*lib.einsum('pg,xygh,xyvh->pv',Y_a,t2_1_ab,t2_1_ab,optimize=True)
        TY_b[:nocc_b,nocc_b:] -= 0.5*lib.einsum('pg,xyhg,xyhv->pv',Y_b,t2_1_ab,t2_1_ab,optimize=True)
        TY_a[:nocc_a,nocc_a:] -= 0.5*lib.einsum('xv,xygh,pygh->pv',Y_a,t2_1_ab,t2_1_ab,optimize=True)
        TY_b[:nocc_b,nocc_b:] -= 0.5*lib.einsum('xv,yxgh,ypgh->pv',Y_b,t2_1_ab,t2_1_ab,optimize=True)
        TY_a[nocc_a:,:nocc_a] += lib.einsum('xg,pxvg->vp',Y_b,t2_1_ab,optimize=True)
        TY_b[nocc_b:,:nocc_b] += lib.einsum('xg,xpgv->vp',Y_a,t2_1_ab,optimize=True)
        del t2_1_ab 
       
       
       
       # T_U1_Y0 qp block
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('pg,qg->pq',Y_a,t1_1_a,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('pg,qg->pq',Y_b,t1_1_b,optimize=True)

        TY_a[nocc_a:,nocc_a:] += lib.einsum('xr,xv->vr',Y_a,t1_1_a,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xr,xv->vr',Y_b,t1_1_b,optimize=True)



        TY_a[:nocc_a,nocc_a:] -= 0.5*np.einsum('pg,xg,xv->pv',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[:nocc_b,nocc_b:] -= 0.5*np.einsum('pg,xg,xv->pv',Y_b,t1_1_b,t1_1_b,optimize=True)

        TY_a[:nocc_a,nocc_a:] -= 0.5*np.einsum('xv,xg,pg->pv',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[:nocc_b,nocc_b:] -= 0.5*np.einsum('xv,xg,pg->pv',Y_b,t1_1_b,t1_1_b,optimize=True)

        TY_a[nocc_a:,:nocc_a] -= lib.einsum('xg,pg,xv->vp',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= lib.einsum('xg,pg,xv->vp',Y_b,t1_1_b,t1_1_b,optimize=True)


        t2_2_a = t2[1][0][:]
        TY_a[nocc_a:,:nocc_a] += lib.einsum('xg,pxvg->vp',Y_a,t2_2_a,optimize=True)
        del t2_2_a                                     
                                                       
                                                       
        t2_2_b = t2[1][2][:]                           
        TY_b[nocc_b:,:nocc_b] += lib.einsum('xg,pxvg->vp',Y_b,t2_2_b,optimize=True)
        del t2_2_b                                     
                                                       
        t2_2_ab = t2[1][1][:]                          
        TY_b[nocc_b:,:nocc_b] += lib.einsum('xg,xpgv->vp',Y_a,t2_2_ab,optimize=True)
        TY_a[nocc_a:,:nocc_a] += lib.einsum('xg,pxvg->vp',Y_b,t2_2_ab,optimize=True)
        del t2_2_ab

        if (method == "adc(2)"):
            del Y1_abab
            del Y1_oovv_u_a
            del Y1_oovv_u_b



        if (method == "adc(2)-x") or (method == "adc(3)"):
            # T_U2_Y1 qp block
            t2_2_a = t2[1][0][:]
            TY_a[:nocc_a,:nocc_a] -= 0.5*lib.einsum('pxgh,qxgh->pq',Y1_oovv_u_a,t2_2_a,optimize=True)
            TY_a[nocc_a:,nocc_a:] += 0.5*lib.einsum('xyrg,xyvg->vr',Y1_oovv_u_a,t2_2_a,optimize=True)
            del t2_2_a
            
            t2_2_b = t2[1][2][:]
            TY_b[:nocc_b,:nocc_b] -= 0.5*lib.einsum('pxgh,qxgh->pq',Y1_oovv_u_b,t2_2_b,optimize=True)
            TY_b[nocc_b:,nocc_b:] += 0.5*lib.einsum('xyrg,xyvg->vr',Y1_oovv_u_b,t2_2_b,optimize=True)
            del t2_2_b

            t2_2_ab = t2[1][1][:]
            TY_a[:nocc_a,:nocc_a] -= lib.einsum('pxgh,qxgh->pq',Y1_abab,t2_2_ab,optimize=True)
            TY_b[:nocc_b,:nocc_b] -= lib.einsum('xpgh,xqgh->pq',Y1_abab,t2_2_ab,optimize=True)
            # T_U2_Y1 rv block
            TY_a[nocc_a:,nocc_a:] += lib.einsum('xyrg,xyvg->vr',Y1_abab,t2_2_ab,optimize=True)
            TY_b[nocc_b:,nocc_b:] += lib.einsum('xygr,xygv->vr',Y1_abab,t2_2_ab,optimize=True)
            del t2_2_ab

 #           # T_U2_Y1 vp block

            t2_1_a = t2[0][0][:]
            TY_a[:nocc_a,nocc_a:] += 0.25*lib.einsum('pxgh,yv,xygh->pv',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[:nocc_a,nocc_a:] += 0.25*np.einsum('xyvg,ph,xygh->pv',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[nocc_a:,:nocc_a] -= 0.5*np.einsum('xygh,pg,xyvh->vp',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[nocc_a:,:nocc_a] += 0.5*np.einsum('xygh,yv,pxgh->vp',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            del t2_1_a

            t2_1_ab = t2[0][1][:]
            TY_a[:nocc_a,nocc_a:] -= 0.5*np.einsum('xyvg,ph,xyhg->pv',Y1_abab,t1_1_a,t2_1_ab,optimize=True)
            TY_a[:nocc_a,nocc_a:] -= 0.5*lib.einsum('pxgh,yv,yxgh->pv',Y1_abab,t1_1_a,t2_1_ab,optimize=True)
            TY_b[:nocc_b,nocc_b:] -= 0.5*lib.einsum('xpgh,yv,xygh->pv',Y1_abab,t1_1_b,t2_1_ab,optimize=True)
            TY_b[:nocc_b,nocc_b:] -= 0.5*np.einsum('xygv,ph,xygh->pv',Y1_abab,t1_1_b,t2_1_ab,optimize=True)
            TY_a[nocc_a:,:nocc_a] -= np.einsum('xygh,pg,xyvh->vp',Y1_abab,t1_1_a,t2_1_ab,optimize=True)
            TY_b[nocc_b:,:nocc_b] -= np.einsum('xyhg,pg,xyhv->vp',Y1_abab,t1_1_b,t2_1_ab,optimize=True)
            TY_a[nocc_a:,:nocc_a] -= np.einsum('yxgh,yv,pxgh->vp',Y1_abab,t1_1_a,t2_1_ab,optimize=True)
            TY_b[nocc_b:,:nocc_b] -= np.einsum('xyhg,yv,xphg->vp',Y1_abab,t1_1_b,t2_1_ab,optimize=True)
            del t2_1_ab


            t2_1_b = t2[0][2][:]
            TY_b[:nocc_b,nocc_b:] += 0.25*lib.einsum('pxgh,yv,xygh->pv',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[:nocc_b,nocc_b:] += 0.25*np.einsum('xyvg,ph,xygh->pv',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[nocc_b:,:nocc_b] -= 0.5*np.einsum('xygh,pg,xyvh->vp',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[nocc_b:,:nocc_b] += 0.5*np.einsum('xygh,yv,pxgh->vp',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            del t2_1_b

        if (method == "adc(2)-x"):
            del Y1_abab
            del Y1_oovv_u_a
            del Y1_oovv_u_b
        else:
            t2_ce_aa = t1[0][0][:]
            t2_ce_bb = t1[0][1][:]
            einsum_type = True

            t1_ccee_aaaa = t2[0][0][:]
            t1_ccee_bbbb = t2[0][2][:]
            t1_ccee_abab = t2[0][1][:]

            t2_ccee_aaaa = t2[1][0][:]
            t2_ccee_bbbb = t2[1][2][:]
            t2_ccee_abab = t2[1][1][:]
            
        if adc.f_ov is None:
            f_ov_a = np.zeros((nocc_a, nvir_a))
            f_ov_b = np.zeros((nocc_b, nvir_b))
            t1_ce_aa = np.zeros((nocc_a, nvir_a))
            t1_ce_bb = np.zeros((nocc_b, nvir_b))
        else:
            f_ov_a, f_ov_b = adc.f_ov
            t1_ce_aa = t1[2][0][:]
            t1_ce_bb = t1[2][1][:]

        if (method == "adc(3)"):

#            TY_a[:nocc_a,:nocc_a] -= lib.einsum('Ia,La->IL', Y_a, t3_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,Liab,ib->IL', Y_a, t1_ccee_aaaa, t2_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,Liab,ib->IL', Y_a, t1_ccee_abab, t2_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,ib,Liab->IL', Y_a, t1_ce_aa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Ia,ib,Liab->IL', Y_a, t1_ce_bb, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += lib.einsum('ia,Ib,Liab->IL', Y_a, t1_ce_aa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += lib.einsum('ia,Liab,Ib->IL', Y_a, t1_ccee_aaaa, t2_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.166666667 * lib.einsum('Ia,Liab,ijbc,jc->IL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.166666667 * lib.einsum('Ia,Liab,ijbc,jc->IL', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.166666667 * lib.einsum('Ia,Liab,ijbc,jc->IL', Y_a, t1_ccee_abab, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.166666667 * lib.einsum('Ia,Liab,jicb,jc->IL', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.0833333335 * lib.einsum('Ia,ia,Ljbc,ijbc->IL', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.166666667 * lib.einsum('Ia,ia,Lb,ib->IL', Y_a, t1_ce_aa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.166666667 * lib.einsum('Ia,ia,Ljbc,ijbc->IL', Y_a, t1_ce_aa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.0833333335 * lib.einsum('Ia,ijab,Lc,ijbc->IL', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.166666667 * lib.einsum('Ia,ijab,Lc,ijcb->IL', Y_a, t1_ccee_abab, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,Liab,Ijbc,jc->IL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,Liab,Ijbc,jc->IL', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.2500000005 * lib.einsum('ia,Libc,ja,Ijbc->IL', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,ijab,Ijbc,Lc->IL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.2500000005 * lib.einsum('ia,ijbc,La,Ijbc->IL', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,ib,La,Ib->IL', Y_a, t1_ce_aa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,ib,Ljac,Ijbc->IL', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,ib,Ljac,Ijbc->IL', Y_a, t1_ce_aa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.500000001 * lib.einsum('ia,ijab,Ijcb,Lc->IL', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,ijbc,La,Ijbc->IL', Y_a, t1_ccee_abab, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            
            TY_a[:nocc_a,:nocc_a] -= lib.einsum('ia,Ib,Liba->IL', Y_b, t1_ce_aa, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= lib.einsum('ia,Liba,Ib->IL', Y_b, t1_ccee_abab, t2_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.500000001 * lib.einsum('ia,Liba,Ijbc,jc->IL', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.500000001 * lib.einsum('ia,Liba,Ijbc,jc->IL', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,Libc,ja,Ijbc->IL', Y_b, t1_ccee_abab, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] -= 0.500000001 * lib.einsum('ia,ijab,Ijcb,Lc->IL', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,ib,Ljca,Ijcb->IL', Y_b, t1_ce_bb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,:nocc_a] += 0.500000001 * lib.einsum('ia,jiba,Ijbc,Lc->IL', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            
#            TY_a[nocc_a:,nocc_a:] += lib.einsum('iC,iA->AC', Y_a, t3_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,ijAa,ja->AC', Y_a, t1_ccee_aaaa, t2_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,ijAa,ja->AC', Y_a, t1_ccee_abab, t2_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,ja,ijAa->AC', Y_a, t1_ce_aa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 1/2 * lib.einsum('iC,ja,ijAa->AC', Y_a, t1_ce_bb, t2_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= lib.einsum('ia,jC,ijAa->AC', Y_a, t1_ce_aa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= lib.einsum('ia,ijAa,jC->AC', Y_a, t1_ccee_aaaa, t2_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.166666667 * lib.einsum('iC,ijAa,jkab,kb->AC', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.166666667 * lib.einsum('iC,ijAa,jkab,kb->AC', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.0833333335 * lib.einsum('iC,ijab,kA,jkab->AC', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.166666667 * lib.einsum('iC,ia,jA,ja->AC', Y_a, t1_ce_aa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.0833333335 * lib.einsum('iC,ia,jkAb,jkab->AC', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.166666667 * lib.einsum('iC,ia,jkAb,jkab->AC', Y_a, t1_ce_aa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.166666667 * lib.einsum('iC,ijAa,jkab,kb->AC', Y_a, t1_ccee_abab, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.166666667 * lib.einsum('iC,ijAa,kjba,kb->AC', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.166666667 * lib.einsum('iC,ijab,kA,kjab->AC', Y_a, t1_ccee_abab, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,iA,ja,jC->AC', Y_a, t1_ce_aa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.2500000005 * lib.einsum('ia,iA,jkab,jkCb->AC', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,iA,jkab,jkCb->AC', Y_a, t1_ce_aa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,ijAa,jkCb,kb->AC', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,ijAa,jkCb,kb->AC', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.500000001 * lib.einsum('ia,ijAb,ka,jkCb->AC', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,ijab,kA,jkCb->AC', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.2500000005 * lib.einsum('ia,ib,jkAa,jkCb->AC', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,ijAb,ka,kjCb->AC', Y_a, t1_ccee_abab, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.500000001 * lib.einsum('ia,ijab,kA,kjCb->AC', Y_a, t1_ccee_abab, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            
            
            TY_a[nocc_a:,nocc_a:] += lib.einsum('ia,jC,jiAa->AC', Y_b, t1_ce_aa, t2_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += lib.einsum('ia,jiAa,jC->AC', Y_b, t1_ccee_abab, t2_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.500000001 * lib.einsum('ia,ijab,kA,kjCb->AC', Y_b, t1_ccee_bbbb, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,ib,jkAa,jkCb->AC', Y_b, t1_ce_bb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.500000001 * lib.einsum('ia,jiAa,jkCb,kb->AC', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] += 0.500000001 * lib.einsum('ia,jiAa,jkCb,kb->AC', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,jiAb,ka,jkCb->AC', Y_b, t1_ccee_abab, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,nocc_a:] -= 0.500000001 * lib.einsum('ia,jiba,kA,jkCb->AC', Y_b, t1_ccee_abab, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            
            
            TY_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('Ia,iC,ia->IC', Y_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('Ia,ia,iC->IC', Y_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/4 * lib.einsum('Ia,ijCb,ijab->IC', Y_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/4 * lib.einsum('Ia,ijab,ijCb->IC', Y_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('Ia,ijCb,ijab->IC', Y_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('Ia,ijab,ijCb->IC', Y_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/4 * lib.einsum('iC,Ijab,ijab->IC', Y_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('iC,Ia,ia->IC', Y_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('iC,Ijab,ijab->IC', Y_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/4 * lib.einsum('iC,ijab,Ijab->IC', Y_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('iC,ia,Ia->IC', Y_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('iC,ijab,Ijab->IC', Y_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ia,IjCb,ijab->IC', Y_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ia,IjCb,ijab->IC', Y_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ia,ijab,IjCb->IC', Y_a, t1_ccee_aaaa, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ia,ijab,IjCb->IC', Y_a, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.333333334 * lib.einsum('Ia,ia,ijCb,jb->IC', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.333333334 * lib.einsum('Ia,ia,ijCb,jb->IC', Y_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.166666667 * lib.einsum('Ia,ijab,iC,jb->IC', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.166666667 * lib.einsum('Ia,ijab,iC,jb->IC', Y_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.166666667 * lib.einsum('iC,ijab,Ia,jb->IC', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.333333334 * lib.einsum('iC,ia,Ijab,jb->IC', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.333333334 * lib.einsum('iC,ia,Ijab,jb->IC', Y_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.166666667 * lib.einsum('iC,ijab,Ia,jb->IC', Y_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.666666668 * lib.einsum('ia,ijab,jC,Ib->IC', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.333333334 * lib.einsum('ia,ib,ja,IjCb->IC', Y_a, t1_ce_aa, t1_ce_aa, t1_ccee_aaaa, optimize = einsum_type)
            
            
            TY_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ia,IjCb,jiba->IC', Y_b, t1_ccee_aaaa, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ia,IjCb,ijab->IC', Y_b, t1_ccee_abab, t2_ccee_bbbb, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ia,ijab,IjCb->IC', Y_b, t1_ccee_bbbb, t2_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ia,jiba,IjCb->IC', Y_b, t1_ccee_abab, t2_ccee_aaaa, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.333333334 * lib.einsum('ia,ib,ja,IjCb->IC', Y_b, t1_ce_bb, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_a[:nocc_a,nocc_a:] -= 0.666666668 * lib.einsum('ia,jiba,jC,Ib->IC', Y_b, t1_ccee_abab, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            
            
#            TY_a[nocc_a:,:nocc_a] += lib.einsum('ia,LiAa->AL', Y_a, t3_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= lib.einsum('ia,iA,La->AL', Y_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= lib.einsum('ia,La,iA->AL', Y_a, t1_ce_aa, t2_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.500000001 * lib.einsum('ia,iA,Ljab,jb->AL', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.500000001 * lib.einsum('ia,iA,Ljab,jb->AL', Y_a, t1_ce_aa, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.166666667 * lib.einsum('ia,LiAb,ja,jb->AL', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.0833333335 * lib.einsum('ia,LiAb,jkac,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.166666667 * lib.einsum('ia,LiAb,jkac,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.666666668 * lib.einsum('ia,Liab,jA,jb->AL', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.333333334 * lib.einsum('ia,Liab,jkAc,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.666666668 * lib.einsum('ia,Liab,jkAc,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,Libc,jkAa,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.333333334 * lib.einsum('ia,ijAa,Lkbc,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.666666668 * lib.einsum('ia,ijAa,Lb,jb->AL', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.666666668 * lib.einsum('ia,ijAa,Lkbc,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.500000001 * lib.einsum('ia,ijAb,La,jb->AL', Y_a, t1_ccee_aaaa, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.666666668 * lib.einsum('ia,ijAb,Lkac,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.666666668 * lib.einsum('ia,ijAb,Lkac,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,ijab,LkAc,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,ijab,LkAc,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.0833333335 * lib.einsum('ia,ijbc,LkAa,jkbc->AL', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.166666667 * lib.einsum('ia,ib,LjAa,jb->AL', Y_a, t1_ce_aa, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.500000001 * lib.einsum('ia,ijAb,La,jb->AL', Y_a, t1_ccee_abab, t1_ce_aa, t1_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.666666668 * lib.einsum('ia,ijAb,Lkac,kjcb->AL', Y_a, t1_ccee_abab, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.666666668 * lib.einsum('ia,ijAb,Lkac,jkbc->AL', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,ijab,LkAc,kjcb->AL', Y_a, t1_ccee_abab, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,ijab,LkAc,jkbc->AL', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.166666667 * lib.einsum('ia,ijbc,LkAa,kjbc->AL', Y_a, t1_ccee_abab, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            
            
#            TY_a[nocc_a:,:nocc_a] += lib.einsum('ia,LiAa->AL', Y_b, t3_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.166666667 * lib.einsum('ia,LiAb,ja,jb->AL', Y_b, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.166666667 * lib.einsum('ia,LiAb,jkca,jkcb->AL', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.0833333335 * lib.einsum('ia,LiAb,jkac,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.666666668 * lib.einsum('ia,Liba,jA,jb->AL', Y_b, t1_ccee_abab, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.333333334 * lib.einsum('ia,Liba,jkAc,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.666666668 * lib.einsum('ia,Liba,jkAc,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.666666668 * lib.einsum('ia,Libc,jkAa,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,ijab,LkAc,kjcb->AL', Y_b, t1_ccee_bbbb, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,ijab,LkAc,jkbc->AL', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.0833333335 * lib.einsum('ia,ijbc,LkAa,jkbc->AL', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ccee_bbbb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.166666667 * lib.einsum('ia,ib,LjAa,jb->AL', Y_b, t1_ce_bb, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.333333334 * lib.einsum('ia,jiAa,Lkbc,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.666666668 * lib.einsum('ia,jiAa,Lb,jb->AL', Y_b, t1_ccee_abab, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.666666668 * lib.einsum('ia,jiAa,Lkbc,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.666666668 * lib.einsum('ia,jiAb,Lkca,jkcb->AL', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,jiba,LkAc,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] += 0.166666667 * lib.einsum('ia,jiba,LkAc,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_a[nocc_a:,:nocc_a] -= 0.166666667 * lib.einsum('ia,jibc,LkAa,jkbc->AL', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
            TY_b[:nocc_b,:nocc_b] -= lib.einsum('ja,ib,jlab->il', Y_a, t1_ce_bb, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= lib.einsum('ja,jlab,ib->il', Y_a, t1_ccee_abab, t2_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.500000001 * lib.einsum('ja,jlab,ikbc,kc->il', Y_a, t1_ccee_abab, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.500000001 * lib.einsum('ja,jlab,kicb,kc->il', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,jlbc,ka,kibc->il', Y_a, t1_ccee_abab, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.500000001 * lib.einsum('ja,jkab,lc,kibc->il', Y_a, t1_ccee_aaaa, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,jb,klac,kibc->il', Y_a, t1_ce_aa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,jkab,ikbc,lc->il', Y_a, t1_ccee_abab, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            
#            TY_b[:nocc_b,:nocc_b] -= lib.einsum('ia,la->il', Y_b, t3_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,ljab,jb->il', Y_b, t1_ccee_bbbb, t2_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jlba,jb->il', Y_b, t1_ccee_abab, t2_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,jlba->il', Y_b, t1_ce_aa, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ia,jb,ljab->il', Y_b, t1_ce_bb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += lib.einsum('ja,ib,ljab->il', Y_b, t1_ce_bb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += lib.einsum('ja,ljab,ib->il', Y_b, t1_ccee_bbbb, t2_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.166666667 * lib.einsum('ia,ljab,jkbc,kc->il', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.166666667 * lib.einsum('ia,ljab,kjcb,kc->il', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.0833333335 * lib.einsum('ia,ja,lkbc,jkbc->il', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.166666667 * lib.einsum('ia,ja,lb,jb->il', Y_b, t1_ce_bb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.166666667 * lib.einsum('ia,ja,klbc,kjbc->il', Y_b, t1_ce_bb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.166666667 * lib.einsum('ia,jlba,jkbc,kc->il', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.166666667 * lib.einsum('ia,jlba,jkbc,kc->il', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.166666667 * lib.einsum('ia,jkba,lc,jkbc->il', Y_b, t1_ccee_abab, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.0833333335 * lib.einsum('ia,jkab,lc,jkbc->il', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,ljab,ikbc,kc->il', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,ljab,kicb,kc->il', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.2500000005 * lib.einsum('ja,ljbc,ka,ikbc->il', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,jkab,ikbc,lc->il', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.2500000005 * lib.einsum('ja,jkbc,la,ikbc->il', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,jb,la,ib->il', Y_b, t1_ce_bb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,jb,lkac,ikbc->il', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,jb,klca,kicb->il', Y_b, t1_ce_bb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] -= 0.500000001 * lib.einsum('ja,kjba,lc,kibc->il', Y_b, t1_ccee_abab, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,:nocc_b] += 0.500000001 * lib.einsum('ja,kjbc,la,kibc->il', Y_b, t1_ccee_abab, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            
            TY_b[nocc_b:,nocc_b:] += lib.einsum('ib,jc,ijba->ac', Y_a, t1_ce_bb, t2_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += lib.einsum('ib,ijba,jc->ac', Y_a, t1_ccee_abab, t2_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.500000001 * lib.einsum('ib,ijbd,ka,jkdc->ac', Y_a, t1_ccee_aaaa, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,id,jkba,jkdc->ac', Y_a, t1_ce_aa, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.500000001 * lib.einsum('ib,ijba,jkcd,kd->ac', Y_a, t1_ccee_abab, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.500000001 * lib.einsum('ib,ijba,kjdc,kd->ac', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,ijbd,ka,jkcd->ac', Y_a, t1_ccee_abab, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,ijda,kb,kjdc->ac', Y_a, t1_ccee_abab, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            
            
#            TY_b[nocc_b:,nocc_b:] += lib.einsum('ic,ia->ac', Y_b, t3_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,ijab,jb->ac', Y_b, t1_ccee_bbbb, t2_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jiba,jb->ac', Y_b, t1_ccee_abab, t2_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,jiba->ac', Y_b, t1_ce_aa, t2_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 1/2 * lib.einsum('ic,jb,ijab->ac', Y_b, t1_ce_bb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= lib.einsum('ib,jc,ijab->ac', Y_b, t1_ce_bb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= lib.einsum('ib,ijab,jc->ac', Y_b, t1_ccee_bbbb, t2_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.166666667 * lib.einsum('ic,ijab,jkbd,kd->ac', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.166666667 * lib.einsum('ic,ijab,kjdb,kd->ac', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.0833333335 * lib.einsum('ic,ijbd,ka,jkbd->ac', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.166666667 * lib.einsum('ic,ib,ja,jb->ac', Y_b, t1_ce_bb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.166666667 * lib.einsum('ic,ib,jkda,jkdb->ac', Y_b, t1_ce_bb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.0833333335 * lib.einsum('ic,ib,jkad,jkbd->ac', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.166666667 * lib.einsum('ic,jiba,jkbd,kd->ac', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.166666667 * lib.einsum('ic,jiba,jkbd,kd->ac', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.166666667 * lib.einsum('ic,jibd,ka,jkbd->ac', Y_b, t1_ccee_abab, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,ia,jb,jc->ac', Y_b, t1_ce_bb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,ia,jkdb,jkdc->ac', Y_b, t1_ce_bb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.2500000005 * lib.einsum('ib,ia,jkbd,jkcd->ac', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,ijab,jkcd,kd->ac', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,ijab,kjdc,kd->ac', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.500000001 * lib.einsum('ib,ijad,kb,jkcd->ac', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,ijbd,ka,jkcd->ac', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.2500000005 * lib.einsum('ib,id,jkab,jkcd->ac', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] -= 0.500000001 * lib.einsum('ib,jida,kb,jkdc->ac', Y_b, t1_ccee_abab, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,nocc_b:] += 0.500000001 * lib.einsum('ib,jidb,ka,jkdc->ac', Y_b, t1_ccee_abab, t1_ce_bb, t1_ccee_abab, optimize = einsum_type)
            
            
            TY_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,ikcb,jkab->ic', Y_a, t1_ccee_bbbb, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,jkab,kibc->ic', Y_a, t1_ccee_aaaa, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,jkab,ikcb->ic', Y_a, t1_ccee_abab, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,kibc,jkab->ic', Y_a, t1_ccee_abab, t2_ccee_aaaa, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.333333334 * lib.einsum('ja,jb,ka,kibc->ic', Y_a, t1_ce_aa, t1_ce_aa, t1_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.666666668 * lib.einsum('ja,jkab,kc,ib->ic', Y_a, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            
            
            TY_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ia,jc,ja->ic', Y_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ia,ja,jc->ic', Y_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ia,jkbc,jkba->ic', Y_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ia,jkba,jkbc->ic', Y_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/4 * lib.einsum('ia,jkcb,jkab->ic', Y_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/4 * lib.einsum('ia,jkab,jkcb->ic', Y_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/4 * lib.einsum('jc,ikab,jkab->ic', Y_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('jc,ia,ja->ic', Y_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/4 * lib.einsum('jc,jkab,ikab->ic', Y_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('jc,ja,ia->ic', Y_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('jc,kiab,kjab->ic', Y_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('jc,kjab,kiab->ic', Y_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,ikcb,jkab->ic', Y_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,jkab,ikcb->ic', Y_b, t1_ccee_bbbb, t2_ccee_bbbb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,kibc,kjba->ic', Y_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,kjba,kibc->ic', Y_b, t1_ccee_abab, t2_ccee_abab, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.333333334 * lib.einsum('ia,ja,jkcb,kb->ic', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.333333334 * lib.einsum('ia,ja,kjbc,kb->ic', Y_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.166666667 * lib.einsum('ia,jkba,kc,jb->ic', Y_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.166666667 * lib.einsum('ia,jkab,jc,kb->ic', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.166666667 * lib.einsum('jc,jkab,ia,kb->ic', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.333333334 * lib.einsum('jc,ja,ikab,kb->ic', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.333333334 * lib.einsum('jc,ja,kiba,kb->ic', Y_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.166666667 * lib.einsum('jc,kjab,ib,ka->ic', Y_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.666666668 * lib.einsum('ja,jkab,kc,ib->ic', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[:nocc_b,nocc_b:] -= 0.333333334 * lib.einsum('ja,jb,ka,ikcb->ic', Y_b, t1_ce_bb, t1_ce_bb, t1_ccee_bbbb, optimize = einsum_type)
            
            
            
#            TY_b[nocc_b:,:nocc_b] += lib.einsum('ib,ilba->al', Y_a, t3_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.666666668 * lib.einsum('ib,ilbc,ja,jc->al', Y_a, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.666666668 * lib.einsum('ib,ilbc,jkda,jkdc->al', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.333333334 * lib.einsum('ib,ilbc,jkad,jkcd->al', Y_a, t1_ccee_abab, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.166666667 * lib.einsum('ib,ilca,jb,jc->al', Y_a, t1_ccee_abab, t1_ce_aa, t1_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.0833333335 * lib.einsum('ib,ilca,jkbd,jkcd->al', Y_a, t1_ccee_abab, t1_ccee_aaaa, t1_ccee_aaaa, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.166666667 * lib.einsum('ib,ilca,jkbd,jkcd->al', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.666666668 * lib.einsum('ib,ilcd,jkba,jkcd->al', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,ijbc,lkad,jkcd->al', Y_a, t1_ccee_aaaa, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,ijbc,jkcd,klda->al', Y_a, t1_ccee_aaaa, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.0833333335 * lib.einsum('ib,ijcd,klba,jkcd->al', Y_a, t1_ccee_aaaa, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.166666667 * lib.einsum('ib,ic,jlba,jc->al', Y_a, t1_ce_aa, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.333333334 * lib.einsum('ib,ijba,lkcd,jkcd->al', Y_a, t1_ccee_abab, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.666666668 * lib.einsum('ib,ijba,lc,jc->al', Y_a, t1_ccee_abab, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.666666668 * lib.einsum('ib,ijba,klcd,kjcd->al', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,ijbc,lkad,jkcd->al', Y_a, t1_ccee_abab, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,ijbc,klda,kjdc->al', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.666666668 * lib.einsum('ib,ijca,klbd,kjcd->al', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.166666667 * lib.einsum('ib,ijcd,klba,kjcd->al', Y_a, t1_ccee_abab, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            
#            TY_b[nocc_b:,:nocc_b] += lib.einsum('ib,liab->al', Y_b, t3_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= lib.einsum('ib,ia,lb->al', Y_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= lib.einsum('ib,lb,ia->al', Y_b, t1_ce_bb, t2_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.500000001 * lib.einsum('ib,ia,ljbc,jc->al', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.500000001 * lib.einsum('ib,ia,jlcb,jc->al', Y_b, t1_ce_bb, t1_ccee_abab, t1_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.166666667 * lib.einsum('ib,liac,jb,jc->al', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.166666667 * lib.einsum('ib,liac,jkdb,jkdc->al', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.0833333335 * lib.einsum('ib,liac,jkbd,jkcd->al', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.666666668 * lib.einsum('ib,libc,ja,jc->al', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.666666668 * lib.einsum('ib,libc,jkda,jkdc->al', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.333333334 * lib.einsum('ib,libc,jkad,jkcd->al', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,licd,jkab,jkcd->al', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.333333334 * lib.einsum('ib,ijab,lkcd,jkcd->al', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.666666668 * lib.einsum('ib,ijab,lc,jc->al', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.666666668 * lib.einsum('ib,ijab,klcd,kjcd->al', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.500000001 * lib.einsum('ib,ijac,lb,jc->al', Y_b, t1_ccee_bbbb, t1_ce_bb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.666666668 * lib.einsum('ib,ijac,lkbd,jkcd->al', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.666666668 * lib.einsum('ib,ijac,kldb,kjdc->al', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,ijbc,lkad,jkcd->al', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,ijbc,klda,kjdc->al', Y_b, t1_ccee_bbbb, t1_ccee_abab, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.0833333335 * lib.einsum('ib,ijcd,lkab,jkcd->al', Y_b, t1_ccee_bbbb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.166666667 * lib.einsum('ib,ic,ljab,jc->al', Y_b, t1_ce_bb, t1_ccee_bbbb, t1_ce_bb, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.500000001 * lib.einsum('ib,jica,lb,jc->al', Y_b, t1_ccee_abab, t1_ce_bb, t1_ce_aa, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.666666668 * lib.einsum('ib,jica,lkbd,jkcd->al', Y_b, t1_ccee_abab, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.666666668 * lib.einsum('ib,jica,kldb,jkcd->al', Y_b, t1_ccee_abab, t1_ccee_abab, t1_ccee_aaaa, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,jicb,lkad,jkcd->al', Y_b, t1_ccee_abab, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] += 0.166666667 * lib.einsum('ib,jicb,jkcd,klda->al', Y_b, t1_ccee_abab, t1_ccee_aaaa, t1_ccee_abab, optimize = einsum_type)
            TY_b[nocc_b:,:nocc_b] -= 0.166666667 * lib.einsum('ib,jicd,lkab,jkcd->al', Y_b, t1_ccee_abab, t1_ccee_bbbb, t1_ccee_abab, optimize = einsum_type)



        dx = lib.einsum("rqp,qp->r", dm_a, TY_a, optimize = True)
        dx += lib.einsum("rqp,qp->r", dm_b, TY_b, optimize = True)

        TY_aa = np.append(TY_aa,TY_a)
        TY_bb = np.append(TY_bb,TY_b)
        TY = (TY_aa, TY_bb)

        x = np.append(x,dx)
    x = x.reshape(nroots, 3)

    return TY, x


#def get_trans_moments(adc):
#
#    cput0 = (logger.process_clock(), logger.perf_counter())
#    log = logger.Logger(adc.stdout, adc.verbose)
#    nmo_a  = adc.nmo_a
#    nmo_b  = adc.nmo_b
#
#    T_a = []
#    T_b = []
#
#    for orb in range(nmo_a):
#        T_aa = get_trans_moments_orbital(adc,orb, spin = "alpha")
#        T_a.append(T_aa)
#
#    for orb in range(nmo_b):
#        T_bb = get_trans_moments_orbital(adc,orb, spin = "beta")
#        T_b.append(T_bb)
#
#    cput0 = log.timer_debug1("completed spec vactor calc in ADC(3) calculation", *cput0)
#    return (T_a, T_b)
#
#
#def get_trans_moments_orbital(adc, orb, spin="alpha"):
#
#    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
#        raise NotImplementedError(adc.method)
#
#    method = adc.method
#
#    if (adc.approx_trans_moments == False or adc.method == "adc(3)"):
#        t1_2_a, t1_2_b = adc.t1[0]
#
#    t2_1_a = adc.t2[0][0][:]
#    t2_1_ab = adc.t2[0][1][:]
#    t2_1_b = adc.t2[0][2][:]
#
#    nocc_a = adc.nocc_a
#    nocc_b = adc.nocc_b
#    nvir_a = adc.nvir_a
#    nvir_b = adc.nvir_b
#
#    ij_ind_a = np.tril_indices(nocc_a, k=-1)
#    ij_ind_b = np.tril_indices(nocc_b, k=-1)
#
#    n_singles_a = nocc_a
#    n_singles_b = nocc_b
#    n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2
#    n_doubles_bab = nvir_b * nocc_a* nocc_b
#    n_doubles_aba = nvir_a * nocc_b* nocc_a
#    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2
#
#    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb
#
#    idn_occ_a = np.identity(nocc_a)
#    idn_occ_b = np.identity(nocc_b)
#
#    s_a = 0
#    f_a = n_singles_a
#    s_b = f_a
#    f_b = s_b + n_singles_b
#    s_aaa = f_b
#    f_aaa = s_aaa + n_doubles_aaa
#    s_bab = f_aaa
#    f_bab = s_bab + n_doubles_bab
#    s_aba = f_bab
#    f_aba = s_aba + n_doubles_aba
#    s_bbb = f_aba
#    f_bbb = s_bbb + n_doubles_bbb
#
#    T = np.zeros((dim))
#
######### spin = alpha  ############################################
#
#    if spin == "alpha":
#        pass  # placeholder to mute flake8 warning
#
######### ADC(2) 1h part  ############################################
#
#        t2_1_a = adc.t2[0][0][:]
#        t2_1_ab = adc.t2[0][1][:]
#        if orb < nocc_a:
#            T[s_a:f_a]  = idn_occ_a[orb, :]
#            T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
#            T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
#            T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
#        else:
#            if (adc.approx_trans_moments == False or adc.method == "adc(3)"):
#                T[s_a:f_a] += t1_2_a[:,(orb-nocc_a)]
#
######### ADC(2) 2h-1p  part  ############################################
#
#            t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
#            t2_1_t_a = t2_1_t.transpose(2,1,0)
#            t2_1_t_ab = t2_1_ab.transpose(2,3,1,0)
#
#            T[s_aaa:f_aaa] = t2_1_t_a[(orb-nocc_a),:,:].reshape(-1)
#            T[s_bab:f_bab] = t2_1_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
#
######### ADC(3) 2h-1p  part  ############################################
#
#        if (adc.method == "adc(2)-x" and adc.approx_trans_moments == False) or (adc.method == "adc(3)"):
#
#            t2_2_a = adc.t2[1][0][:]
#            t2_2_ab = adc.t2[1][1][:]
#
#            if orb >= nocc_a:
#                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:]
#                t2_2_t_a = t2_2_t.transpose(2,1,0)
#                t2_2_t_ab = t2_2_ab.transpose(2,3,1,0)
#
#                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
#                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
#
######### ADC(3) 1h part  ############################################
#
#        if (method == 'adc(3)'):
#
#            if (adc.approx_trans_moments == False):
#                t1_3_a, t1_3_b = adc.t1[1]
#
#            if orb < nocc_a:
#
#                t2_1_a_tmp = np.ascontiguousarray(t2_1_a[:,orb,:,:])
#                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[orb,:,:,:])
#
#                T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a_tmp, t2_2_a, optimize = True)
#                T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab_tmp, t2_2_ab, optimize = True)
#                T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab_tmp, t2_2_ab, optimize = True)
#
#                del t2_1_a_tmp, t2_1_ab_tmp
#
#                t2_2_a_tmp = np.ascontiguousarray(t2_2_a[:,orb,:,:])
#                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[orb,:,:,:])
#
#                T[s_a:f_a] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a,  t2_2_a_tmp,optimize = True)
#                T[s_a:f_a] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1_ab, t2_2_ab_tmp,optimize = True)
#                T[s_a:f_a] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1_ab, t2_2_ab_tmp,optimize = True)
#
#                del t2_2_a_tmp, t2_2_ab_tmp
#            else:
#                t2_1_a_tmp =  np.ascontiguousarray(t2_1_a[:,:,(orb-nocc_a),:])
#                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:,(orb-nocc_a),:])
#
#                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_a_tmp, t1_2_a,optimize = True)
#                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_ab_tmp, t1_2_b,optimize = True)
#                if (adc.approx_trans_moments == False):
#                    T[s_a:f_a] += t1_3_a[:,(orb-nocc_a)]
#                del t2_1_a_tmp, t2_1_ab_tmp
#
#                del t2_2_a
#                del t2_2_ab
#
#        del t2_1_a
#        del t2_1_ab
######### spin = beta  ############################################
#
#    else:
#        pass  # placeholder
#
######### ADC(2) 1h part  ############################################
#
#        t2_1_b = adc.t2[0][2][:]
#        t2_1_ab = adc.t2[0][1][:]
#        if orb < nocc_b:
#
#            t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
#            t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])
#
#            T[s_b:f_b] = idn_occ_b[orb, :]
#            T[s_b:f_b]+= 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_1_b, optimize = True)
#            T[s_b:f_b]-= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_1_ab, optimize = True)
#            T[s_b:f_b]-= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_1_ab, optimize = True)
#            del t2_1_b_tmp, t2_1_ab_tmp
#        else:
#            if (adc.approx_trans_moments == False or adc.method == "adc(3)"):
#                T[s_b:f_b] += t1_2_b[:,(orb-nocc_b)]
#
######### ADC(2) 2h-1p part  ############################################
#
#            t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
#            t2_1_t_b = t2_1_t.transpose(2,1,0)
#            t2_1_t_ab = t2_1_ab.transpose(2,3,0,1)
#
#            T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
#            T[s_aba:f_aba] = t2_1_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
#
######### ADC(3) 2h-1p part  ############################################
#
#        if (adc.method == "adc(2)-x" and adc.approx_trans_moments == False) or (adc.method == "adc(3)"):
#
#            t2_2_a = adc.t2[1][0][:]
#            t2_2_ab = adc.t2[1][1][:]
#            t2_2_b = adc.t2[1][2][:]
#
#            if orb >= nocc_b:
#                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:]
#                t2_2_t_b = t2_2_t.transpose(2,1,0)
#
#                t2_2_t_ab = t2_2_ab.transpose(2,3,0,1)
#
#                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
#                T[s_aba:f_aba] += t2_2_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
#
######### ADC(3) 1h part  ############################################
#
#        if (method=='adc(3)'):
#
#            if (adc.approx_trans_moments == False):
#                t1_3_a, t1_3_b = adc.t1[1]
#
#            if orb < nocc_b:
#
#                t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
#                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])
#
#                T[s_b:f_b] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_2_b, optimize = True)
#                T[s_b:f_b] -= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_2_ab, optimize = True)
#                T[s_b:f_b] -= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_2_ab, optimize = True)
#
#                del t2_1_b_tmp, t2_1_ab_tmp
#
#                t2_2_b_tmp = np.ascontiguousarray(t2_2_b[:,orb,:,:])
#                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[:,orb,:,:])
#
#                T[s_b:f_b] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_b,  t2_2_b_tmp ,optimize = True)
#                T[s_b:f_b] -= 0.25*lib.einsum('kicd,kcd->i',t2_1_ab, t2_2_ab_tmp,optimize = True)
#                T[s_b:f_b] -= 0.25*lib.einsum('kidc,kdc->i',t2_1_ab, t2_2_ab_tmp,optimize = True)
#
#                del t2_2_b_tmp, t2_2_ab_tmp
#
#            else:
#                t2_1_b_tmp  = np.ascontiguousarray(t2_1_b[:,:,(orb-nocc_b),:])
#                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:,:,(orb-nocc_b)])
#
#                T[s_b:f_b] += 0.5*lib.einsum('ikc,kc->i',t2_1_b_tmp, t1_2_b,optimize = True)
#                T[s_b:f_b] += 0.5*lib.einsum('kic,kc->i',t2_1_ab_tmp, t1_2_a,optimize = True)
#                if (adc.approx_trans_moments == False):
#                    T[s_b:f_b] += t1_3_b[:,(orb-nocc_b)]
#                del t2_1_b_tmp, t2_1_ab_tmp
#                del t2_2_b
#                del t2_2_ab
#
#        del t2_1_b
#        del t2_1_ab
#
#    return T
#
#
#def analyze_eigenvector(adc):
#
#    nocc_a = adc.nocc_a
#    nocc_b = adc.nocc_b
#    nvir_a = adc.nvir_a
#    nvir_b = adc.nvir_b
#    evec_print_tol = adc.evec_print_tol
#
#    logger.info(adc, "Number of alpha occupied orbitals = %d", nocc_a)
#    logger.info(adc, "Number of beta occupied orbitals = %d", nocc_b)
#    logger.info(adc, "Number of alpha virtual orbitals =  %d", nvir_a)
#    logger.info(adc, "Number of beta virtual orbitals =  %d", nvir_b)
#    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)
#
#    ij_a = np.tril_indices(nocc_a, k=-1)
#    ij_b = np.tril_indices(nocc_b, k=-1)
#
#    n_singles_a = nocc_a
#    n_singles_b = nocc_b
#    n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2
#    n_doubles_bab = nvir_b * nocc_a* nocc_b
#    n_doubles_aba = nvir_a * nocc_b* nocc_a
#    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2
#
#    s_a = 0
#    f_a = n_singles_a
#    s_b = f_a
#    f_b = s_b + n_singles_b
#    s_aaa = f_b
#    f_aaa = s_aaa + n_doubles_aaa
#    s_bab = f_aaa
#    f_bab = s_bab + n_doubles_bab
#    s_aba = f_bab
#    f_aba = s_aba + n_doubles_aba
#    s_bbb = f_aba
#    f_bbb = s_bbb + n_doubles_bbb
#
#    U = adc.U
#
#    for I in range(U.shape[1]):
#        U1 = U[:f_b,I]
#        U2 = U[f_b:,I]
#        U1dotU1 = np.dot(U1, U1)
#        U2dotU2 = np.dot(U2, U2)
#
#        temp_aaa = np.zeros((nvir_a, nocc_a, nocc_a))
#        temp_aaa[:,ij_a[0],ij_a[1]] =  U[s_aaa:f_aaa,I].reshape(nvir_a,-1).copy()
#        temp_aaa[:,ij_a[1],ij_a[0]] = -U[s_aaa:f_aaa,I].reshape(nvir_a,-1).copy()
#        U_aaa = temp_aaa.reshape(-1).copy()
#
#        temp_bbb = np.zeros((nvir_b, nocc_b, nocc_b))
#        temp_bbb[:,ij_b[0],ij_b[1]] =  U[s_bbb:f_bbb,I].reshape(nvir_b,-1).copy()
#        temp_bbb[:,ij_b[1],ij_b[0]] = -U[s_bbb:f_bbb,I].reshape(nvir_b,-1).copy()
#        U_bbb = temp_bbb.reshape(-1).copy()
#
#        U_sq = U[:,I].copy()**2
#        ind_idx = np.argsort(-U_sq)
#        U_sq = U_sq[ind_idx]
#        U_sorted = U[ind_idx,I].copy()
#
#        U_sq_aaa = U_aaa.copy()**2
#        U_sq_bbb = U_bbb.copy()**2
#        ind_idx_aaa = np.argsort(-U_sq_aaa)
#        ind_idx_bbb = np.argsort(-U_sq_bbb)
#        U_sq_aaa = U_sq_aaa[ind_idx_aaa]
#        U_sq_bbb = U_sq_bbb[ind_idx_bbb]
#        U_sorted_aaa = U_aaa[ind_idx_aaa].copy()
#        U_sorted_bbb = U_bbb[ind_idx_bbb].copy()
#
#        U_sorted = U_sorted[U_sq > evec_print_tol**2]
#        ind_idx = ind_idx[U_sq > evec_print_tol**2]
#        U_sorted_aaa = U_sorted_aaa[U_sq_aaa > evec_print_tol**2]
#        U_sorted_bbb = U_sorted_bbb[U_sq_bbb > evec_print_tol**2]
#        ind_idx_aaa = ind_idx_aaa[U_sq_aaa > evec_print_tol**2]
#        ind_idx_bbb = ind_idx_bbb[U_sq_bbb > evec_print_tol**2]
#
#        singles_a_idx = []
#        singles_b_idx = []
#        doubles_aaa_idx = []
#        doubles_bab_idx = []
#        doubles_aba_idx = []
#        doubles_bbb_idx = []
#        singles_a_val = []
#        singles_b_val = []
#        doubles_bab_val = []
#        doubles_aba_val = []
#        iter_idx = 0
#        for orb_idx in ind_idx:
#
#            if orb_idx in range(s_a,f_a):
#                i_idx = orb_idx + 1
#                singles_a_idx.append(i_idx)
#                singles_a_val.append(U_sorted[iter_idx])
#
#            if orb_idx in range(s_b,f_b):
#                i_idx = orb_idx - s_b + 1
#                singles_b_idx.append(i_idx)
#                singles_b_val.append(U_sorted[iter_idx])
#
#            if orb_idx in range(s_bab,f_bab):
#                aij_idx = orb_idx - s_bab
#                ij_rem = aij_idx % (nocc_a*nocc_b)
#                a_idx = aij_idx//(nocc_a*nocc_b)
#                i_idx = ij_rem//nocc_a
#                j_idx = ij_rem % nocc_a
#                doubles_bab_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
#                doubles_bab_val.append(U_sorted[iter_idx])
#
#            if orb_idx in range(s_aba,f_aba):
#                aij_idx = orb_idx - s_aba
#                ij_rem = aij_idx % (nocc_b*nocc_a)
#                a_idx = aij_idx//(nocc_b*nocc_a)
#                i_idx = ij_rem//nocc_b
#                j_idx = ij_rem % nocc_b
#                doubles_aba_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
#                doubles_aba_val.append(U_sorted[iter_idx])
#
#            iter_idx += 1
#
#        for orb_aaa in ind_idx_aaa:
#            ij_rem = orb_aaa % (nocc_a*nocc_a)
#            a_idx = orb_aaa//(nocc_a*nocc_a)
#            i_idx = ij_rem//nocc_a
#            j_idx = ij_rem % nocc_a
#            doubles_aaa_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
#
#        for orb_bbb in ind_idx_bbb:
#            ij_rem = orb_bbb % (nocc_b*nocc_b)
#            a_idx = orb_bbb//(nocc_b*nocc_b)
#            i_idx = ij_rem//nocc_b
#            j_idx = ij_rem % nocc_b
#            doubles_bbb_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
#
#        doubles_aaa_val = list(U_sorted_aaa)
#        doubles_bbb_val = list(U_sorted_bbb)
#
#        logger.info(adc,'%s | root %d | norm(1h)  = %6.4f | norm(2h1p) = %6.4f ',adc.method ,I, U1dotU1, U2dotU2)
#
#        if singles_a_val:
#            logger.info(adc, "\n1h(alpha) block: ")
#            logger.info(adc, "     i     U(i)")
#            logger.info(adc, "------------------")
#            for idx, print_singles in enumerate(singles_a_idx):
#                logger.info(adc, '  %4d   %7.4f', print_singles, singles_a_val[idx])
#
#        if singles_b_val:
#            logger.info(adc, "\n1h(beta) block: ")
#            logger.info(adc, "     i     U(i)")
#            logger.info(adc, "------------------")
#            for idx, print_singles in enumerate(singles_b_idx):
#                logger.info(adc, '  %4d   %7.4f', print_singles, singles_b_val[idx])
#
#        if doubles_aaa_val:
#            logger.info(adc, "\n2h1p(alpha|alpha|alpha) block: ")
#            logger.info(adc, "     i     j     a     U(i,j,a)")
#            logger.info(adc, "-------------------------------")
#            for idx, print_doubles in enumerate(doubles_aaa_idx):
#                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
#                            print_doubles[1], print_doubles[2], print_doubles[0], doubles_aaa_val[idx])
#
#        if doubles_bab_val:
#            logger.info(adc, "\n2h1p(beta|alpha|beta) block: ")
#            logger.info(adc, "     i     j     a     U(i,j,a)")
#            logger.info(adc, "-------------------------------")
#            for idx, print_doubles in enumerate(doubles_bab_idx):
#                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1],
#                            print_doubles[2], print_doubles[0], doubles_bab_val[idx])
#
#        if doubles_aba_val:
#            logger.info(adc, "\n2h1p(alpha|beta|alpha) block: ")
#            logger.info(adc, "     i     j     a     U(i,j,a)")
#            logger.info(adc, "-------------------------------")
#            for idx, print_doubles in enumerate(doubles_aba_idx):
#                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
#                            print_doubles[1], print_doubles[2], print_doubles[0], doubles_aba_val[idx])
#
#        if doubles_bbb_val:
#            logger.info(adc, "\n2h1p(beta|beta|beta) block: ")
#            logger.info(adc, "     i     j     a     U(i,j,a)")
#            logger.info(adc, "-------------------------------")
#            for idx, print_doubles in enumerate(doubles_bbb_idx):
#                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
#                            print_doubles[1], print_doubles[2], print_doubles[0], doubles_bbb_val[idx])
#
#        logger.info(adc, "\n*************************************************************\n")


def analyze_spec_factor(adc):
    X_ab, props = adc.X
    
    energy = adc.E*27.2114

    X_a = (X_ab[0].copy())**2
    X_b = (X_ab[1].copy())**2

    nmo_a = adc.nocc_a + adc.nvir_a
    nmo_b = adc.nocc_b + adc.nvir_b
    
    logger.info(adc, "Print spectroscopic factors > %E\n", adc.spec_factor_print_tol)
    
    
    if isinstance(adc._scf, scf.rohf.ROHF):
        sym_a = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff.orbsym]
        sym_b = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff.orbsym]
    elif not adc.mol.symmetry:
        sym_a = np.repeat(['A'], X_a.shape[0])
        sym_b = np.repeat(['A'], X_b.shape[0])
    else:
        sym_a = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff[0].orbsym]
        sym_b = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff[1].orbsym]

    sym_a = np.array(sym_a)
    sym_b = np.array(sym_b)

    spin_a = "Alpha"
    spin_b = "Beta"

    thresh = adc.spec_factor_print_tol
    
    for root_a in range(X_a.shape[0]):


        X_a_root = X_a[root_a,:]

        sort_a = np.argsort(-X_a_root)

        X_a_root = X_a_root[sort_a]
        
        X_a_root = X_a_root[X_a_root > thresh]
        
        if np.sum(X_a_root) == 0.0:
            continue

        logger.info(adc, '%s | root %d %s\n', adc.method, root_a, spin_a)
        logger.info(adc, "Hole_MO       Particle_MO        X_a^2       Orbital symmetry      excitation energy")
        logger.info(adc, "-----------------------------------------------------------")

        for hp in range(X_a_root.shape[0]):
            P = ((sort_a[hp]) %  nmo_a)
            H = ((sort_a[hp]) // nmo_a)

            logger.info(adc, '%4.d           %4.d            %10.8f      %s -> %s           %12.8f ', (H+1), (P+1), X_a_root[hp], sym_a[H], sym_a[P], energy[root_a])

        logger.info(adc, '\nPartial norm of X_a = %10.8f', np.linalg.norm(np.sqrt(X_a_root)))
        logger.info(adc, "*************************************************************\n")

    for root_b in range(X_b.shape[0]):

        X_b_root = X_b[root_b,:]

        sort_b = np.argsort(-X_b_root)

        X_b_root = X_b_root[sort_b]

        X_b_root = X_b_root[X_b_root > thresh]

        if np.sum(X_b_root) == 0.0:
            continue
        logger.info(adc, '%s | root %d %s\n', adc.method, root_b, spin_b)
        logger.info(adc, "Hole_MO       Particle_MO        X_b^2       Orbital symmetry      excitation energy")
        logger.info(adc, "-----------------------------------------------------------")

        for hp in range(X_b_root.shape[0]):
            P = sort_b[hp] %  nmo_b
            H = sort_b[hp] // nmo_b

            logger.info(adc, '%4.d           %4.d            %10.8f      %s -> %s           %12.8f', (H+1), (P+1), X_b_root[hp], sym_b[H], sym_b[P], energy[root_b])
            
        logger.info(adc, '\nPartial norm of X_b = %10.8f', np.linalg.norm(np.sqrt(X_b_root)))
        logger.info(adc, "*************************************************************\n")


#@profile
def get_properties(adc, nroots=1):

    #Transition moments
    TY, dx  = adc.get_X()

    X_a = TY[0].reshape(nroots,-1)
    X_b = TY[1].reshape(nroots,-1)

    X = (X_a,X_b)

    if adc.opdm is True:
        opdm = adc.get_opdm()
    else:
        opdm = None
    
    if adc.tpdm is True:
        tpdm = adc.get_tpdm()
    else:
        tpdm = None
    
    if adc.old_spin_c is True:
        spin = adc.get_old_spin_contamination()
    
    if adc.spin_c is True:
        spin = adc.get_spin_contamination()
    else:
        spin = None
   
    density_matrix = (spin, opdm, tpdm)

    P = np.square(dx.T)*adc.E*(2/3)
    P = P[0] + P[1] + P[2]
    
    return P, (X,density_matrix)


def analyze(myadc):

  #  header = ("\n*************************************************************"
  #            "\n           Eigenvector analysis summary"
  #            "\n*************************************************************")
  #  logger.info(myadc, header)

  #  myadc.analyze_eigenvector()

    if myadc.compute_properties:

        header = ("\n*************************************************************"
                  "\n            Spectroscopic factors analysis summary"
                  "\n*************************************************************")
        logger.info(myadc, header)

        myadc.analyze_spec_factor()


#def compute_dyson_mo(myadc):
#
#    X_a = myadc.X[0]
#    X_b = myadc.X[1]
#
#    if X_a is None:
#        nroots = myadc.U.shape[1]
#        P,X_a,X_b = myadc.get_properties(nroots)
#
#    nroots = X_a.shape[1]
#    dyson_mo_a = np.dot(myadc.mo_coeff[0],X_a)
#    dyson_mo_b = np.dot(myadc.mo_coeff[1],X_b)
#
#    dyson_mo = (dyson_mo_a,dyson_mo_b)
#
#    return dyson_mo


class UADCEE(uadc.UADC):
    '''unrestricted ADC for EE energies and spectroscopic amplitudes

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).
        conv_tol : float
            Convergence threshold for Davidson iterations.  Default is 1e-12.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.UADC(mf).run()
            >>> myadcee = adc.UADC(myadc).run()

    Saved results

        e_ee : float or list of floats
            EE energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ee : array
            Eigenvectors for each EE transition.
        p_ee : float
            Spectroscopic amplitudes for each EE transition.
    '''
    def __init__(self, adc):
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.f_ov = adc.f_ov
        self.dm_a = adc.dm_a
        self.dm_b = adc.dm_b
        self.opdm = adc.opdm
        self.tpdm = adc.tpdm
        self.old_spin_c = adc.old_spin_c
        self.spin_c = adc.spin_c
        self.nucl_dip = adc.nucl_dip
        self.imds = adc.imds
        self.e_corr = adc.e_corr
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self.nocc_a = adc._nocc[0]
        self.nocc_b = adc._nocc[1]
        self.nvir_a = adc._nvir[0]
        self.nvir_b = adc._nvir[1]
        self.mo_coeff = adc.mo_coeff
        self.mo_energy_a = adc.mo_energy_a
        self.mo_energy_b = adc.mo_energy_b
        self.nmo_a = adc._nmo[0]
        self.nmo_b = adc._nmo[1]
        self.mol = adc.mol
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.spec_factor_print_tol = adc.spec_factor_print_tol
        self.evec_print_tol = adc.evec_print_tol

        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments
        self.E = adc.E
        self.U = adc.U
        self.P = adc.P
        self.X = adc.X

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method',
                    'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory',
                    't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = uadc.kernel
    get_imds = get_imds
    get_diag = get_diag
    matvec = matvec
    get_X = get_X
    get_old_spin_contamination = get_old_spin_contamination
    get_spin_contamination = get_spin_contamination
    get_tpdm = get_tpdm
    get_opdm = get_opdm
#    get_trans_moments = get_trans_moments
    get_properties = get_properties

    analyze_spec_factor = analyze_spec_factor
  #  analyze_eigenvector = analyze_eigenvector
    analyze = analyze
  #  compute_dyson_mo = compute_dyson_mo

 #   @profile
    def get_init_guess(self, nroots=1, diag=None, ascending = True):
        if diag is None :
            diag = self.get_diag()
        idx = None
        if ascending:
            idx = np.argsort(diag)
        else:
            idx = np.argsort(diag)[::-1]
        guess = np.zeros((diag.shape[0], nroots))
        min_shape = min(diag.shape[0], nroots)
        guess[:min_shape,:min_shape] = np.identity(min_shape)
        g = np.zeros((diag.shape[0], nroots))
        g[idx] = guess.copy()
        guess = []
        for p in range(g.shape[1]):
            guess.append(g[:,p])
        return guess

  #  @profile
    def gen_matvec(self, imds=None, eris=None):
        if imds is None: imds = self.get_imds(eris)
       # exit()
        diag = self.get_diag(imds,eris)
        matvec = self.matvec(imds, eris)
        #matvec = lambda x: self.matvec()
        return matvec, diag

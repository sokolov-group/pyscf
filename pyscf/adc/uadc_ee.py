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
#@profile
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

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

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
#

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
            t1_1_a = t1[2][0][:]
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
            t1_1_b = t1[2][1][:]
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
            t1_1_a = t1[2][0][:]
            t1_1_b = t1[2][1][:]
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
            t1_1_a = t1[2][0][:]
            t1_1_b = t1[2][1][:]
            eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 2*lib.einsum('me,meba->ab',t1_1_a,eris_ovVV,optimize = True)
            M_aabb += lib.einsum('je,iaeb->iajb',t1_1_b,eris_ovVV,optimize = True)
            del eris_ovVV
        del t1_1_a
        del t1_1_b

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

        if (method == "adc(2)-x"):
            
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

            #s[s_abab:f_ab] += interim_abab.reshape(-1)
            del r1_ab
            del r2_a
            del r2_b
            
        return s

    return sigma_

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

    tpdm_a = np.array([])
    tpdm_b = np.array([])
    mixed_spin = np.array([])

    spin = np.array([])
    
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
        
        
        temp_a[:nocc_a,nocc_a:]  = lib.einsum('IjCa,ja->IC', Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IjCa,ja->IC', Y_abab, Y_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IC,ja,ja->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IC,ja,ja->IC', t1_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= lib.einsum('jC,ja,Ia->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_aaaa, Y_abab, Y_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('Ijab,jkab,kC->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= lib.einsum('Ia,ja,jC->IC', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IjCa,jkab,kb->IC', t1_ccee_abab, Y_bbbb, Y_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IjCa,kjba,kb->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= lib.einsum('Ijab,kjab,kC->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('jkCa,jkab,Ib->IC', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= lib.einsum('jkCa,jkba,Ib->IC', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IC,ja,ja->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += lib.einsum('IC,ja,ja->IC', t2_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= lib.einsum('jC,ja,Ia->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= lib.einsum('Ia,ja,jC->IC', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('Ia,ja,jkCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('Ia,ja,jkCb,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('Ia,jb,ka,jkCb->IC', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('Ia,jb,ka,kjCb->IC', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('jC,ja,Ikab,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('jC,ja,Ikab,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('jC,ka,jb,Ikab->IC', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('jC,ka,jb,Ikba->IC', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('ja,jb,IkCa,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('ja,ka,IjCb,kb->IC', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_bb, Y_bb, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[:nocc_a,nocc_a:] += 1/2 * lib.einsum('ja,ja,IkCb,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('ja,jb,IkCa,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('ja,ka,IjCb,kb->IC', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        
        
        temp_a[nocc_a:,:nocc_a]  = lib.einsum('LiAb,ib->AL', Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LiAb,ib->AL', Y_abab, Y_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LA,ib,ib->AL', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LA,ib,ib->AL', t1_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= lib.einsum('iA,ib,Lb->AL', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LiAb,ijbc,jc->AL', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LiAb,ijbc,jc->AL', t1_ccee_aaaa, Y_abab, Y_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] += 1/2 * lib.einsum('Libc,ijbc,jA->AL', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= lib.einsum('Lb,ib,iA->AL', t1_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LiAb,ijbc,jc->AL', t1_ccee_abab, Y_bbbb, Y_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LiAb,jicb,jc->AL', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= lib.einsum('Libc,jibc,jA->AL', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += 1/2 * lib.einsum('ijAb,ijbc,Lc->AL', t1_ccee_aaaa, Y_aaaa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= lib.einsum('ijAb,ijcb,Lc->AL', t1_ccee_abab, Y_abab, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LA,ib,ib->AL', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += lib.einsum('LA,ib,ib->AL', t2_ce_aa, Y_bb, Y_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= lib.einsum('iA,ib,Lb->AL', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= lib.einsum('Lb,ib,iA->AL', t2_ce_aa, Y_aa, Y_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('Lb,ib,ijAc,jc->AL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('Lb,ib,ijAc,jc->AL', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] += 1/2 * lib.einsum('Lb,ic,jb,ijAc->AL', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('Lb,ic,jb,jiAc->AL', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('iA,ib,Ljbc,jc->AL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('iA,ib,Ljbc,jc->AL', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] += 1/2 * lib.einsum('iA,jb,ic,Ljbc->AL', Y_aa, Y_aa, t1_ce_aa, t1_ccee_aaaa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('iA,jb,ic,Ljcb->AL', Y_aa, Y_bb, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_a[nocc_a:,:nocc_a] += 1/2 * lib.einsum('ib,ib,LjAc,jc->AL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += 1/2 * lib.einsum('ib,ib,LjAc,jc->AL', Y_aa, Y_aa, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('ib,ic,LjAb,jc->AL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('ib,jb,LiAc,jc->AL', Y_aa, Y_aa, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += 1/2 * lib.einsum('ib,ib,LjAc,jc->AL', Y_bb, Y_bb, t1_ccee_aaaa, t1_ce_aa, optimize = True)
        temp_a[nocc_a:,:nocc_a] += 1/2 * lib.einsum('ib,ib,LjAc,jc->AL', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('ib,ic,LjAb,jc->AL', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('ib,jb,LiAc,jc->AL', Y_bb, Y_bb, t1_ccee_abab, t1_ce_bb, optimize = True)
        
        
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
        
        
        temp_b[:nocc_b,nocc_b:]  = lib.einsum('ijca,ja->ic', Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('jiac,ja->ic', Y_abab, Y_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('ic,ja,ja->ic', t1_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('ic,ja,ja->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= lib.einsum('jc,ja,ia->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('ijca,jkab,kb->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('ijca,kjba,kb->ic', t1_ccee_bbbb, Y_abab, Y_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ijab,jkab,kc->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= lib.einsum('ia,ja,jc->ic', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('jiac,jkab,kb->ic', t1_ccee_abab, Y_aaaa, Y_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('jiac,jkab,kb->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= lib.einsum('jiab,jkab,kc->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= lib.einsum('jkac,jkab,ib->ic', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('jkca,jkab,ib->ic', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('ic,ja,ja->ic', t2_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] += lib.einsum('ic,ja,ja->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= lib.einsum('jc,ja,ia->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= lib.einsum('ia,ja,jc->ic', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ia,jb,ka,jkbc->ic', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ia,ja,jkcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ia,ja,kjbc,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ia,jb,ka,jkcb->ic', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,ja,ikcb,kb->ic', Y_aa, Y_aa, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,ja,kibc,kb->ic', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ja,jb,ka,kibc->ic', Y_aa, Y_aa, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ja,ka,jibc,kb->ic', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('jc,ja,ikab,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('jc,ja,kiba,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('jc,ka,jb,kiab->ic', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('jc,ka,jb,ikab->ic', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,ja,ikcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] += 1/2 * lib.einsum('ja,ja,kibc,kb->ic', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ja,jb,ikca,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ja,ka,ijcb,kb->ic', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        
        temp_b[nocc_b:,:nocc_b]  = lib.einsum('liab,ib->al', Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('ilba,ib->al', Y_abab, Y_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('la,ib,ib->al', t1_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('la,ib,ib->al', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= lib.einsum('ia,ib,lb->al', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('liab,ijbc,jc->al', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('liab,jicb,jc->al', t1_ccee_bbbb, Y_abab, Y_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] += 1/2 * lib.einsum('libc,ijbc,ja->al', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= lib.einsum('lb,ib,ia->al', t1_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('ilba,ijbc,jc->al', t1_ccee_abab, Y_aaaa, Y_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('ilba,ijbc,jc->al', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= lib.einsum('ilbc,ijbc,ja->al', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= lib.einsum('ijba,ijbc,lc->al', t1_ccee_abab, Y_abab, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += 1/2 * lib.einsum('ijab,ijbc,lc->al', t1_ccee_bbbb, Y_bbbb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('la,ib,ib->al', t2_ce_bb, Y_aa, Y_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] += lib.einsum('la,ib,ib->al', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= lib.einsum('ia,ib,lb->al', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= lib.einsum('lb,ib,ia->al', t2_ce_bb, Y_bb, Y_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('lb,ic,jb,ijca->al', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('lb,ib,ijac,jc->al', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('lb,ib,jica,jc->al', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] += 1/2 * lib.einsum('lb,ic,jb,ijac->al', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += 1/2 * lib.einsum('ib,ib,ljac,jc->al', Y_aa, Y_aa, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += 1/2 * lib.einsum('ib,ib,jlca,jc->al', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('ib,ic,jb,jlca->al', Y_aa, Y_aa, t1_ce_aa, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('ib,jb,ilca,jc->al', Y_aa, Y_aa, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('ia,ib,ljbc,jc->al', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('ia,ib,jlcb,jc->al', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('ia,jb,ic,jlbc->al', Y_bb, Y_aa, t1_ce_bb, t1_ccee_abab, optimize = True)
        temp_b[nocc_b:,:nocc_b] += 1/2 * lib.einsum('ia,jb,ic,ljbc->al', Y_bb, Y_bb, t1_ce_bb, t1_ccee_bbbb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += 1/2 * lib.einsum('ib,ib,ljac,jc->al', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] += 1/2 * lib.einsum('ib,ib,jlca,jc->al', Y_bb, Y_bb, t1_ccee_abab, t1_ce_aa, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('ib,ic,ljab,jc->al', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
        temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('ib,jb,liac,jc->al', Y_bb, Y_bb, t1_ccee_bbbb, t1_ce_bb, optimize = True)
            
# OPDM ADC(2)-X
        if (method == "adc(2)-x"):

            temp_a[:nocc_a,nocc_a:] += 1/4 * lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_a[:nocc_a,nocc_a:] += lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_a[:nocc_a,nocc_a:] += 1/4 * lib.einsum('IC,jkab,jkab->IC', t1_ce_aa, Y_bbbb, Y_bbbb, optimize = True)
            temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('jC,jkab,Ikab->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_a[:nocc_a,nocc_a:] -= lib.einsum('jC,jkab,Ikab->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_a[:nocc_a,nocc_a:] -= 1/2 * lib.einsum('Ia,jkab,jkCb->IC', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_a[:nocc_a,nocc_a:] -= lib.einsum('Ia,jkab,jkCb->IC', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            
            temp_a[nocc_a:,:nocc_a] += 1/4 * lib.einsum('LA,ijbc,ijbc->AL', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_a[nocc_a:,:nocc_a] += lib.einsum('LA,ijbc,ijbc->AL', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_a[nocc_a:,:nocc_a] += 1/4 * lib.einsum('LA,ijbc,ijbc->AL', t1_ce_aa, Y_bbbb, Y_bbbb, optimize = True)
            temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('iA,ijbc,Ljbc->AL', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_a[nocc_a:,:nocc_a] -= lib.einsum('iA,ijbc,Ljbc->AL', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            temp_a[nocc_a:,:nocc_a] -= 1/2 * lib.einsum('Lb,ijbc,ijAc->AL', t1_ce_aa, Y_aaaa, Y_aaaa, optimize = True)
            temp_a[nocc_a:,:nocc_a] -= lib.einsum('Lb,ijbc,ijAc->AL', t1_ce_aa, Y_abab, Y_abab, optimize = True)
            
            temp_b[:nocc_b,nocc_b:] += 1/4 * lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_aaaa, Y_aaaa, optimize = True)
            temp_b[:nocc_b,nocc_b:] += lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_b[:nocc_b,nocc_b:] += 1/4 * lib.einsum('ic,jkab,jkab->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('jc,jkab,ikab->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_b[:nocc_b,nocc_b:] -= lib.einsum('jc,kjab,kiab->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_b[:nocc_b,nocc_b:] -= lib.einsum('ia,jkba,jkbc->ic', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_b[:nocc_b,nocc_b:] -= 1/2 * lib.einsum('ia,jkab,jkcb->ic', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            
            temp_b[nocc_b:,:nocc_b] += 1/4 * lib.einsum('la,ijbc,ijbc->al', t1_ce_bb, Y_aaaa, Y_aaaa, optimize = True)
            temp_b[nocc_b:,:nocc_b] += lib.einsum('la,ijbc,ijbc->al', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_b[nocc_b:,:nocc_b] += 1/4 * lib.einsum('la,ijbc,ijbc->al', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('ia,ijbc,ljbc->al', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)
            temp_b[nocc_b:,:nocc_b] -= lib.einsum('ia,jibc,jlbc->al', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_b[nocc_b:,:nocc_b] -= lib.einsum('lb,ijcb,ijca->al', t1_ce_bb, Y_abab, Y_abab, optimize = True)
            temp_b[nocc_b:,:nocc_b] -= 1/2 * lib.einsum('lb,ijbc,ijac->al', t1_ce_bb, Y_bbbb, Y_bbbb, optimize = True)

        opdm_a = np.append(opdm_a,temp_a)
        opdm_b = np.append(opdm_b,temp_b)

        #pdm = (opdm_a,opdm_b, None ,None ,None)

       # norm = np.linalg.norm(temp_a - temp_a.transpose(1,0))
       # print("total OPDM_a singles norm for Hermiticity",norm)
       # print("opdm trace_a",np.einsum('pp',temp_a))
       # 
       # 
       # norm = np.linalg.norm(temp_b - temp_b.transpose(1,0))
       # print("total OPDM_b singles norm for Hermiticity",norm)
       # print("opdm trace_b",np.einsum('pp',temp_b))
        
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

#TPDM ADC(2)

        if adc.tpdm is True:
#####################################################################################################################################################################

###########################Below things are fine####################################################################################################################

            temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a]  = 1/2 * lib.einsum('IJab,KLab->IJKL', Y_aaaa, Y_aaaa, optimize = True)
            
            temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= 1/2 * lib.einsum('Jmab,Lmab->JL', Y_aaaa, Y_aaaa, optimize = True)
            temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('Ja,La->JL', Y_aa, Y_aa, optimize = True)
            temp_2a[occ_list_a,:nocc_a,occ_list_a,:nocc_a] -= lib.einsum('Jmab,Lmab->JL', Y_abab, Y_abab, optimize = True)
            
            temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += 1/2 * lib.einsum('Jmab,Kmab->JK', Y_aaaa, Y_aaaa, optimize = True)
            temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('Ja,Ka->JK', Y_aa, Y_aa, optimize = True)
            temp_2a[occ_list_a,:nocc_a,:nocc_a,occ_list_a] += lib.einsum('Jmab,Kmab->JK', Y_abab, Y_abab, optimize = True)
            
            block_1 = np.zeros((nocc_a,nocc_a,nocc_a,nocc_a))
            block_1[:,occ_list_a,:,occ_list_a] = 1/2 * lib.einsum('Imab,Lmab->IL', Y_aaaa, Y_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('Ia,La->IL', Y_aa, Y_aa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('Imab,Lmab->IL', Y_abab, Y_abab, optimize = True)
            temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += block_1.transpose(0,1,3,2)
            
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
            block_1[:,occ_list_a,:,occ_list_a] = lib.einsum('Ia,Lmab,mb->IL',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('Ia,Lmab,mb->IL',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('La,Imab,mb->IL',  t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('La,Imab,mb->IL',  t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += block_1.transpose(0,1,3,2)
            
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
            block_1[:,occ_list_a,:,occ_list_a] = -1/4 * lib.einsum('Ia,ma,mnbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= 1/2 * lib.einsum('Ia,ma,mb,Lb->IL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= 1/2 * lib.einsum('Ia,ma,mnbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('Ia,mb,Lnac,mnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('Ia,mb,Lnac,mnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('Ia,mb,Lnac,nmcb->IL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('Ia,mb,Lnac,mnbc->IL',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= 1/4 * lib.einsum('La,ma,mnbc,Inbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= 1/2 * lib.einsum('La,ma,mb,Ib->IL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= 1/2 * lib.einsum('La,ma,mnbc,Inbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('La,mb,Inac,mnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('La,mb,Inac,mnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('La,mb,Inac,nmcb->IL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('La,mb,Inac,mnbc->IL',  Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('ma,ma,Inbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('ma,ma,Ib,Lb->IL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('ma,ma,Inbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= lib.einsum('ma,mb,Ia,Lb->IL',  Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= lib.einsum('ma,mb,Inac,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= lib.einsum('ma,mb,Inac,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= 1/2 * lib.einsum('ma,na,Imbc,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('ma,nb,Imac,Lnbc->IL',  Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= lib.einsum('ma,nb,Imac,Lncb->IL',  Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += 1/2 * lib.einsum('ma,ma,Inbc,Lnbc->IL',  Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('ma,ma,Ib,Lb->IL',  Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('ma,ma,Inbc,Lnbc->IL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= lib.einsum('ma,mb,Inca,Lncb->IL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= lib.einsum('ma,nb,Imca,Lnbc->IL',  Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] -= lib.einsum('ma,na,Imbc,Lnbc->IL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            block_1[:,occ_list_a,:,occ_list_a] += lib.einsum('ma,nb,Imca,Lncb->IL',  Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2a[:nocc_a,:nocc_a,:nocc_a,:nocc_a] += block_1.transpose(0,1,3,2)
            
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
            
            
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:]  = lib.einsum('IC,JkDa,ka->IJCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IC,JkDa,ka->IJCD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JC,IkDa,ka->IJCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JC,IkDa,ka->IJCD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('kC,ka,IJDa->IJCD', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('ID,JkCa,ka->IJCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('ID,JkCa,ka->IJCD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JD,IkCa,ka->IJCD', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JD,IkCa,ka->IJCD', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('kD,ka,IJCa->IJCD', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IJCD,ka,ka->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IJCD,ka,ka->IJCD', t1_ccee_aaaa, Y_bb, Y_bb, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IJCa,ka,kD->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IJDa,ka,kC->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkCD,ka,Ja->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IkCa,ka,JD->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkDa,ka,JC->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('Ia,ka,JkCD->IJCD', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IkCa,ka,JD->IJCD', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkDa,ka,JC->IJCD', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkCD,ka,Ia->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JkCa,ka,ID->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkDa,ka,IC->IJCD', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('Ja,ka,IkCD->IJCD', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JkCa,ka,ID->IJCD', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkDa,ka,IC->IJCD', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IJCD,ka,ka->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IJCD,ka,ka->IJCD', t2_ccee_aaaa, Y_bb, Y_bb, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IJCa,ka,kD->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IJDa,ka,kC->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkCD,ka,Ja->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IkCa,ka,JD->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkDa,ka,JC->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IkCa,ka,JD->IJCD', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkDa,ka,JC->IJCD', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkCD,ka,Ia->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JkCa,ka,ID->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkDa,ka,IC->IJCD', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JkCa,ka,ID->IJCD', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkDa,ka,IC->IJCD', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IC,ka,kD,Ja->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('ID,ka,kC,Ja->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('Ia,ka,kC,JD->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('Ia,ka,kD,JC->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JC,ka,kD,Ia->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JD,ka,kC,Ia->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('Ja,ka,kC,ID->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('Ja,ka,kD,IC->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('kC,ka,Ia,JD->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('kC,ka,Ja,ID->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('kD,ka,Ia,JC->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('kD,ka,Ja,IC->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('ka,ka,IC,JD->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('ka,ka,JC,ID->IJCD', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('ka,ka,IC,JD->IJCD', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('ka,ka,JC,ID->IJCD', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            
            
            
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a]  = lib.einsum('KA,LiBc,ic->ABKL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KA,LiBc,ic->ABKL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LA,KiBc,ic->ABKL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LA,KiBc,ic->ABKL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('iA,ic,KLBc->ABKL', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KB,LiAc,ic->ABKL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KB,LiAc,ic->ABKL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LB,KiAc,ic->ABKL', t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LB,KiAc,ic->ABKL', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('iB,ic,KLAc->ABKL', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KLAB,ic,ic->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KLAB,ic,ic->ABKL', t1_ccee_aaaa, Y_bb, Y_bb, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KLAc,ic,iB->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KLBc,ic,iA->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiAB,ic,Lc->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KiAc,ic,LB->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiBc,ic,LA->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('Kc,ic,LiAB->ABKL', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KiAc,ic,LB->ABKL', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiBc,ic,LA->ABKL', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiAB,ic,Kc->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LiAc,ic,KB->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiBc,ic,KA->ABKL', t1_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('Lc,ic,KiAB->ABKL', t1_ce_aa, Y_aa, Y_aaaa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LiAc,ic,KB->ABKL', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiBc,ic,KA->ABKL', t1_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KLAB,ic,ic->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KLAB,ic,ic->ABKL', t2_ccee_aaaa, Y_bb, Y_bb, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KLAc,ic,iB->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KLBc,ic,iA->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiAB,ic,Lc->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KiAc,ic,LB->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiBc,ic,LA->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KiAc,ic,LB->ABKL', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiBc,ic,LA->ABKL', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiAB,ic,Kc->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LiAc,ic,KB->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiBc,ic,KA->ABKL', t2_ccee_aaaa, Y_aa, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LiAc,ic,KB->ABKL', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiBc,ic,KA->ABKL', t2_ccee_abab, Y_bb, Y_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KA,ic,iB,Lc->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KB,ic,iA,Lc->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('Kc,ic,iA,LB->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('Kc,ic,iB,LA->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LA,ic,iB,Kc->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LB,ic,iA,Kc->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('Lc,ic,iA,KB->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('Lc,ic,iB,KA->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('iA,ic,Kc,LB->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('iA,ic,Lc,KB->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('iB,ic,Kc,LA->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('iB,ic,Lc,KA->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('ic,ic,KA,LB->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('ic,ic,LA,KB->ABKL', Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('ic,ic,KA,LB->ABKL', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('ic,ic,LA,KB->ABKL', Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            
            
            temp =- lib.einsum('ID,KA->IAKD', Y_aa, Y_aa, optimize = True)
            temp -= lib.einsum('IjDb,KjAb->IAKD', Y_aaaa, Y_aaaa, optimize = True)
            temp -= lib.einsum('IjDb,KjAb->IAKD', Y_abab, Y_abab, optimize = True)
            temp += lib.einsum('IK,jA,jD->IAKD', np.identity(nocc_a), Y_aa, Y_aa, optimize = True)
            temp += 1/2 * lib.einsum('IK,jlAb,jlDb->IAKD', np.identity(nocc_a), Y_aaaa, Y_aaaa, optimize = True)
            temp += lib.einsum('IK,jlAb,jlDb->IAKD', np.identity(nocc_a), Y_abab, Y_abab, optimize = True)
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
            temp += lib.einsum('IK,jA,jlDb,lb->IAKD', np.identity(nocc_a), t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp += lib.einsum('IK,jA,jlDb,lb->IAKD', np.identity(nocc_a), t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp += lib.einsum('IK,jD,jlAb,lb->IAKD', np.identity(nocc_a), t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            temp += lib.einsum('IK,jD,jlAb,lb->IAKD', np.identity(nocc_a), t1_ce_aa, Y_abab, Y_bb, optimize = True)
            temp -= 1/2 * lib.einsum('IK,jA,jb,lb,lD->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp -= 1/4 * lib.einsum('IK,jA,jb,lmbc,lmDc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp -= 1/2 * lib.einsum('IK,jA,jb,lmbc,lmDc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('IK,jA,lb,jmDc,lmbc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp += 1/2 * lib.einsum('IK,jA,lb,jmDc,lmbc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('IK,jA,lb,jmDc,mlcb->IAKD', np.identity(nocc_a), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('IK,jA,lb,jmDc,lmbc->IAKD', np.identity(nocc_a), Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            temp -= 1/2 * lib.einsum('IK,jD,jb,lb,lA->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp -= 1/4 * lib.einsum('IK,jD,jb,lmbc,lmAc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp -= 1/2 * lib.einsum('IK,jD,jb,lmbc,lmAc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('IK,jD,lb,jmAc,lmbc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp += 1/2 * lib.einsum('IK,jD,lb,jmAc,lmbc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('IK,jD,lb,jmAc,mlcb->IAKD', np.identity(nocc_a), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('IK,jD,lb,jmAc,lmbc->IAKD', np.identity(nocc_a), Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            temp += lib.einsum('IK,jb,jb,lA,lD->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp += 1/2 * lib.einsum('IK,jb,jb,lmAc,lmDc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp += lib.einsum('IK,jb,jb,lmAc,lmDc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp -= 1/2 * lib.einsum('IK,jb,jc,lmAb,lmDc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp -= lib.einsum('IK,jb,lb,jA,lD->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            temp -= lib.einsum('IK,jb,lb,jmAc,lmDc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp -= lib.einsum('IK,jb,lb,jmAc,lmDc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += lib.einsum('IK,jb,lc,jmAb,lmDc->IAKD', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp -= lib.einsum('IK,jb,lc,jmAb,mlDc->IAKD', np.identity(nocc_a), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            temp -= lib.einsum('IK,jb,lc,jmDb,mlAc->IAKD', np.identity(nocc_a), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            temp += lib.einsum('IK,jb,jb,lA,lD->IAKD', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            temp += 1/2 * lib.einsum('IK,jb,jb,lmAc,lmDc->IAKD', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            temp += lib.einsum('IK,jb,jb,lmAc,lmDc->IAKD', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp -= lib.einsum('IK,jb,jc,lmAb,lmDc->IAKD', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp -= lib.einsum('IK,jb,lb,mjAc,mlDc->IAKD', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += lib.einsum('IK,jb,lc,mjAb,mlDc->IAKD', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            


            temp_2a[:nocc_a,nocc_a:,:nocc_a,nocc_a:] = temp
            temp_2a[:nocc_a,nocc_a:,nocc_a:,:nocc_a] = -temp.transpose(0,1,3,2)
            temp_2a[nocc_a:,:nocc_a,nocc_a:,:nocc_a] = temp.transpose(1,0,3,2)
            temp_2a[nocc_a:,:nocc_a,:nocc_a,nocc_a:] = -temp.transpose(1,0,2,3)


            norm = np.linalg.norm(temp_2a - temp_2a.transpose(2,3,0,1))
            print("Total TPDM_a  Hermiticity",norm)
            norm = np.linalg.norm(temp_2a + temp_2a.transpose(1,0,2,3))
            print("Total TPDM_a bra symmetry",norm)
            norm = np.linalg.norm(temp_2a + temp_2a.transpose(0,1,3,2))
            print("Total TPDM_a ket symmetry",norm)
            print("TPDM_a trace",np.einsum('pqpq',temp_2a))
            print("") 














            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b]  = 1/2 * lib.einsum('ijab,klab->ijkl', Y_bbbb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,jmab,lmab->ijkl', np.identity(nocc_b), Y_bbbb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ja,la->ijkl', np.identity(nocc_b), Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,mjab,mlab->ijkl', np.identity(nocc_b), Y_abab, Y_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,jmab,kmab->ijkl', np.identity(nocc_b), Y_bbbb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ja,ka->ijkl', np.identity(nocc_b), Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,mjab,mkab->ijkl', np.identity(nocc_b), Y_abab, Y_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,imab,lmab->ijkl', np.identity(nocc_b), Y_bbbb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ia,la->ijkl', np.identity(nocc_b), Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,miab,mlab->ijkl', np.identity(nocc_b), Y_abab, Y_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,imab,kmab->ijkl', np.identity(nocc_b), Y_bbbb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ia,ka->ijkl', np.identity(nocc_b), Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,miab,mkab->ijkl', np.identity(nocc_b), Y_abab, Y_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ia,klab,jb->ijkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ja,klab,ib->ijkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ka,ijab,lb->ijkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('la,ijab,kb->ijkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/4 * lib.einsum('mnab,mnab,ik,jl->ijkl', Y_aaaa, Y_aaaa, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/4 * lib.einsum('mnab,mnab,il,jk->ijkl', Y_aaaa, Y_aaaa, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ma,ma,ik,jl->ijkl', Y_aa, Y_aa, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ma,ma,il,jk->ijkl', Y_aa, Y_aa, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('mnab,mnab,ik,jl->ijkl', Y_abab, Y_abab, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('mnab,mnab,il,jk->ijkl', Y_abab, Y_abab, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/4 * lib.einsum('mnab,mnab,ik,jl->ijkl', Y_bbbb, Y_bbbb, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/4 * lib.einsum('mnab,mnab,il,jk->ijkl', Y_bbbb, Y_bbbb, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ma,ma,ik,jl->ijkl', Y_bb, Y_bb, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ma,ma,il,jk->ijkl', Y_bb, Y_bb, np.identity(nocc_b), np.identity(nocc_b), optimize = True)
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
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ja,lmab,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ja,mlba,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,la,jmab,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,la,mjba,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ja,kmab,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ja,mkba,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ka,jmab,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ka,mjba,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ia,lmab,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ia,mlba,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,la,imab,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,la,miba,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ia,kmab,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ia,mkba,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ka,imab,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ka,miba,mb->ijkl', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,ja,mb,lnac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,ja,mb,nlca,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/4 * lib.einsum('ik,ja,ma,mnbc,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ik,ja,ma,mb,lb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ik,ja,ma,nmbc,nlbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,ja,mb,lnac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,ja,mb,nlca,nmcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,la,mb,jnac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,la,mb,njca,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/4 * lib.einsum('ik,la,ma,mnbc,jnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ik,la,ma,mb,jb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ik,la,ma,nmbc,njbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,la,mb,jnac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,la,mb,njca,nmcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,ma,ma,jnbc,lnbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ma,ma,jb,lb->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ma,ma,njbc,nlbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ik,ma,mb,njac,nlbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ik,ma,na,mjbc,nlbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ma,nb,mjac,nlbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ik,ma,ma,jnbc,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ma,ma,jb,lb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ma,ma,njbc,nlbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ik,ma,mb,ja,lb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ik,ma,mb,jnac,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ik,ma,mb,njca,nlcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ik,ma,nb,jmac,nlbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('ik,ma,nb,lmac,njbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ik,ma,na,jmbc,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('ik,ma,nb,jmac,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ja,mb,knac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ja,mb,nkca,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/4 * lib.einsum('il,ja,ma,mnbc,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('il,ja,ma,mb,kb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('il,ja,ma,nmbc,nkbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ja,mb,knac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ja,mb,nkca,nmcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ka,mb,jnac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ka,mb,njca,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/4 * lib.einsum('il,ka,ma,mnbc,jnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('il,ka,ma,mb,jb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('il,ka,ma,nmbc,njbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ka,mb,jnac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ka,mb,njca,nmcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ma,ma,jnbc,knbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ma,ma,jb,kb->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ma,ma,njbc,nkbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('il,ma,mb,njac,nkbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('il,ma,na,mjbc,nkbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ma,nb,mjac,nkbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('il,ma,ma,jnbc,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ma,ma,jb,kb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ma,ma,njbc,nkbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('il,ma,mb,ja,kb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('il,ma,mb,jnac,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('il,ma,mb,njca,nkcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('il,ma,nb,jmac,nkbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('il,ma,nb,kmac,njbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('il,ma,na,jmbc,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('il,ma,nb,jmac,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,ia,mb,lnac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,ia,mb,nlca,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/4 * lib.einsum('jk,ia,ma,mnbc,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jk,ia,ma,mb,lb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jk,ia,ma,nmbc,nlbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,ia,mb,lnac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,ia,mb,nlca,nmcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,la,mb,inac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,la,mb,nica,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/4 * lib.einsum('jk,la,ma,mnbc,inbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jk,la,ma,mb,ib->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jk,la,ma,nmbc,nibc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,la,mb,inac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,la,mb,nica,nmcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,ma,ma,inbc,lnbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ma,ma,ib,lb->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ma,ma,nibc,nlbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jk,ma,mb,niac,nlbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jk,ma,na,mibc,nlbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ma,nb,miac,nlbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jk,ma,ma,inbc,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ma,ma,ib,lb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ma,ma,nibc,nlbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jk,ma,mb,ia,lb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jk,ma,mb,inac,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jk,ma,mb,nica,nlcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jk,ma,nb,imac,nlbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jk,ma,nb,lmac,nibc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jk,ma,na,imbc,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jk,ma,nb,imac,lnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ia,mb,knac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ia,mb,nkca,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/4 * lib.einsum('jl,ia,ma,mnbc,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jl,ia,ma,mb,kb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jl,ia,ma,nmbc,nkbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ia,mb,knac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ia,mb,nkca,nmcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ka,mb,inac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ka,mb,nica,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/4 * lib.einsum('jl,ka,ma,mnbc,inbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jl,ka,ma,mb,ib->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jl,ka,ma,nmbc,nibc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ka,mb,inac,mnbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ka,mb,nica,nmcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ma,ma,inbc,knbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ma,ma,ib,kb->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ma,ma,nibc,nkbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jl,ma,mb,niac,nkbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jl,ma,na,mibc,nkbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ma,nb,miac,nkbc->ijkl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('jl,ma,ma,inbc,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ma,ma,ib,kb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ma,ma,nibc,nkbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jl,ma,mb,ia,kb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jl,ma,mb,inac,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jl,ma,mb,nica,nkcb->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jl,ma,nb,imac,nkbc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += lib.einsum('jl,ma,nb,kmac,nibc->ijkl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] += 1/2 * lib.einsum('jl,ma,na,imbc,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,:nocc_b,:nocc_b] -= lib.einsum('jl,ma,nb,imac,knbc->ijkl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            
            
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
            
            
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:]  = lib.einsum('ic,jkda,ka->ijcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ic,kjad,ka->ijcd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('jc,ikda,ka->ijcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('jc,kiad,ka->ijcd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kc,ka,ijda->ijcd', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('id,jkca,ka->ijcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('id,kjac,ka->ijcd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jd,ikca,ka->ijcd', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jd,kiac,ka->ijcd', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kd,ka,ijca->ijcd', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ijcd,ka,ka->ijcd', t1_ccee_bbbb, Y_aa, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ijcd,ka,ka->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ijca,ka,kd->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ijda,ka,kc->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ikcd,ka,ja->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ikca,ka,jd->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ikda,ka,jc->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ia,ka,jkcd->ijcd', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jkcd,ka,ia->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('jkca,ka,id->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jkda,ka,ic->ijcd', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ja,ka,ikcd->ijcd', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kiac,ka,jd->ijcd', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kiad,ka,jc->ijcd', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kjac,ka,id->ijcd', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kjad,ka,ic->ijcd', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ijcd,ka,ka->ijcd', t2_ccee_bbbb, Y_aa, Y_aa, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ijcd,ka,ka->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ijca,ka,kd->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ijda,ka,kc->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ikcd,ka,ja->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ikca,ka,jd->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ikda,ka,jc->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jkcd,ka,ia->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('jkca,ka,id->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jkda,ka,ic->ijcd', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kiac,ka,jd->ijcd', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kiad,ka,jc->ijcd', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kjac,ka,id->ijcd', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kjad,ka,ic->ijcd', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ic,ka,kd,ja->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('id,ka,kc,ja->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ia,ka,kc,jd->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ia,ka,kd,jc->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jc,ka,kd,ia->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('jd,ka,kc,ia->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ja,ka,kc,id->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ja,ka,kd,ic->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ka,ka,ic,jd->ijcd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ka,ka,jc,id->ijcd', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kc,ka,ia,jd->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kc,ka,ja,id->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kd,ka,ia,jc->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kd,ka,ja,ic->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ka,ka,ic,jd->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ka,ka,jc,id->ijcd', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            
            
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b]  = lib.einsum('ka,libc,ic->abkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ka,ilcb,ic->abkl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('la,kibc,ic->abkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('la,ikcb,ic->abkl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ia,ic,klbc->abkl', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kb,liac,ic->abkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kb,ilca,ic->abkl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('lb,kiac,ic->abkl', t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('lb,ikca,ic->abkl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ib,ic,klac->abkl', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('klab,ic,ic->abkl', t1_ccee_bbbb, Y_aa, Y_aa, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('klab,ic,ic->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('klac,ic,ib->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('klbc,ic,ia->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kiab,ic,lc->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('kiac,ic,lb->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kibc,ic,la->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('kc,ic,liab->abkl', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('liab,ic,kc->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('liac,ic,kb->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('libc,ic,ka->abkl', t1_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('lc,ic,kiab->abkl', t1_ce_bb, Y_bb, Y_bbbb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ikca,ic,lb->abkl', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ikcb,ic,la->abkl', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ilca,ic,kb->abkl', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ilcb,ic,ka->abkl', t1_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('klab,ic,ic->abkl', t2_ccee_bbbb, Y_aa, Y_aa, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('klab,ic,ic->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('klac,ic,ib->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('klbc,ic,ia->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kiab,ic,lc->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('kiac,ic,lb->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kibc,ic,la->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('liab,ic,kc->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('liac,ic,kb->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('libc,ic,ka->abkl', t2_ccee_bbbb, Y_bb, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ikca,ic,lb->abkl', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ikcb,ic,la->abkl', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ilca,ic,kb->abkl', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ilcb,ic,ka->abkl', t2_ccee_abab, Y_aa, Y_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ka,ic,ib,lc->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('kb,ic,ia,lc->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kc,ic,ia,lb->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('kc,ic,ib,la->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('la,ic,ib,kc->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('lb,ic,ia,kc->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('lc,ic,ia,kb->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('lc,ic,ib,ka->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ic,ic,ka,lb->abkl', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ic,ic,la,kb->abkl', Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ia,ic,kc,lb->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ia,ic,lc,kb->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ib,ic,kc,la->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ib,ic,lc,ka->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ic,ic,ka,lb->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ic,ic,la,kb->abkl', Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
                
                
            temp =- lib.einsum('id,ka->iakd', Y_bb, Y_bb, optimize = True)
            temp -= lib.einsum('ijdb,kjab->iakd', Y_bbbb, Y_bbbb, optimize = True)
            temp -= lib.einsum('jibd,jkba->iakd', Y_abab, Y_abab, optimize = True)
            temp += lib.einsum('ik,jlba,jlbd->iakd', np.identity(nocc_b), Y_abab, Y_abab, optimize = True)
            temp += lib.einsum('ik,ja,jd->iakd', np.identity(nocc_b), Y_bb, Y_bb, optimize = True)
            temp += 1/2 * lib.einsum('ik,jlab,jldb->iakd', np.identity(nocc_b), Y_bbbb, Y_bbbb, optimize = True)
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
            temp += lib.einsum('ik,ja,jldb,lb->iakd', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp += lib.einsum('ik,ja,ljbd,lb->iakd', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp += lib.einsum('ik,jd,jlab,lb->iakd', np.identity(nocc_b), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            temp += lib.einsum('ik,jd,ljba,lb->iakd', np.identity(nocc_b), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            temp += lib.einsum('ik,jb,jb,la,ld->iakd', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            temp += lib.einsum('ik,jb,jb,lmca,lmcd->iakd', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('ik,jb,jb,lmac,lmdc->iakd', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp -= lib.einsum('ik,jb,jc,lmba,lmcd->iakd', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp -= lib.einsum('ik,jb,lb,jmca,lmcd->iakd', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += lib.einsum('ik,jb,lc,jmba,lmcd->iakd', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp -= 1/2 * lib.einsum('ik,ja,jb,lb,ld->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp -= 1/2 * lib.einsum('ik,ja,jb,lmcb,lmcd->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp -= 1/4 * lib.einsum('ik,ja,jb,lmbc,lmdc->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp += 1/2 * lib.einsum('ik,ja,lb,jmdc,lmbc->iakd', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('ik,ja,lb,mjcd,lmbc->iakd', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp += 1/2 * lib.einsum('ik,ja,lb,jmdc,lmbc->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp += 1/2 * lib.einsum('ik,ja,lb,mjcd,mlcb->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp -= 1/2 * lib.einsum('ik,jd,jb,lb,la->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp -= 1/2 * lib.einsum('ik,jd,jb,lmcb,lmca->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp -= 1/4 * lib.einsum('ik,jd,jb,lmbc,lmac->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp += 1/2 * lib.einsum('ik,jd,lb,jmac,lmbc->iakd', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('ik,jd,lb,mjca,lmbc->iakd', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            temp += 1/2 * lib.einsum('ik,jd,lb,jmac,lmbc->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp += 1/2 * lib.einsum('ik,jd,lb,mjca,mlcb->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += lib.einsum('ik,jb,jb,la,ld->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp += lib.einsum('ik,jb,jb,lmca,lmcd->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += 1/2 * lib.einsum('ik,jb,jb,lmac,lmdc->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp -= 1/2 * lib.einsum('ik,jb,jc,lmab,lmdc->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp -= lib.einsum('ik,jb,lc,jmab,lmcd->iakd', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp -= lib.einsum('ik,jb,lc,jmdb,lmca->iakd', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            temp -= lib.einsum('ik,jb,lb,ja,ld->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            temp -= lib.einsum('ik,jb,lb,jmac,lmdc->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            temp -= lib.einsum('ik,jb,lb,mjca,mlcd->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            temp += lib.einsum('ik,jb,lc,jmab,lmdc->iakd', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
                
            temp_2b[:nocc_b,nocc_b:,:nocc_b,nocc_b:] = temp
            temp_2b[:nocc_b,nocc_b:,nocc_b:,:nocc_b] = -temp.transpose(0,1,3,2)
            temp_2b[nocc_b:,:nocc_b,nocc_b:,:nocc_b] = temp.transpose(1,0,3,2)
            temp_2b[nocc_b:,:nocc_b,:nocc_b,nocc_b:] = -temp.transpose(1,0,2,3)
    
#TPDM ADC(2)-X
            if (method == "adc(2)-x"):

                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += 1/4 * lib.einsum('IJCD,klab,klab->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IJCD,klab,klab->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += 1/4 * lib.einsum('IJCD,klab,klab->IJCD', t1_ccee_aaaa, Y_bbbb, Y_bbbb, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= 1/2 * lib.einsum('IJCa,klab,klDb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IJCa,klab,klDb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += 1/2 * lib.einsum('IJDa,klab,klCb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IJDa,klab,klCb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += 1/4 * lib.einsum('IJab,klab,klCD->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= 1/2 * lib.einsum('IkCD,klab,Jlab->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkCD,klab,Jlab->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IkCa,klab,JlDb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IkCa,klab,JlDb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkDa,klab,JlCb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkDa,klab,JlCb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= 1/2 * lib.einsum('Ikab,klab,JlCD->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IkCa,klab,JlDb->IJCD', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('IkCa,lkba,JlDb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkDa,klab,JlCb->IJCD', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('IkDa,lkba,JlCb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('Ikab,lkab,JlCD->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += 1/2 * lib.einsum('JkCD,klab,Ilab->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkCD,klab,Ilab->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JkCa,klab,IlDb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JkCa,klab,IlDb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkDa,klab,IlCb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkDa,klab,IlCb->IJCD', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += 1/2 * lib.einsum('Jkab,klab,IlCD->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JkCa,klab,IlDb->IJCD', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('JkCa,lkba,IlDb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkDa,klab,IlCb->IJCD', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('JkDa,lkba,IlCb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('Jkab,lkab,IlCD->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += 1/4 * lib.einsum('klCD,klab,IJab->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= 1/2 * lib.einsum('klCa,klab,IJDb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += 1/2 * lib.einsum('klDa,klab,IJCb->IJCD', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] += lib.einsum('klCa,klba,IJDb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[:nocc_a,:nocc_a,nocc_a:,nocc_a:] -= lib.einsum('klDa,klba,IJCb->IJCD', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += 1/4 * lib.einsum('KLAB,ijcd,ijcd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KLAB,ijcd,ijcd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += 1/4 * lib.einsum('KLAB,ijcd,ijcd->ABKL', t1_ccee_aaaa, Y_bbbb, Y_bbbb, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= 1/2 * lib.einsum('KLAc,ijcd,ijBd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KLAc,ijcd,ijBd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += 1/2 * lib.einsum('KLBc,ijcd,ijAd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KLBc,ijcd,ijAd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += 1/4 * lib.einsum('KLcd,ijcd,ijAB->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= 1/2 * lib.einsum('KiAB,ijcd,Ljcd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiAB,ijcd,Ljcd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KiAc,ijcd,LjBd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KiAc,ijcd,LjBd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiBc,ijcd,LjAd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiBc,ijcd,LjAd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= 1/2 * lib.einsum('Kicd,ijcd,LjAB->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KiAc,ijcd,LjBd->ABKL', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('KiAc,jidc,LjBd->ABKL', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiBc,ijcd,LjAd->ABKL', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('KiBc,jidc,LjAd->ABKL', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('Kicd,jicd,LjAB->ABKL', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += 1/2 * lib.einsum('LiAB,ijcd,Kjcd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiAB,ijcd,Kjcd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LiAc,ijcd,KjBd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LiAc,ijcd,KjBd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiBc,ijcd,KjAd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiBc,ijcd,KjAd->ABKL', t1_ccee_aaaa, Y_abab, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += 1/2 * lib.einsum('Licd,ijcd,KjAB->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LiAc,ijcd,KjBd->ABKL', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('LiAc,jidc,KjBd->ABKL', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiBc,ijcd,KjAd->ABKL', t1_ccee_abab, Y_bbbb, Y_abab, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('LiBc,jidc,KjAd->ABKL', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('Licd,jicd,KjAB->ABKL', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += 1/4 * lib.einsum('ijAB,ijcd,KLcd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= 1/2 * lib.einsum('ijAc,ijcd,KLBd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += 1/2 * lib.einsum('ijBc,ijcd,KLAd->ABKL', t1_ccee_aaaa, Y_aaaa, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] += lib.einsum('ijAc,ijdc,KLBd->ABKL', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                temp_2a[nocc_a:,nocc_a:,:nocc_a,:nocc_a] -= lib.einsum('ijBc,ijdc,KLAd->ABKL', t1_ccee_abab, Y_abab, Y_aaaa, optimize = True)
                
                
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += 1/4 * lib.einsum('ijcd,klab,klab->ijcd', t1_ccee_bbbb, Y_aaaa, Y_aaaa, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ijcd,klab,klab->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += 1/4 * lib.einsum('ijcd,klab,klab->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ijca,klba,klbd->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ijca,klab,kldb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ijda,klba,klbc->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += 1/2 * lib.einsum('ijda,klab,klcb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += 1/4 * lib.einsum('ijab,klab,klcd->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ikcd,klab,jlab->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ikcd,lkab,ljab->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ikca,klab,jldb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('ikca,lkba,ljbd->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ikda,klab,jlcb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('ikda,lkba,ljbc->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= 1/2 * lib.einsum('ikab,klab,jlcd->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += 1/2 * lib.einsum('jkcd,klab,ilab->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jkcd,lkab,liab->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('jkca,klab,ildb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('jkca,lkba,libd->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jkda,klab,ilcb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('jkda,lkba,libc->ijcd', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += 1/2 * lib.einsum('jkab,klab,ilcd->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kiac,klab,ljbd->ijcd', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kiac,klab,jldb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kiad,klab,ljbc->ijcd', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kiad,klab,jlcb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kiab,klab,jlcd->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kjac,klab,libd->ijcd', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kjac,klab,ildb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kjad,klab,libc->ijcd', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('kjad,klab,ilcb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('kjab,klab,ilcd->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += lib.einsum('klac,klab,ijdb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= lib.einsum('klad,klab,ijcb->ijcd', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += 1/4 * lib.einsum('klcd,klab,ijab->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] -= 1/2 * lib.einsum('klca,klab,ijdb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[:nocc_b,:nocc_b,nocc_b:,nocc_b:] += 1/2 * lib.einsum('klda,klab,ijcb->ijcd', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                    
                    
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += 1/4 * lib.einsum('klab,ijcd,ijcd->abkl', t1_ccee_bbbb, Y_aaaa, Y_aaaa, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('klab,ijcd,ijcd->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += 1/4 * lib.einsum('klab,ijcd,ijcd->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('klac,ijdc,ijdb->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('klac,ijcd,ijbd->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('klbc,ijdc,ijda->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += 1/2 * lib.einsum('klbc,ijcd,ijad->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += 1/4 * lib.einsum('klcd,ijcd,ijab->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('kiab,ijcd,ljcd->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kiab,jicd,jlcd->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('kiac,ijcd,ljbd->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('kiac,jidc,jldb->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kibc,ijcd,ljad->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('kibc,jidc,jlda->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('kicd,ijcd,ljab->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += 1/2 * lib.einsum('liab,ijcd,kjcd->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('liab,jicd,jkcd->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('liac,ijcd,kjbd->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('liac,jidc,jkdb->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('libc,ijcd,kjad->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('libc,jidc,jkda->abkl', t1_ccee_bbbb, Y_abab, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += 1/2 * lib.einsum('licd,ijcd,kjab->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ikca,ijcd,jldb->abkl', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ikca,ijcd,ljbd->abkl', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ikcb,ijcd,jlda->abkl', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ikcb,ijcd,ljad->abkl', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ikcd,ijcd,ljab->abkl', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ilca,ijcd,jkdb->abkl', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ilca,ijcd,kjbd->abkl', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ilcb,ijcd,jkda->abkl', t1_ccee_abab, Y_aaaa, Y_abab, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ilcb,ijcd,kjad->abkl', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ilcd,ijcd,kjab->abkl', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += lib.einsum('ijca,ijcd,klbd->abkl', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= lib.einsum('ijcb,ijcd,klad->abkl', t1_ccee_abab, Y_abab, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += 1/4 * lib.einsum('ijab,ijcd,klcd->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] -= 1/2 * lib.einsum('ijac,ijcd,klbd->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)
                temp_2b[nocc_b:,nocc_b:,:nocc_b,:nocc_b] += 1/2 * lib.einsum('ijbc,ijcd,klad->abkl', t1_ccee_bbbb, Y_bbbb, Y_bbbb, optimize = True)

    
           # norm = np.linalg.norm(temp_2b - temp_2b.transpose(2,3,0,1))
           # print("Total TPDM_b  Hermiticity",norm)
           # norm = np.linalg.norm(temp_2b + temp_2b.transpose(1,0,2,3))
           # print("Total TPDM_b bra symmetry",norm)
           # norm = np.linalg.norm(temp_2b + temp_2b.transpose(0,1,3,2))
           # print("Total TPDM_b ket symmetry",norm)
           # print("TPDM trace",np.einsum('pqpq',temp_2b))
           # print("") 
        tpdm_a = np.append(tpdm_a,temp_2a)

        tpdm_b = np.append(tpdm_b,temp_2b)

    
        if adc.spin_c is True or adc.tpdm is True:

            IjKl  = lib.einsum('Ijab,Klab->IjKl', Y_abab, Y_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,jmab,lmab->IjKl', np.identity(nocc_a), Y_bbbb, Y_bbbb, optimize = True)
            IjKl -= lib.einsum('IK,ja,la->IjKl', np.identity(nocc_a), Y_bb, Y_bb, optimize = True)
            IjKl -= lib.einsum('IK,mjab,mlab->IjKl', np.identity(nocc_a), Y_abab, Y_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Imab,Kmab->IjKl', np.identity(nocc_b), Y_aaaa, Y_aaaa, optimize = True)
            IjKl -= lib.einsum('jl,Ia,Ka->IjKl', np.identity(nocc_b), Y_aa, Y_aa, optimize = True)
            IjKl -= lib.einsum('jl,Imab,Kmab->IjKl', np.identity(nocc_b), Y_abab, Y_abab, optimize = True)
            IjKl += lib.einsum('Ia,Klab,jb->IjKl', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            IjKl += lib.einsum('Ka,Ijab,lb->IjKl', t1_ce_aa, Y_abab, Y_bb, optimize = True)
            IjKl += lib.einsum('ja,Klba,Ib->IjKl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            IjKl += lib.einsum('la,Ijba,Kb->IjKl', t1_ce_bb, Y_abab, Y_aa, optimize = True)
            IjKl += 1/4 * lib.einsum('mnab,mnab,IK,jl->IjKl', Y_aaaa, Y_aaaa, np.identity(nocc_a), np.identity(nocc_b), optimize = True)
            IjKl += lib.einsum('ma,ma,IK,jl->IjKl', Y_aa, Y_aa, np.identity(nocc_a), np.identity(nocc_b), optimize = True)
            IjKl += lib.einsum('mnab,mnab,IK,jl->IjKl', Y_abab, Y_abab, np.identity(nocc_a), np.identity(nocc_b), optimize = True)
            IjKl += 1/4 * lib.einsum('mnab,mnab,IK,jl->IjKl', Y_bbbb, Y_bbbb, np.identity(nocc_a), np.identity(nocc_b), optimize = True)
            IjKl += lib.einsum('ma,ma,IK,jl->IjKl', Y_bb, Y_bb, np.identity(nocc_a), np.identity(nocc_b), optimize = True)
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
            IjKl -= lib.einsum('IK,ja,lmab,mb->IjKl', np.identity(nocc_a), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            IjKl -= lib.einsum('IK,ja,mlba,mb->IjKl', np.identity(nocc_a), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            IjKl -= lib.einsum('IK,la,jmab,mb->IjKl', np.identity(nocc_a), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            IjKl -= lib.einsum('IK,la,mjba,mb->IjKl', np.identity(nocc_a), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            IjKl -= lib.einsum('jl,Ia,Kmab,mb->IjKl', np.identity(nocc_b), t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            IjKl -= lib.einsum('jl,Ia,Kmab,mb->IjKl', np.identity(nocc_b), t1_ce_aa, Y_abab, Y_bb, optimize = True)
            IjKl -= lib.einsum('jl,Ka,Imab,mb->IjKl', np.identity(nocc_b), t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            IjKl -= lib.einsum('jl,Ka,Imab,mb->IjKl', np.identity(nocc_b), t1_ce_aa, Y_abab, Y_bb, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,ja,mb,lnac,mnbc->IjKl', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,ja,mb,nlca,mnbc->IjKl', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            IjKl += 1/4 * lib.einsum('IK,ja,ma,mnbc,lnbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl += 1/2 * lib.einsum('IK,ja,ma,mb,lb->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            IjKl += 1/2 * lib.einsum('IK,ja,ma,nmbc,nlbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,ja,mb,lnac,mnbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,ja,mb,nlca,nmcb->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,la,mb,jnac,mnbc->IjKl', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,la,mb,njca,mnbc->IjKl', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            IjKl += 1/4 * lib.einsum('IK,la,ma,mnbc,jnbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl += 1/2 * lib.einsum('IK,la,ma,mb,jb->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            IjKl += 1/2 * lib.einsum('IK,la,ma,nmbc,njbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,la,mb,jnac,mnbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,la,mb,njca,nmcb->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,ma,ma,jnbc,lnbc->IjKl', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl -= lib.einsum('IK,ma,ma,jb,lb->IjKl', np.identity(nocc_a), Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            IjKl -= lib.einsum('IK,ma,ma,njbc,nlbc->IjKl', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl += lib.einsum('IK,ma,mb,njac,nlbc->IjKl', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl += lib.einsum('IK,ma,na,mjbc,nlbc->IjKl', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= lib.einsum('IK,ma,nb,mjac,nlbc->IjKl', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('IK,ma,ma,jnbc,lnbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl -= lib.einsum('IK,ma,ma,jb,lb->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            IjKl -= lib.einsum('IK,ma,ma,njbc,nlbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl += lib.einsum('IK,ma,mb,ja,lb->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            IjKl += lib.einsum('IK,ma,mb,jnac,lnbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl += lib.einsum('IK,ma,mb,njca,nlcb->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl += lib.einsum('IK,ma,nb,jmac,nlbc->IjKl', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            IjKl += lib.einsum('IK,ma,nb,lmac,njbc->IjKl', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            IjKl += 1/2 * lib.einsum('IK,ma,na,jmbc,lnbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl -= lib.einsum('IK,ma,nb,jmac,lnbc->IjKl', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IjKl += 1/4 * lib.einsum('jl,Ia,ma,mnbc,Knbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl += 1/2 * lib.einsum('jl,Ia,ma,mb,Kb->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            IjKl += 1/2 * lib.einsum('jl,Ia,ma,mnbc,Knbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Ia,mb,Knac,mnbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Ia,mb,Knac,mnbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Ia,mb,Knac,nmcb->IjKl', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Ia,mb,Knac,mnbc->IjKl', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            IjKl += 1/4 * lib.einsum('jl,Ka,ma,mnbc,Inbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl += 1/2 * lib.einsum('jl,Ka,ma,mb,Ib->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            IjKl += 1/2 * lib.einsum('jl,Ka,ma,mnbc,Inbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Ka,mb,Inac,mnbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Ka,mb,Inac,mnbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Ka,mb,Inac,nmcb->IjKl', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,Ka,mb,Inac,mnbc->IjKl', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,ma,ma,Inbc,Knbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl -= lib.einsum('jl,ma,ma,Ib,Kb->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            IjKl -= lib.einsum('jl,ma,ma,Inbc,Knbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl += lib.einsum('jl,ma,mb,Ia,Kb->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            IjKl += lib.einsum('jl,ma,mb,Inac,Knbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl += lib.einsum('jl,ma,mb,Inac,Knbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl += 1/2 * lib.einsum('jl,ma,na,Imbc,Knbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl -= lib.einsum('jl,ma,nb,Imac,Knbc->IjKl', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl += lib.einsum('jl,ma,nb,Imac,Kncb->IjKl', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            IjKl -= 1/2 * lib.einsum('jl,ma,ma,Inbc,Knbc->IjKl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            IjKl -= lib.einsum('jl,ma,ma,Ib,Kb->IjKl', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            IjKl -= lib.einsum('jl,ma,ma,Inbc,Knbc->IjKl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl += lib.einsum('jl,ma,mb,Inca,Kncb->IjKl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl += lib.einsum('jl,ma,nb,Imca,Knbc->IjKl', np.identity(nocc_b), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            IjKl += lib.einsum('jl,ma,na,Imbc,Knbc->IjKl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IjKl -= lib.einsum('jl,ma,nb,Imca,Kncb->IjKl', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
                
                
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
                
                
            AbKl  = lib.einsum('KA,libc,ic->AbKl', t1_ce_aa, Y_bbbb, Y_bb, optimize = True)
            AbKl += lib.einsum('KA,ilcb,ic->AbKl', t1_ce_aa, Y_abab, Y_aa, optimize = True)
            AbKl -= lib.einsum('iA,ic,Klcb->AbKl', t1_ce_aa, Y_aa, Y_abab, optimize = True)
            AbKl += lib.einsum('KlAb,ic,ic->AbKl', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
            AbKl += lib.einsum('KlAb,ic,ic->AbKl', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
            AbKl -= lib.einsum('KlAc,ic,ib->AbKl', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
            AbKl -= lib.einsum('Klcb,ic,iA->AbKl', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
            AbKl += lib.einsum('KiAc,ic,lb->AbKl', t1_ccee_aaaa, Y_aa, Y_bb, optimize = True)
            AbKl -= lib.einsum('Kc,ic,ilAb->AbKl', t1_ce_aa, Y_aa, Y_abab, optimize = True)
            AbKl -= lib.einsum('KiAb,ic,lc->AbKl', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
            AbKl += lib.einsum('KiAc,ic,lb->AbKl', t1_ccee_abab, Y_bb, Y_bb, optimize = True)
            AbKl += lib.einsum('lb,KiAc,ic->AbKl', t1_ce_bb, Y_aaaa, Y_aa, optimize = True)
            AbKl += lib.einsum('lb,KiAc,ic->AbKl', t1_ce_bb, Y_abab, Y_bb, optimize = True)
            AbKl -= lib.einsum('ib,ic,KlAc->AbKl', t1_ce_bb, Y_bb, Y_abab, optimize = True)
            AbKl += lib.einsum('libc,ic,KA->AbKl', t1_ccee_bbbb, Y_bb, Y_aa, optimize = True)
            AbKl -= lib.einsum('lc,ic,KiAb->AbKl', t1_ce_bb, Y_bb, Y_abab, optimize = True)
            AbKl -= lib.einsum('ilAb,ic,Kc->AbKl', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
            AbKl += lib.einsum('ilcb,ic,KA->AbKl', t1_ccee_abab, Y_aa, Y_aa, optimize = True)
            AbKl += lib.einsum('KlAb,ic,ic->AbKl', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
            AbKl += lib.einsum('KlAb,ic,ic->AbKl', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
            AbKl -= lib.einsum('KlAc,ic,ib->AbKl', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
            AbKl -= lib.einsum('Klcb,ic,iA->AbKl', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
            AbKl += lib.einsum('KiAc,ic,lb->AbKl', t2_ccee_aaaa, Y_aa, Y_bb, optimize = True)
            AbKl -= lib.einsum('KiAb,ic,lc->AbKl', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
            AbKl += lib.einsum('KiAc,ic,lb->AbKl', t2_ccee_abab, Y_bb, Y_bb, optimize = True)
            AbKl += lib.einsum('libc,ic,KA->AbKl', t2_ccee_bbbb, Y_bb, Y_aa, optimize = True)
            AbKl -= lib.einsum('ilAb,ic,Kc->AbKl', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
            AbKl += lib.einsum('ilcb,ic,KA->AbKl', t2_ccee_abab, Y_aa, Y_aa, optimize = True)
            AbKl -= lib.einsum('KA,ic,ib,lc->AbKl', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            AbKl -= lib.einsum('Kc,ic,iA,lb->AbKl', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
            AbKl -= lib.einsum('lb,ic,iA,Kc->AbKl', Y_bb, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            AbKl -= lib.einsum('lc,ic,ib,KA->AbKl', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
            AbKl -= lib.einsum('iA,ic,Kc,lb->AbKl', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
            AbKl += lib.einsum('ic,ic,KA,lb->AbKl', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
            AbKl -= lib.einsum('ib,ic,lc,KA->AbKl', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
            AbKl += lib.einsum('ic,ic,KA,lb->AbKl', Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
                
                
            IaKd =- lib.einsum('Ijbd,Kjba->IaKd', Y_abab, Y_abab, optimize = True)
            IaKd += lib.einsum('IK,jlba,jlbd->IaKd', np.identity(nocc_a), Y_abab, Y_abab, optimize = True)
            IaKd += lib.einsum('IK,ja,jd->IaKd', np.identity(nocc_a), Y_bb, Y_bb, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,jlab,jldb->IaKd', np.identity(nocc_a), Y_bbbb, Y_bbbb, optimize = True)
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
            IaKd += lib.einsum('IK,ja,jldb,lb->IaKd', np.identity(nocc_a), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            IaKd += lib.einsum('IK,ja,ljbd,lb->IaKd', np.identity(nocc_a), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            IaKd += lib.einsum('IK,jd,jlab,lb->IaKd', np.identity(nocc_a), t1_ce_bb, Y_bbbb, Y_bb, optimize = True)
            IaKd += lib.einsum('IK,jd,ljba,lb->IaKd', np.identity(nocc_a), t1_ce_bb, Y_abab, Y_aa, optimize = True)
            IaKd += lib.einsum('IK,jb,jb,la,ld->IaKd', np.identity(nocc_a), Y_aa, Y_aa, t1_ce_bb, t1_ce_bb, optimize = True)
            IaKd += lib.einsum('IK,jb,jb,lmca,lmcd->IaKd', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,jb,jb,lmac,lmdc->IaKd', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IaKd -= lib.einsum('IK,jb,jc,lmba,lmcd->IaKd', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd -= lib.einsum('IK,jb,lb,jmca,lmcd->IaKd', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd += lib.einsum('IK,jb,lc,jmba,lmcd->IaKd', np.identity(nocc_a), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd -= 1/2 * lib.einsum('IK,ja,jb,lb,ld->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            IaKd -= 1/2 * lib.einsum('IK,ja,jb,lmcb,lmcd->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd -= 1/4 * lib.einsum('IK,ja,jb,lmbc,lmdc->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,ja,lb,jmdc,lmbc->IaKd', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,ja,lb,mjcd,lmbc->IaKd', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,ja,lb,jmdc,lmbc->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,ja,lb,mjcd,mlcb->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd -= 1/2 * lib.einsum('IK,jd,jb,lb,la->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            IaKd -= 1/2 * lib.einsum('IK,jd,jb,lmcb,lmca->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd -= 1/4 * lib.einsum('IK,jd,jb,lmbc,lmac->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,jd,lb,jmac,lmbc->IaKd', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,jd,lb,mjca,lmbc->IaKd', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,jd,lb,jmac,lmbc->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,jd,lb,mjca,mlcb->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd += lib.einsum('IK,jb,jb,la,ld->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            IaKd += lib.einsum('IK,jb,jb,lmca,lmcd->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd += 1/2 * lib.einsum('IK,jb,jb,lmac,lmdc->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IaKd -= 1/2 * lib.einsum('IK,jb,jc,lmab,lmdc->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IaKd -= lib.einsum('IK,jb,lc,jmab,lmcd->IaKd', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            IaKd -= lib.einsum('IK,jb,lc,jmdb,lmca->IaKd', np.identity(nocc_a), Y_bb, Y_aa, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            IaKd -= lib.einsum('IK,jb,lb,ja,ld->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            IaKd -= lib.einsum('IK,jb,lb,jmac,lmdc->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            IaKd -= lib.einsum('IK,jb,lb,mjca,mlcd->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            IaKd += lib.einsum('IK,jb,lc,jmab,lmdc->IaKd', np.identity(nocc_a), Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
                
                
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
            
            
            iAKd =- lib.einsum('KA,id->iAKd', Y_aa, Y_bb, optimize = True)
            iAKd -= lib.einsum('KjAb,jibd->iAKd', Y_aaaa, Y_abab, optimize = True)
            iAKd -= lib.einsum('KjAb,ijdb->iAKd', Y_abab, Y_bbbb, optimize = True)
            iAKd -= lib.einsum('KA,ijdb,jb->iAKd', t1_ce_aa, Y_bbbb, Y_bb, optimize = True)
            iAKd -= lib.einsum('KA,jibd,jb->iAKd', t1_ce_aa, Y_abab, Y_aa, optimize = True)
            iAKd += lib.einsum('jA,jibd,Kb->iAKd', t1_ce_aa, Y_abab, Y_aa, optimize = True)
            iAKd += lib.einsum('Kb,jibd,jA->iAKd', t1_ce_aa, Y_abab, Y_aa, optimize = True)
            iAKd -= lib.einsum('id,KjAb,jb->iAKd', t1_ce_bb, Y_aaaa, Y_aa, optimize = True)
            iAKd -= lib.einsum('id,KjAb,jb->iAKd', t1_ce_bb, Y_abab, Y_bb, optimize = True)
            iAKd += lib.einsum('jd,KjAb,ib->iAKd', t1_ce_bb, Y_abab, Y_bb, optimize = True)
            iAKd += lib.einsum('ib,KjAb,jd->iAKd', t1_ce_bb, Y_abab, Y_bb, optimize = True)
            iAKd += 1/2 * lib.einsum('KA,ib,jb,jd->iAKd', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            iAKd += 1/2 * lib.einsum('KA,ib,jlcb,jlcd->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += 1/4 * lib.einsum('KA,ib,jlbc,jldc->iAKd', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            iAKd -= 1/2 * lib.einsum('KA,jb,jlbc,licd->iAKd', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAKd -= 1/2 * lib.einsum('KA,jb,jlbc,ildc->iAKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd += 1/4 * lib.einsum('KA,jd,jlbc,ilbc->iAKd', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            iAKd += 1/2 * lib.einsum('KA,jd,jb,ib->iAKd', Y_aa, Y_bb, t1_ce_bb, t1_ce_bb, optimize = True)
            iAKd += 1/2 * lib.einsum('KA,jd,ljbc,libc->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd -= 1/2 * lib.einsum('KA,jb,jlbc,ildc->iAKd', Y_aa, Y_bb, t1_ccee_bbbb, t1_ccee_bbbb, optimize = True)
            iAKd -= 1/2 * lib.einsum('KA,jb,ljcb,licd->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += 1/2 * lib.einsum('Kb,id,jb,jA->iAKd', Y_aa, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            iAKd += 1/4 * lib.einsum('Kb,id,jlbc,jlAc->iAKd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAKd += 1/2 * lib.einsum('Kb,id,jlbc,jlAc->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd -= lib.einsum('Kb,ic,jlbd,jlAc->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('Kb,jb,jA,id->iAKd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
            iAKd += lib.einsum('Kb,jb,jlAc,licd->iAKd', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('Kb,jb,jlAc,ildc->iAKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd -= lib.einsum('Kb,jc,libd,jlAc->iAKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            iAKd -= lib.einsum('Kb,jd,libc,ljAc->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('Kb,jc,libd,ljAc->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += 1/4 * lib.einsum('id,jA,jlbc,Klbc->iAKd', Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAKd += 1/2 * lib.einsum('id,jA,jb,Kb->iAKd', Y_bb, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            iAKd += 1/2 * lib.einsum('id,jA,jlbc,Klbc->iAKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd -= 1/2 * lib.einsum('id,jb,jlbc,KlAc->iAKd', Y_bb, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAKd -= 1/2 * lib.einsum('id,jb,jlbc,KlAc->iAKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd -= 1/2 * lib.einsum('id,jb,jlbc,KlAc->iAKd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            iAKd -= 1/2 * lib.einsum('id,jb,ljcb,KlAc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            iAKd -= lib.einsum('ib,jA,Klcb,jlcd->iAKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('ib,jc,KlAb,jlcd->iAKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('ib,jb,jd,KA->iAKd', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
            iAKd += lib.einsum('ib,jb,jldc,KlAc->iAKd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('ib,jb,ljcd,KlAc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            iAKd -= lib.einsum('ib,jc,KlAb,jldc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd += lib.einsum('jA,jb,Kb,id->iAKd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
            iAKd += lib.einsum('jA,jb,Klbc,licd->iAKd', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('jA,jb,Klbc,ildc->iAKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd -= lib.einsum('jA,lb,jicd,Klbc->iAKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            iAKd -= lib.einsum('jA,ld,jibc,Klbc->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('jA,lb,jicd,Klcb->iAKd', Y_aa, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd -= lib.einsum('jb,jb,KA,id->iAKd', Y_aa, Y_aa, t1_ce_aa, t1_ce_bb, optimize = True)
            iAKd -= lib.einsum('jb,jb,KlAc,licd->iAKd', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAKd -= lib.einsum('jb,jb,KlAc,ildc->iAKd', Y_aa, Y_aa, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd += lib.einsum('jb,jc,KlAb,licd->iAKd', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('jb,lb,KjAc,licd->iAKd', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAKd -= lib.einsum('jb,lc,KjAb,licd->iAKd', Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAKd -= lib.einsum('jb,lc,KjAb,ildc->iAKd', Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_bbbb, optimize = True)
            iAKd += lib.einsum('jd,jb,ib,KA->iAKd', Y_bb, Y_bb, t1_ce_bb, t1_ce_aa, optimize = True)
            iAKd += lib.einsum('jd,jb,ilbc,KlAc->iAKd', Y_bb, Y_bb, t1_ccee_bbbb, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('jd,jb,licb,KlAc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_aaaa, optimize = True)
            iAKd += lib.einsum('jd,lb,KjAc,libc->iAKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd -= lib.einsum('jd,lb,KjAc,ilbc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd -= lib.einsum('jb,jb,KA,id->iAKd', Y_bb, Y_bb, t1_ce_aa, t1_ce_bb, optimize = True)
            iAKd -= lib.einsum('jb,jb,KlAc,licd->iAKd', Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAKd -= lib.einsum('jb,jb,KlAc,ildc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd += lib.einsum('jb,jc,KlAb,ildc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd -= lib.einsum('jb,lc,KjAb,licd->iAKd', Y_bb, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAKd += lib.einsum('jb,lb,KjAc,ildc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAKd -= lib.einsum('jb,lc,KjAb,ildc->iAKd', Y_bb, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            
            
            
            iAkD =- lib.einsum('jiDb,jkAb->iAkD', Y_abab, Y_abab, optimize = True)
            iAkD += lib.einsum('ik,jA,jD->iAkD', np.identity(nocc_b), Y_aa, Y_aa, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jlAb,jlDb->iAkD', np.identity(nocc_b), Y_aaaa, Y_aaaa, optimize = True)
            iAkD += lib.einsum('ik,jlAb,jlDb->iAkD', np.identity(nocc_b), Y_abab, Y_abab, optimize = True)
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
            iAkD += lib.einsum('ik,jA,jlDb,lb->iAkD', np.identity(nocc_b), t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            iAkD += lib.einsum('ik,jA,jlDb,lb->iAkD', np.identity(nocc_b), t1_ce_aa, Y_abab, Y_bb, optimize = True)
            iAkD += lib.einsum('ik,jD,jlAb,lb->iAkD', np.identity(nocc_b), t1_ce_aa, Y_aaaa, Y_aa, optimize = True)
            iAkD += lib.einsum('ik,jD,jlAb,lb->iAkD', np.identity(nocc_b), t1_ce_aa, Y_abab, Y_bb, optimize = True)
            iAkD -= 1/2 * lib.einsum('ik,jA,jb,lb,lD->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            iAkD -= 1/4 * lib.einsum('ik,jA,jb,lmbc,lmDc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD -= 1/2 * lib.einsum('ik,jA,jb,lmbc,lmDc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jA,lb,jmDc,lmbc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jA,lb,jmDc,lmbc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jA,lb,jmDc,mlcb->iAkD', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jA,lb,jmDc,lmbc->iAkD', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAkD -= 1/2 * lib.einsum('ik,jD,jb,lb,lA->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            iAkD -= 1/4 * lib.einsum('ik,jD,jb,lmbc,lmAc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD -= 1/2 * lib.einsum('ik,jD,jb,lmbc,lmAc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jD,lb,jmAc,lmbc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jD,lb,jmAc,lmbc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jD,lb,jmAc,mlcb->iAkD', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jD,lb,jmAc,lmbc->iAkD', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_abab, t1_ccee_bbbb, optimize = True)
            iAkD += lib.einsum('ik,jb,jb,lA,lD->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jb,jb,lmAc,lmDc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD += lib.einsum('ik,jb,jb,lmAc,lmDc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD -= 1/2 * lib.einsum('ik,jb,jc,lmAb,lmDc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD -= lib.einsum('ik,jb,lb,jA,lD->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ce_aa, t1_ce_aa, optimize = True)
            iAkD -= lib.einsum('ik,jb,lb,jmAc,lmDc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD -= lib.einsum('ik,jb,lb,jmAc,lmDc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD += lib.einsum('ik,jb,lc,jmAb,lmDc->iAkD', np.identity(nocc_b), Y_aa, Y_aa, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD -= lib.einsum('ik,jb,lc,jmAb,mlDc->iAkD', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAkD -= lib.einsum('ik,jb,lc,jmDb,mlAc->iAkD', np.identity(nocc_b), Y_aa, Y_bb, t1_ccee_aaaa, t1_ccee_abab, optimize = True)
            iAkD += lib.einsum('ik,jb,jb,lA,lD->iAkD', np.identity(nocc_b), Y_bb, Y_bb, t1_ce_aa, t1_ce_aa, optimize = True)
            iAkD += 1/2 * lib.einsum('ik,jb,jb,lmAc,lmDc->iAkD', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_aaaa, t1_ccee_aaaa, optimize = True)
            iAkD += lib.einsum('ik,jb,jb,lmAc,lmDc->iAkD', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD -= lib.einsum('ik,jb,jc,lmAb,lmDc->iAkD', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD -= lib.einsum('ik,jb,lb,mjAc,mlDc->iAkD', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
            iAkD += lib.einsum('ik,jb,lc,mjAb,mlDc->iAkD', np.identity(nocc_b), Y_bb, Y_bb, t1_ccee_abab, t1_ccee_abab, optimize = True)
     
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


                AbKl += 1/4 * lib.einsum('KlAb,ijcd,ijcd->AbKl', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
                AbKl += lib.einsum('KlAb,ijcd,ijcd->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += 1/4 * lib.einsum('KlAb,ijcd,ijcd->AbKl', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
                AbKl -= lib.einsum('KlAc,ijdc,ijdb->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl -= 1/2 * lib.einsum('KlAc,ijcd,ijbd->AbKl', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
                AbKl -= 1/2 * lib.einsum('Klcb,ijcd,ijAd->AbKl', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
                AbKl -= lib.einsum('Klcb,ijcd,ijAd->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += lib.einsum('Klcd,ijcd,ijAb->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += lib.einsum('KiAc,ijcd,jldb->AbKl', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
                AbKl += lib.einsum('KiAc,ijcd,ljbd->AbKl', t1_ccee_aaaa, Y_abab, Y_bbbb, optimize = True)
                AbKl += 1/2 * lib.einsum('Kicd,ijcd,jlAb->AbKl', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
                AbKl -= 1/2 * lib.einsum('KiAb,ijcd,ljcd->AbKl', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
                AbKl -= lib.einsum('KiAb,jicd,jlcd->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += lib.einsum('KiAc,ijcd,ljbd->AbKl', t1_ccee_abab, Y_bbbb, Y_bbbb, optimize = True)
                AbKl += lib.einsum('KiAc,jidc,jldb->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += lib.einsum('Kicb,jicd,jlAd->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl -= lib.einsum('Kicd,jicd,jlAb->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += lib.einsum('libc,ijcd,KjAd->AbKl', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)
                AbKl += lib.einsum('libc,jidc,KjAd->AbKl', t1_ccee_bbbb, Y_abab, Y_aaaa, optimize = True)
                AbKl += 1/2 * lib.einsum('licd,ijcd,KjAb->AbKl', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)
                AbKl -= 1/2 * lib.einsum('ilAb,ijcd,Kjcd->AbKl', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
                AbKl -= lib.einsum('ilAb,ijcd,Kjcd->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += lib.einsum('ilAc,ijdc,Kjdb->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += lib.einsum('ilcb,ijcd,KjAd->AbKl', t1_ccee_abab, Y_aaaa, Y_aaaa, optimize = True)
                AbKl += lib.einsum('ilcb,ijcd,KjAd->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl -= lib.einsum('ilcd,ijcd,KjAb->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += 1/2 * lib.einsum('ijAc,ijcd,Kldb->AbKl', t1_ccee_aaaa, Y_aaaa, Y_abab, optimize = True)
                AbKl += lib.einsum('ijAb,ijcd,Klcd->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl -= lib.einsum('ijAc,ijdc,Kldb->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl -= lib.einsum('ijcb,ijcd,KlAd->AbKl', t1_ccee_abab, Y_abab, Y_abab, optimize = True)
                AbKl += 1/2 * lib.einsum('ijbc,ijcd,KlAd->AbKl', t1_ccee_bbbb, Y_bbbb, Y_abab, optimize = True)

            if adc.tpdm is True:
                abab = (IjKl,AbCd,IjCd,AbKl,IaKd,IakD,iAKd,iAkD)
                mixed_spin = np.append(mixed_spin,abab)


            IjlK = -IjKl.transpose(0,1,3,2)
            AbdC = -AbCd.transpose(0,1,3,2)
            IjdC = -IjCd.transpose(0,1,3,2)
            AblK = -AbKl.transpose(0,1,3,2)

            IadK = -IaKd.transpose(0,1,3,2)
            #IakD
            AidK = iAKd.transpose(1,0,3,2)
            AikD = -iAkD.transpose(1,0,2,3)

            trace_a = lib.einsum('pp',temp_a)
            trace_b = lib.einsum('pp',temp_b)


            spin_c = 0.25*((trace_a - trace_b)**2) + 0.5*(trace_a + trace_b) + lib.einsum('pr,qs,pqrs->',delta[:nocc_a,:nocc_b], delta[:nocc_a,:nocc_b].T, IjlK, optimize = True)
            spin_c += lib.einsum('pr,qs,pqrs->', delta[nocc_a:,nocc_b:], delta[nocc_a:,nocc_b:].T, AbdC, optimize = True)
            spin_c += lib.einsum('pr,qs,pqrs->', delta[:nocc_a,nocc_b:], delta[nocc_a:,:nocc_b].T, IjdC, optimize = True)
            spin_c += lib.einsum('pr,qs,pqrs->', delta[nocc_a:,:nocc_b], delta[:nocc_a,nocc_b:].T,AblK, optimize = True)
            spin_c += lib.einsum('pr,qs,pqrs->', delta[:nocc_a,nocc_b:], delta[:nocc_a,nocc_b:].T,IadK, optimize = True)
            spin_c += lib.einsum('pr,qs,pqrs->', delta[:nocc_a,:nocc_b], delta[nocc_a:,nocc_b:].T,IakD, optimize = True)
            spin_c += lib.einsum('pr,qs,pqrs->', delta[nocc_a:,nocc_b:], delta[:nocc_a,:nocc_b:].T,AidK, optimize = True)
            spin_c += lib.einsum('pr,qs,pqrs->', delta[nocc_a:,:nocc_b], delta[nocc_a:,:nocc_b].T,AikD, optimize = True)
            
            spin = np.append(spin,spin_c)



    opdm_a = opdm_a.reshape(nroots,nmo_a,nmo_a)
    opdm_b = opdm_b.reshape(nroots,nmo_b,nmo_b)

    if adc.spin_c is True or adc.tpdm is True:
        tpdm_a = tpdm_a.reshape(nroots,nmo_a, nmo_a, nmo_a, nmo_a)
        tpdm_b = tpdm_a.reshape(nroots,nmo_b, nmo_b, nmo_b, nmo_b)
        spin = spin.reshape(nroots,1)
    else:
        spin = None

    pdm = (opdm_a ,opdm_b ,tpdm_a,tpdm_b ,mixed_spin)

    return spin, pdm




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




    dm_a = adc.dm_a.copy()
    dm_b = adc.dm_b.copy()


    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    n_singles_a = nocc_a * nvir_a
    n_singles_b = nocc_b * nvir_b
    n_doubles_aaaa = nocc_a * (nocc_a - 1) * nvir_a * (nvir_a -1) // 4
    n_doubles_ab = nocc_a * nocc_b * nvir_a * nvir_b
    n_doubles_bbbb = nocc_b * (nocc_b - 1) * nvir_b * (nvir_b -1) // 4

    dim = n_singles_a + n_singles_b + n_doubles_aaaa + n_doubles_ab + n_doubles_bbbb

    TY_a = np.zeros((nmo_a,nmo_a))
    TY_b = np.zeros((nmo_b,nmo_b))
    
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
        TY_a[:nocc_a,:nocc_a] = -lib.einsum('pg,qg->qp',Y_a,t1_2_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] = lib.einsum('xr,xv->rv',Y_a,t1_2_a,optimize=True)
        del t1_2_a


        TY_a[nocc_a:,:nocc_a] = Y_a.T
        TY_b[nocc_b:,:nocc_b] = Y_b.T



        t1_2_b = t1[0][1][:]
        TY_b[:nocc_b,:nocc_b] = -lib.einsum('pg,qg->qp',Y_b,t1_2_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] = lib.einsum('xr,xv->rv',Y_b,t1_2_b,optimize=True)
        del t1_2_b

        t2_1_a = t2[0][0][:]
        t2_1_ab = t2[0][1][:]
        TY_a[:nocc_a,:nocc_a] += lib.einsum('xg,ph,qxgh->qp',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('pg,xh,qxgh->qp',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= 0.5*lib.einsum('pxgh,qxgh->qp',Y1_oovv_u_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] += 0.5*lib.einsum('xyrg,xyvg->rv',Y1_oovv_u_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] -= lib.einsum('xg,yr,xyvg->rv',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] += 0.5*np.einsum('xr,yg,xyvg->rv',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,:nocc_a] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_a,t2_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,:nocc_a] -= 0.25*lib.einsum('pg,xygh,xyvh->vp',Y_a,t2_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,nocc_a:] = lib.einsum('xg,pxvg->pv',Y_a,t2_1_a,optimize=True)
        TY_b[nocc_b:,:nocc_b] += 0.5*lib.einsum('xg,xygh,yphv->vp',Y_a,t2_1_a,t2_1_ab,optimize=True)
        TY_a[nocc_a:,:nocc_a] += 0.5*lib.einsum('xg,yxhg,pyvh->vp',Y_b,t2_1_ab,t2_1_a,optimize=True)
        TY_a[nocc_a:,:nocc_a] -= 0.25*lib.einsum('xv,xygh,pygh->vp',Y_a,t2_1_a,t2_1_a,optimize=True)
        del t2_1_a
        del t2_1_ab


        t2_1_b = t2[0][2][:]
        t2_1_ab = t2[0][1][:]
        TY_b[:nocc_b,:nocc_b] += lib.einsum('xg,ph,qxgh->qp',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('pg,xh,qxgh->qp',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= 0.5*lib.einsum('pxgh,qxgh->qp',Y1_oovv_u_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] += 0.5*lib.einsum('xyrg,xyvg->rv',Y1_oovv_u_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] -= lib.einsum('xg,yr,xyvg->rv',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] += 0.5*np.einsum('xr,yg,xyvg->rv',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,:nocc_b] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.25*lib.einsum('xv,xygh,pygh->vp',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,nocc_b:] = lib.einsum('xg,pxvg->pv',Y_b,t2_1_b,optimize=True)
        TY_a[nocc_a:,:nocc_a] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_b,t2_1_b,t2_1_ab,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.25*lib.einsum('pg,xygh,xyvh->vp',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,:nocc_b] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_a,t2_1_ab,t2_1_b,optimize=True)
        del t2_1_b
        del t2_1_ab

        t2_1_ab = t2[0][1][:]
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('xg,ph,qxhg->qp',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('xg,ph,xqgh->qp',Y_a,t1_1_b,t2_1_ab,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('pg,xh,qxgh->qp',Y_a,t1_1_b,t2_1_ab,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('pg,xh,xqhg->qp',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('pxgh,qxgh->qp',Y1_abab,t2_1_ab,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('xpgh,xqgh->qp',Y1_abab,t2_1_ab,optimize=True)
        TY_a[nocc_a:,nocc_a:] += lib.einsum('xyrg,xyvg->rv',Y1_abab,t2_1_ab,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xygr,xygv->rv',Y1_abab,t2_1_ab,optimize=True)
        TY_a[nocc_a:,nocc_a:] += lib.einsum('xg,yr,yxvg->rv',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xg,yr,xygv->rv',Y_a,t1_1_b,t2_1_ab,optimize=True)
        TY_a[nocc_a:,nocc_a:] += 0.5*np.einsum('xr,yg,xyvg->rv',Y_a,t1_1_b,t2_1_ab,optimize=True)
        TY_b[nocc_b:,nocc_b:] += 0.5*np.einsum('xr,yg,yxgv->rv',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_a[nocc_a:,:nocc_a] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_a,t2_1_ab,t2_1_ab,optimize=True)
        TY_b[nocc_b:,:nocc_b] += 0.5*lib.einsum('xg,yxhg,yphv->vp',Y_b,t2_1_ab,t2_1_ab,optimize=True)
        TY_a[nocc_a:,:nocc_a] -= 0.5*lib.einsum('pg,xygh,xyvh->vp',Y_a,t2_1_ab,t2_1_ab,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.5*lib.einsum('pg,xyhg,xyhv->vp',Y_b,t2_1_ab,t2_1_ab,optimize=True)
        TY_a[nocc_a:,:nocc_a] -= 0.5*lib.einsum('xv,xygh,pygh->vp',Y_a,t2_1_ab,t2_1_ab,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.5*lib.einsum('xv,yxgh,ypgh->vp',Y_b,t2_1_ab,t2_1_ab,optimize=True)
        TY_a[:nocc_a,nocc_a:] += lib.einsum('xg,pxvg->pv',Y_b,t2_1_ab,optimize=True)
        TY_b[:nocc_b,nocc_b:] += lib.einsum('xg,xpgv->pv',Y_a,t2_1_ab,optimize=True)
        del t2_1_ab 
       
       
       
       # T_U1_Y0 qp block
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('pg,qg->qp',Y_a,t1_1_a,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('pg,qg->qp',Y_b,t1_1_b,optimize=True)

        TY_a[nocc_a:,nocc_a:] += lib.einsum('xr,xv->rv',Y_a,t1_1_a,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xr,xv->rv',Y_b,t1_1_b,optimize=True)



        TY_a[nocc_a:,:nocc_a] -= 0.5*np.einsum('pg,xg,xv->vp',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.5*np.einsum('pg,xg,xv->vp',Y_b,t1_1_b,t1_1_b,optimize=True)

        TY_a[nocc_a:,:nocc_a] -= 0.5*np.einsum('xv,xg,pg->vp',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.5*np.einsum('xv,xg,pg->vp',Y_b,t1_1_b,t1_1_b,optimize=True)

        TY_a[:nocc_a,nocc_a:] -= lib.einsum('xg,pg,xv->pv',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[:nocc_b,nocc_b:] -= lib.einsum('xg,pg,xv->pv',Y_b,t1_1_b,t1_1_b,optimize=True)


        t2_2_a = t2[1][0][:]
        TY_a[:nocc_a,nocc_a:] += lib.einsum('xg,pxvg->pv',Y_a,t2_2_a,optimize=True)
        del t2_2_a


        t2_2_b = t2[1][2][:]
        TY_b[:nocc_b,nocc_b:] += lib.einsum('xg,pxvg->pv',Y_b,t2_2_b,optimize=True)
        del t2_2_b

        t2_2_ab = t2[1][1][:]
        TY_b[:nocc_b,nocc_b:] += lib.einsum('xg,xpgv->pv',Y_a,t2_2_ab,optimize=True)
        TY_a[:nocc_a,nocc_a:] += lib.einsum('xg,pxvg->pv',Y_b,t2_2_ab,optimize=True)
        del t2_2_ab

        if (method == "adc(2)"):
            del Y1_abab
            del Y1_oovv_u_a
            del Y1_oovv_u_b



        if (method == "adc(2)-x"):
            # T_U2_Y1 qp block
            t2_2_a = t2[1][0][:]
            TY_a[:nocc_a,:nocc_a] -= 0.5*lib.einsum('pxgh,qxgh->qp',Y1_oovv_u_a,t2_2_a,optimize=True)
            TY_a[nocc_a:,nocc_a:] += 0.5*lib.einsum('xyrg,xyvg->rv',Y1_oovv_u_a,t2_2_a,optimize=True)
            del t2_2_a
            
            t2_2_b = t2[1][2][:]
            TY_b[:nocc_b,:nocc_b] -= 0.5*lib.einsum('pxgh,qxgh->qp',Y1_oovv_u_b,t2_2_b,optimize=True)
            TY_b[nocc_b:,nocc_b:] += 0.5*lib.einsum('xyrg,xyvg->rv',Y1_oovv_u_b,t2_2_b,optimize=True)
            del t2_2_b

            t2_2_ab = t2[1][1][:]
            TY_a[:nocc_a,:nocc_a] -= lib.einsum('pxgh,qxgh->qp',Y1_abab,t2_2_ab,optimize=True)
            TY_b[:nocc_b,:nocc_b] -= lib.einsum('xpgh,xqgh->qp',Y1_abab,t2_2_ab,optimize=True)
            # T_U2_Y1 rv block
            TY_a[nocc_a:,nocc_a:] += lib.einsum('xyrg,xyvg->rv',Y1_abab,t2_2_ab,optimize=True)
            TY_b[nocc_b:,nocc_b:] += lib.einsum('xygr,xygv->rv',Y1_abab,t2_2_ab,optimize=True)
            del t2_2_ab

 #           # T_U2_Y1 vp block

            t2_1_a = t2[0][0][:]
            TY_a[nocc_a:,:nocc_a] += 0.25*lib.einsum('pxgh,yv,xygh->vp',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[nocc_a:,:nocc_a] += 0.25*np.einsum('xyvg,ph,xygh->vp',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[:nocc_a,nocc_a:] -= 0.5*np.einsum('xygh,pg,xyvh->pv',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[:nocc_a,nocc_a:] += 0.5*np.einsum('xygh,yv,pxgh->pv',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            del t2_1_a

            t2_1_ab = t2[0][1][:]
            TY_a[nocc_a:,:nocc_a] -= 0.5*np.einsum('xyvg,ph,xyhg->vp',Y1_abab,t1_1_a,t2_1_ab,optimize=True)
            TY_a[nocc_a:,:nocc_a] -= 0.5*lib.einsum('pxgh,yv,yxgh->vp',Y1_abab,t1_1_a,t2_1_ab,optimize=True)
            TY_b[nocc_b:,:nocc_b] -= 0.5*lib.einsum('xpgh,yv,xygh->vp',Y1_abab,t1_1_b,t2_1_ab,optimize=True)
            TY_b[nocc_b:,:nocc_b] -= 0.5*np.einsum('xygv,ph,xygh->vp',Y1_abab,t1_1_b,t2_1_ab,optimize=True)
            TY_a[:nocc_a,nocc_a:] -= np.einsum('xygh,pg,xyvh->pv',Y1_abab,t1_1_a,t2_1_ab,optimize=True)
            TY_b[:nocc_b,nocc_b:] -= np.einsum('xyhg,pg,xyhv->pv',Y1_abab,t1_1_b,t2_1_ab,optimize=True)
            TY_a[:nocc_a,nocc_a:] -= np.einsum('yxgh,yv,pxgh->pv',Y1_abab,t1_1_a,t2_1_ab,optimize=True)
            TY_b[:nocc_b,nocc_b:] -= np.einsum('xyhg,yv,xphg->pv',Y1_abab,t1_1_b,t2_1_ab,optimize=True)
            del t2_1_ab


            t2_1_b = t2[0][2][:]
            TY_b[nocc_b:,:nocc_b] += 0.25*lib.einsum('pxgh,yv,xygh->vp',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[nocc_b:,:nocc_b] += 0.25*np.einsum('xyvg,ph,xygh->vp',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[:nocc_b,nocc_b:] -= 0.5*np.einsum('xygh,pg,xyvh->pv',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[:nocc_b,nocc_b:] += 0.5*np.einsum('xygh,yv,pxgh->pv',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            del t2_1_b

            del Y1_abab
            del Y1_oovv_u_a
            del Y1_oovv_u_b

        TY = (TY_a, TY_b)


        dx = lib.einsum("rqp,qp->r", dm_a, TY_a, optimize = True)
        dx += lib.einsum("rqp,qp->r", dm_b, TY_b, optimize = True)

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
#
#
#def analyze_spec_factor(adc):
#
#    X_a = adc.X[0]
#    X_b = adc.X[1]
#
#    logger.info(adc, "Print spectroscopic factors > %E\n", adc.spec_factor_print_tol)
#
#    X_tot = (X_a, X_b)
#
#    for iter_idx, X in enumerate(X_tot):
#        if iter_idx == 0:
#            spin = "alpha"
#        else:
#            spin = "beta"
#
#        X_2 = (X.copy()**2)
#
#        thresh = adc.spec_factor_print_tol
#
#        for i in range(X_2.shape[1]):
#
#            sort = np.argsort(-X_2[:,i])
#            X_2_row = X_2[:,i]
#
#            X_2_row = X_2_row[sort]
#
#            if not adc.mol.symmetry:
#                sym = np.repeat(['A'], X_2_row.shape[0])
#            else:
#                if spin == "alpha":
#                    sym = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff[0].orbsym]
#                    sym = np.array(sym)
#                else:
#                    sym = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff[1].orbsym]
#                    sym = np.array(sym)
#
#                sym = sym[sort]
#
#            spec_Contribution = X_2_row[X_2_row > thresh]
#            index_mo = sort[X_2_row > thresh]+1
#
#            if np.sum(spec_Contribution) == 0.0:
#                continue
#
#            logger.info(adc, '%s | root %d %s\n', adc.method, i, spin)
#            logger.info(adc, "     HF MO     Spec. Contribution     Orbital symmetry")
#            logger.info(adc, "-----------------------------------------------------------")
#
#            for c in range(index_mo.shape[0]):
#                logger.info(adc, '     %3.d          %10.8f                %s',
#                            index_mo[c], spec_Contribution[c], sym[c])
#
#            logger.info(adc, '\nPartial spec. factor sum = %10.8f', np.sum(spec_Contribution))
#            logger.info(adc, "\n*************************************************************\n")


#@profile
def get_properties(adc, nroots=1):

    #Transition moments
    TY, dx  = adc.get_X()

    if adc.opdm is True or adc.tpdm is True or adc.spin_c is True:
        spin  = adc.get_spin_contamination()
    else:
        spin = None
    
    


    P = np.square(dx.T)*adc.E*(2/3)
    P = P[0] + P[1] + P[2]


    return P, (TY,spin)


#def analyze(myadc):
#
#    header = ("\n*************************************************************"
#              "\n           Eigenvector analysis summary"
#              "\n*************************************************************")
#    logger.info(myadc, header)
#
#    myadc.analyze_eigenvector()
#
#    if myadc.compute_properties:
#
#        header = ("\n*************************************************************"
#                  "\n            Spectroscopic factors analysis summary"
#                  "\n*************************************************************")
#        logger.info(myadc, header)
#
#        myadc.analyze_spec_factor()


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
    get_spin_contamination = get_spin_contamination
#    get_trans_moments = get_trans_moments
    get_properties = get_properties

 #   analyze_spec_factor = analyze_spec_factor
 #   analyze_eigenvector = analyze_eigenvector
#    analyze = analyze
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

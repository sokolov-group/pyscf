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

@profile
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


    if adc.f_ov is None:
        f_ov_a = np.zeros((nocc_a, nvir_a))
        f_ov_b = np.zeros((nocc_b, nvir_b))
        t1_1_a = np.zeros((nocc_a, nvir_a))
        t1_1_b = np.zeros((nocc_b, nvir_b))
    else:
        f_ov_a, f_ov_b = adc.f_ov


        t1_1_a = t1[2][0][:]
        t1_1_b = t1[2][1][:]



    t2_1_a = t2[0][0][:]
    t2_1_ab = t2[0][1][:]
    t2_1_b = t2[0][2][:]


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
    M_a = np.zeros((dim_a,dim_a))
    M_b = np.zeros((dim_b,dim_b))

#    M_ia_jb_a = np.zeros((nocc_a,nvir_a,nocc_a,nvir_a))
#    M_ia_jb_b = np.zeros((nocc_b,nvir_b,nocc_b,nvir_b))
#    M_aabb = np.zeros((nocc_a,nvir_a,nocc_b,nvir_b))

    if eris is None:
        eris = adc.transform_integrals()

    d_ai_a = adc.mo_energy_a[nocc_a:][:,None] - adc.mo_energy_a[:nocc_a]
    np.fill_diagonal(M_a, d_ai_a.transpose().reshape(-1))
    d_ai_b = adc.mo_energy_b[nocc_b:][:,None] - adc.mo_energy_b[:nocc_b]
    np.fill_diagonal(M_b, d_ai_b.transpose().reshape(-1))

    M_ia_jb_a = -lib.einsum('ijba->iajb', eris.oovv, optimize = True).copy()
    M_ia_jb_a += lib.einsum('jbai->iajb', eris.ovvo, optimize = True)
    M_ia_jb_b = -lib.einsum('ijba->iajb', eris.OOVV, optimize = True).copy()
    M_ia_jb_b += lib.einsum('jbai->iajb', eris.OVVO, optimize = True)
    M_aabb = lib.einsum('jbai->iajb', eris.OVvo, optimize = True).copy()

#    #M^(2)_0 term 3 iemf
#

    vir_list_a = np.array(range(nvir_a))
    vir_list_b = np.array(range(nvir_b))
    occ_list_a = np.array(range(nocc_a))
    occ_list_b = np.array(range(nocc_b))

    M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.25*lib.einsum('jmef,iefm->ij',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.5*lib.einsum('jmef,iefm->ij',t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] += 0.25*lib.einsum('jmef,ifem->ij',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.25*lib.einsum('jmef,iefm->ij',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.5*lib.einsum('mjfe,iefm->ij',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] += 0.25*lib.einsum('jmef,ifem->ij',t2_1_b, eris.OVVO, optimize = True)

    #M^(2)_0 term 4
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.25*lib.einsum('imef,jefm->ij',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.5*lib.einsum('imef,jefm->ij',t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a[:, vir_list_a, :, vir_list_a] += 0.25*lib.einsum('imef,jfem->ij',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.25*lib.einsum('imef,jefm->ij',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.5*lib.einsum('mife,jefm->ij',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b[:, vir_list_b, :, vir_list_b] += 0.25*lib.einsum('imef,jfem->ij',t2_1_b, eris.OVVO, optimize = True)

    #M^(2)_0 term 5
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.25*lib.einsum('mnae,mben->ab',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.5*lib.einsum('mnae,mben->ab',t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] += 0.25*lib.einsum('mnae,mebn->ab',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.25*lib.einsum('mnae,mben->ab',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.5*lib.einsum('nmea,mben->ab',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 0.25*lib.einsum('mnae,mebn->ab',t2_1_b, eris.OVVO, optimize = True)
#
#    #M^(2)_0 term 6
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.25*lib.einsum('mnbe,maen->ab',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.5*lib.einsum('mnbe,maen->ab',t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a[occ_list_a, :, occ_list_a, :] += 0.25*lib.einsum('mnbe,mean->ab',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.25*lib.einsum('mnbe,maen->ab',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.5*lib.einsum('nmeb,maen->ab',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b[occ_list_b, :, occ_list_b, :] += 0.25*lib.einsum('mnbe,mean->ab',t2_1_b, eris.OVVO, optimize = True)
#
#    #M^(2)_0 term 7
    M_ia_jb_a += 0.5*lib.einsum('jmbe,iaem->iajb', t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a += 0.5*lib.einsum('jmbe,iaem->iajb', t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a -= 0.5*lib.einsum('jmbe,ieam->iajb',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_b += 0.5*lib.einsum('jmbe,iaem->iajb',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b += 0.5*lib.einsum('mjeb,iaem->iajb',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b -= 0.5*lib.einsum('jmbe,ieam->iajb',t2_1_b, eris.OVVO, optimize = True)
    M_aabb += 0.5*lib.einsum('mjeb,iaem->iajb',t2_1_ab, eris.ovvo, optimize = True)
    M_aabb += 0.5*lib.einsum('mjeb,iaem->iajb',t2_1_b, eris.ovVO, optimize = True)
    M_aabb -= 0.5*lib.einsum('mjeb,ieam->iajb',t2_1_ab, eris.ovvo, optimize = True)

#
#    #M^(2)_0 term 8
    M_ia_jb_a += 0.5*lib.einsum('imae,jbem->iajb', t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_a += 0.5*lib.einsum('imae,jbem->iajb', t2_1_ab, eris.ovVO, optimize = True)
    M_ia_jb_a -= 0.5*lib.einsum('imae,jebm->iajb',t2_1_a, eris.ovvo, optimize = True)
    M_ia_jb_b += 0.5*lib.einsum('imae,jbem->iajb',t2_1_b, eris.OVVO, optimize = True)
    M_ia_jb_b += 0.5*lib.einsum('miea,jbem->iajb',t2_1_ab, eris.OVvo, optimize = True)
    M_ia_jb_b -= 0.5*lib.einsum('imae,jebm->iajb',t2_1_b, eris.OVVO, optimize = True)
    M_aabb += 0.5*lib.einsum('imae,jbem->iajb',t2_1_a, eris.OVvo, optimize = True)
    M_aabb += 0.5*lib.einsum('imae,jbem->iajb',t2_1_ab, eris.OVVO, optimize = True)
    M_aabb -= 0.5*lib.einsum('imae,jebm->iajb',t2_1_ab, eris.OVVO, optimize = True)
#    print("M_b norm", np.linalg.norm(M_aabb))
#    exit()

############################################################################
# k   A_temp -= 0.5*np.einsum('ab,je,ie->iajb',idn_vir,t1_1,f_ia,optimize = True)
# k   A_temp -= 0.5*np.einsum('ab,ie,je->iajb',idn_vir,t1_1,f_ia,optimize = True)
# k   A_temp -= 0.5*np.einsum('ij,ma,mb->iajb',idn_occ,f_ia,t1_1,optimize = True)
# k   A_temp -= 0.5*np.einsum('ij,mb,ma->iajb',idn_occ,f_ia,t1_1,optimize = True)
# k   A_temp -= np.einsum('ab,me,imje->iajb',idn_vir,t1_1,v2e_so_ooov,optimize = True)
# k   A_temp -= np.einsum('ab,me,jmie->iajb',idn_vir,t1_1,v2e_so_ooov,optimize = True)
# k   A_temp += np.einsum('ij,me,bmae->iajb',idn_occ,t1_1,v2e_so_vovv,optimize = True)
# k   A_temp += np.einsum('ij,me,beam->iajb',idn_occ,t1_1,v2e_so_vvvo,optimize = True)
# k   A_temp -= np.einsum('je,aebi->iajb',t1_1,v2e_so_vvvo,optimize = True)
# k   A_temp -= np.einsum('ma,jmbi->iajb',t1_1,v2e_so_oovo,optimize = True)
# k   A_temp -= np.einsum('mb,imaj->iajb',t1_1,v2e_so_oovo,optimize = True)
# k   A_temp -= np.einsum('ie,beaj->iajb',t1_1,v2e_so_vvvo,optimize = True)
############################################################################

    if isinstance(adc._scf, scf.rohf.ROHF):

        M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.5*lib.einsum('je,ie->ij',t1_1_a,f_ov_a,optimize = True)
        M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 0.5*lib.einsum('ie,je->ij',t1_1_a,f_ov_a,optimize = True)
        M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.5*lib.einsum('ma,mb->ab',f_ov_a,t1_1_a,optimize = True)
        M_ia_jb_a[occ_list_a, :, occ_list_a, :] -= 0.5*lib.einsum('mb,ma->ab',f_ov_a,t1_1_a,optimize = True)

        M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 2*lib.einsum('me,meij->ij',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_a[:, vir_list_a, :, vir_list_a] += lib.einsum('me,iemj->ij',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_a[:, vir_list_a, :, vir_list_a] -= 2*lib.einsum('me,meij->ij',t1_1_b,eris.OVoo,optimize = True)

        M_ia_jb_a[:, vir_list_a, :, vir_list_a] += lib.einsum('me,jemi->ij',t1_1_a,eris.ovoo,optimize = True)

        M_ia_jb_a -= lib.einsum('ma,jbmi->iajb',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_a += lib.einsum('ma,mbji->iajb',t1_1_a,eris.ovoo,optimize = True)
        M_aabb -= lib.einsum('ma,jbmi->iajb',t1_1_a,eris.OVoo,optimize = True)


        M_ia_jb_a -= lib.einsum('mb,iamj->iajb',t1_1_a,eris.ovoo,optimize = True)
        M_ia_jb_a += lib.einsum('mb,maij->iajb',t1_1_a,eris.ovoo,optimize = True)
        M_aabb -= lib.einsum('mb,iamj->iajb',t1_1_b,eris.ovOO,optimize = True)

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




        M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.5*lib.einsum('je,ie->ij',t1_1_b,f_ov_b,optimize = True)
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 0.5*lib.einsum('ie,je->ij',t1_1_b,f_ov_b,optimize = True)
        M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.5*lib.einsum('ma,mb->ab',f_ov_b,t1_1_b,optimize = True)
        M_ia_jb_b[occ_list_b, :, occ_list_b, :] -= 0.5*lib.einsum('mb,ma->ab',f_ov_b,t1_1_b,optimize = True)

        M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 2*lib.einsum('me,meij->ij',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] += lib.einsum('me,iemj->ij',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b[:, vir_list_b, :, vir_list_b] -= 2*lib.einsum('me,meij->ij',t1_1_a,eris.ovOO,optimize = True)

        M_ia_jb_b[:, vir_list_b, :, vir_list_b] += lib.einsum('me,jemi->ij',t1_1_b,eris.OVOO,optimize = True)

        M_ia_jb_b -= lib.einsum('ma,jbmi->iajb',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b += lib.einsum('ma,mbji->iajb',t1_1_b,eris.OVOO,optimize = True)

        M_ia_jb_b -= lib.einsum('mb,iamj->iajb',t1_1_b,eris.OVOO,optimize = True)
        M_ia_jb_b += lib.einsum('mb,maij->iajb',t1_1_b,eris.OVOO,optimize = True)


    M_a += M_ia_jb_a.reshape(n_singles_a, n_singles_a)
    M_b += M_ia_jb_b.reshape(n_singles_b, n_singles_b)
    M_aabb = M_aabb.reshape(n_singles_a, n_singles_b)
        
#    print("M_a norm", np.linalg.norm(M_a.reshape(nocc_a,nvir_a,nocc_a,nvir_a)))
#    print("M_b norm", np.linalg.norm(M_b.reshape(nocc_b,nvir_b,nocc_b,nvir_b)))
#    print("M_aabb norm", np.linalg.norm(M_aabb.reshape(nocc_a,nvir_a,nocc_b,nvir_b)))
#    exit()
    M_ia_jb = (M_a, M_b, M_aabb)

    cput0 = log.timer_debug1("Completed M_ia_jb  ADC calculation", *cput0)




    return M_ia_jb


@profile
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

    if eris is None:
        eris = adc.transform_integrals()

    d_ij_a = e_occ_a[:,None]+e_occ_a
    d_ij_b = e_occ_b[:,None]+e_occ_b
    d_ab_a = e_vir_a[:,None]+e_vir_a
    d_ab_b = e_vir_b[:,None]+e_vir_b
    D_n_a = -d_ij_a.reshape(-1,1) + d_ab_a.reshape(-1)
    D_n_b = -d_ij_b.reshape(-1,1) + d_ab_b.reshape(-1)
    D_n_a = D_n_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))
    D_n_b = D_n_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))

    D_ijab_a = D_n_a.copy()[:,:,ab_ind_a[0],ab_ind_a[1]]
    D_ijab_a = D_ijab_a.copy()[ij_ind_a[0],ij_ind_a[1]].reshape(-1)
    D_ijab_b = D_n_b.copy()[:,:,ab_ind_b[0],ab_ind_b[1]]
    D_ijab_b = D_ijab_b.copy()[ij_ind_b[0],ij_ind_b[1]].reshape(-1)

    d_ij_abab = e_occ_a[:,None]+e_occ_b
    d_ab_abab = e_vir_a[:,None]+e_vir_b
    D_n_abab = -d_ij_abab.reshape(-1,1) + d_ab_abab.reshape(-1)
    D_ijab_abab = D_n_abab.reshape(-1)

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

    if M_ia_jb is None:
        M_ia_jb  = adc.get_imds()

    M_ia_jb_a, M_ia_jb_b, M_aabb = M_ia_jb[0], M_ia_jb[1], M_ia_jb[2] 

    diag = np.zeros(dim)

    # Compute precond
    M_ia_jb_a_diag = np.diagonal(M_ia_jb_a)
    M_ia_jb_b_diag = np.diagonal(M_ia_jb_b)

    diag[s_a:f_a] = M_ia_jb_a_diag
    diag[s_b:f_b] = M_ia_jb_b_diag

    # Compute precond

    diag[s_aaaa:f_aaaa] = D_ijab_a
    diag[s_abab:f_abab] = D_ijab_abab
    diag[s_bbbb:f_bbbb] = D_ijab_b

#    print("diag", np.linalg.norm(diag))
#    exit()
  

    return diag


@profile
def matvec(adc, M_ia_jb=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ia_jb is None:
        M_ia_jb  = adc.get_imds()

    M_ia_jb_a, M_ia_jb_b, M_aabb = M_ia_jb[0], M_ia_jb[1], M_ia_jb[2]


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
    
    if eris is None:
        eris = adc.transform_integrals()

    diag = get_diag(adc)

    #print(diag.shape)

    #Calculate sigma vector
    @profile
    def sigma_(r):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]

        r_a_ov = r_a.reshape(nocc_a, nvir_a)
        r_b_ov = r_b.reshape(nocc_b, nvir_b)
        
        r_aaaa = r[s_aaaa:f_aaaa]
        r_abab = r[s_abab:f_ab]
        r_bbbb = r[s_bbbb:f_bbbb]

        r_aaaa = r_aaaa.reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2)).copy()
        r_bbbb = r_bbbb.reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2)).copy()

        r_vv_u_a = None
        r_vv_u_a = np.zeros((int((nocc_a * (nocc_a - 1))/2),nvir_a, nvir_a))
        r_vv_u_a[:,ab_ind_a[0],ab_ind_a[1]]= r_aaaa.copy()
        r_vv_u_a[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaaa.copy()
        r_oovv_u_a = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        r_oovv_u_a[ij_ind_a[0],ij_ind_a[1],:,:]= r_vv_u_a.copy()
        r_oovv_u_a[ij_ind_a[1],ij_ind_a[0],:,:]= -r_vv_u_a.copy()
        
        r_oovv_a = r_oovv_u_a.copy()[:,:,ab_ind_a[0],ab_ind_a[1]]
        r_packed_a = r_oovv_a.copy()[ij_ind_a[0],ij_ind_a[1]].reshape(-1)

        r_vv_u_b = None
        r_vv_u_b = np.zeros((int((nocc_b * (nocc_b - 1))/2),nvir_b, nvir_b))
        r_vv_u_b[:,ab_ind_b[0],ab_ind_b[1]]= r_bbbb.copy()
        r_vv_u_b[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbbb.copy()
        r_oovv_u_b = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        r_oovv_u_b[ij_ind_b[0],ij_ind_b[1],:,:]= r_vv_u_b.copy()
        r_oovv_u_b[ij_ind_b[1],ij_ind_b[0],:,:]= -r_vv_u_b.copy()

        r_oovv_b = r_oovv_u_b.copy()[:,:,ab_ind_b[0],ab_ind_b[1]]
        r_packed_b = r_oovv_b.copy()[ij_ind_b[0],ij_ind_b[1]].reshape(-1)


        s = np.zeros(dim)


############## ADC(2) 1 block ############################
#
        
        s[s_a:f_a] = lib.einsum('ab,b->a',M_ia_jb_a,r_a, optimize = True)
        s[s_a:f_a] += lib.einsum('ab,b->a',M_aabb,  r_b, optimize = True)
       
        s[s_b:f_b] = lib.einsum('ab,b->a',M_ia_jb_b,r_b, optimize = True)
        s[s_b:f_b] += lib.einsum('ab,b->a',M_aabb.T,r_a, optimize = True)

        D_ijab_a = diag[s_aaaa:f_aaaa]
        D_ijab_abab = diag[s_abab:f_ab]
        D_ijab_b = diag[s_bbbb:f_bbbb]
        
        s[s_aaaa:f_aaaa] = D_ijab_a*r_packed_a
        s[s_abab:f_ab] = D_ijab_abab*r_abab
        s[s_bbbb:f_bbbb] = D_ijab_b*r_packed_b

        r_abab = r_abab.reshape(nocc_a, nocc_b, nvir_a, nvir_b)
        # M^(1)_h0_h1
        temp_a = np.zeros((nocc_a, nocc_a, nvir_a, nvir_a))

        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovvv = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                k = eris_ovvv.shape[0]
                s[s_a:f_a] += 0.5*lib.einsum('imef,mfea->ia',r_oovv_u_a[:,a:a+k], eris_ovvv, optimize = True).reshape(-1)
                s[s_a:f_a] -= 0.5*lib.einsum('imef,mefa->ia',r_oovv_u_a[:,a:a+k], eris_ovvv, optimize = True).reshape(-1)
                temp_a[:,a:a+k] = -lib.einsum('ie,jabe->ijab',r_a_ov, eris_ovvv, optimize = True)
                temp_a[:,a:a+k] += lib.einsum('ie,jbae->ijab',r_a_ov, eris_ovvv, optimize = True)
                temp_a[a:a+k] += lib.einsum('je,iabe->ijab',r_a_ov, eris_ovvv, optimize = True)
                temp_a[a:a+k] -= lib.einsum('je,ibae->ijab',r_a_ov, eris_ovvv, optimize = True)
                del eris_ovvv
                a += k
        else:
            eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            s[s_a:f_a] += 0.5*lib.einsum('imef,mfea->ia',r_oovv_u_a, eris_ovvv, optimize = True).reshape(-1)
            s[s_a:f_a] -= 0.5*lib.einsum('imef,mefa->ia',r_oovv_u_a, eris_ovvv, optimize = True).reshape(-1)
            temp_a = -lib.einsum('ie,jabe->ijab',r_a_ov, eris_ovvv, optimize = True)
            temp_a += lib.einsum('ie,jbae->ijab',r_a_ov, eris_ovvv, optimize = True)
            temp_a += lib.einsum('je,iabe->ijab',r_a_ov, eris_ovvv, optimize = True)
            temp_a -= lib.einsum('je,ibae->ijab',r_a_ov, eris_ovvv, optimize = True)
        temp_a = temp_a[:,:,ab_ind_a[0],ab_ind_a[1]]
        s[s_aaaa:f_aaaa] += temp_a[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)
        del temp_a

        temp_abab = np.zeros((nocc_a, nocc_b, nvir_a, nvir_b))
        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVvv = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                k = eris_OVvv.shape[0]
                s[s_a:f_a] += lib.einsum('imef,mfea->ia',r_abab[:,a:a+k], eris_OVvv, optimize = True).reshape(-1)
                temp_abab[:,a:a+k] = lib.einsum('ie,jbae->ijab',r_a_ov, eris_OVvv, optimize = True)
                del eris_OVvv
                a += k
        else:
            eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            s[s_a:f_a] += lib.einsum('imef,mfea->ia',r_abab, eris_OVvv, optimize = True).reshape(-1)
            temp_abab = lib.einsum('ie,jbae->ijab',r_a_ov, eris_OVvv, optimize = True)
        s[s_abab:f_ab] += temp_abab.reshape(-1)
        del temp_abab
#
#
        temp_b = np.zeros((nocc_b, nocc_b, nvir_b, nvir_b))
        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVVV = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                k = eris_OVVV.shape[0]
                s[s_b:f_b] += 0.5*lib.einsum('imef,mfea->ia',r_oovv_u_b[:,a:a+k], eris_OVVV, optimize = True).reshape(-1)
                s[s_b:f_b] -= 0.5*lib.einsum('imef,mefa->ia',r_oovv_u_b[:,a:a+k], eris_OVVV, optimize = True).reshape(-1)
                temp_b[:,a:a+k] = -lib.einsum('ie,jabe->ijab',r_b_ov, eris_OVVV, optimize = True)
                temp_b[:,a:a+k] += lib.einsum('ie,jbae->ijab',r_b_ov, eris_OVVV, optimize = True)
                temp_b[a:a+k] += lib.einsum('je,iabe->ijab',r_b_ov, eris_OVVV, optimize = True)
                temp_b[a:a+k] -= lib.einsum('je,ibae->ijab',r_b_ov, eris_OVVV, optimize = True)
                del eris_OVVV
                a += k
        else:
            eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            s[s_b:f_b] += 0.5*lib.einsum('imef,mfea->ia',r_oovv_u_b, eris_OVVV, optimize = True).reshape(-1)
            s[s_b:f_b] -= 0.5*lib.einsum('imef,mefa->ia',r_oovv_u_b, eris_OVVV, optimize = True).reshape(-1)
            temp_b = -lib.einsum('ie,jabe->ijab',r_b_ov, eris_OVVV, optimize = True)
            temp_b += lib.einsum('ie,jbae->ijab',r_b_ov, eris_OVVV, optimize = True)
            temp_b += lib.einsum('je,iabe->ijab',r_b_ov, eris_OVVV, optimize = True)
            temp_b -= lib.einsum('je,ibae->ijab',r_b_ov, eris_OVVV, optimize = True)
        temp_b = temp_b[:,:,ab_ind_b[0],ab_ind_b[1]]
        s[s_bbbb:f_bbbb] += temp_b[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
        del temp_b

#
        temp_abab = np.zeros((nocc_a, nocc_b, nvir_a, nvir_b))
        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovVV = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                k = eris_ovVV.shape[0]
                s[s_b:f_b] += lib.einsum('mife,mfea->ia',r_abab[a:a+k], eris_ovVV, optimize = True).reshape(-1)
                temp_abab[a:a+k] = lib.einsum('je,iabe->ijab',r_b_ov, eris_ovVV, optimize = True)
                del eris_ovVV
                a += k
        else:
            eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            s[s_b:f_b] += lib.einsum('mife,mfea->ia',r_abab, eris_ovVV, optimize = True).reshape(-1)
            temp_abab = lib.einsum('je,iabe->ijab',r_b_ov, eris_ovVV, optimize = True)
        s[s_abab:f_ab] += temp_abab.reshape(-1)
        del temp_abab

        s[s_a:f_a] -= 0.5*lib.einsum('mnae,neim->ia',r_oovv_u_a, eris.ovoo, optimize = True).reshape(-1)
        s[s_a:f_a] += 0.5*lib.einsum('mnae,mein->ia',r_oovv_u_a, eris.ovoo, optimize = True).reshape(-1)
        s[s_a:f_a] -= lib.einsum('mnae,neim->ia',r_abab, eris.OVoo, optimize = True).reshape(-1)

        s[s_b:f_b] -= 0.5*lib.einsum('mnae,neim->ia',r_oovv_u_b, eris.OVOO, optimize = True).reshape(-1)
        s[s_b:f_b] += 0.5*lib.einsum('mnae,mein->ia',r_oovv_u_b, eris.OVOO, optimize = True).reshape(-1)
        s[s_b:f_b] -= lib.einsum('mnea,mein->ia',r_abab, eris.ovOO, optimize = True).reshape(-1)

#        # # M^(1)_h1_h0

        temp_a = lib.einsum('ma,ibjm->ijab',r_a_ov, eris.ovoo, optimize = True)
        temp_a -= lib.einsum('ma,jbim->ijab',r_a_ov, eris.ovoo, optimize = True)
        temp_abab = -lib.einsum('ma,jbim->ijab',r_a_ov, eris.OVoo, optimize = True)

        temp_b = lib.einsum('ma,ibjm->ijab',r_b_ov, eris.OVOO, optimize = True)
        temp_b -= lib.einsum('ma,jbim->ijab',r_b_ov, eris.OVOO, optimize = True)

        temp_a -= lib.einsum('mb,iajm->ijab',r_a_ov, eris.ovoo, optimize = True)
        temp_a += lib.einsum('mb,jaim->ijab',r_a_ov, eris.ovoo, optimize = True)

        temp_b -= lib.einsum('mb,iajm->ijab',r_b_ov, eris.OVOO, optimize = True)
        temp_b += lib.einsum('mb,jaim->ijab',r_b_ov, eris.OVOO, optimize = True)
        temp_abab -= lib.einsum('mb,iajm->ijab',r_b_ov, eris.ovOO, optimize = True)
        
        temp_a = temp_a[:,:,ab_ind_a[0],ab_ind_a[1]]
        s[s_aaaa:f_aaaa] += temp_a[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)

        s[s_abab:f_ab] += temp_abab.reshape(-1)

        temp_b = temp_b[:,:,ab_ind_b[0],ab_ind_b[1]]
        s[s_bbbb:f_bbbb] += temp_b[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
        #exit()

#        print("norm of s after", np.linalg.norm(s))

        if (method == "adc(2)-x"):
            
            if isinstance(eris.vvvv_p, np.ndarray):
                interim = np.ascontiguousarray(r_oovv_u_a[:,:,ab_ind_a[0],ab_ind_a[1]]).reshape(nocc_a*nocc_a,-1)
                interim_1 = np.dot(interim,eris.vvvv_p.T).reshape(nocc_a, nocc_a, -1)
                s[s_aaaa:f_aaaa] += interim_1[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)

            elif isinstance(eris.vvvv_p, list) :
                interim_1 = uadc_amplitudes.contract_ladder_antisym(adc,r_oovv_u_a,eris.vvvv_p)
                s[s_aaaa:f_aaaa] += interim_1[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)

            else:
                interim_1 = uadc_amplitudes.contract_ladder_antisym(adc,r_oovv_u_a,eris.Lvv)
                s[s_aaaa:f_aaaa] += interim_1[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)


            if isinstance(eris.vVvV_p, np.ndarray):
                interim_abab = np.dot(r_abab.reshape(nocc_a*nocc_b,nvir_a*nvir_b),eris.vVvV_p.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
            elif isinstance(eris.vVvV_p, list):
                interim_abab = uadc_amplitudes.contract_ladder(adc,r_abab,eris.vVvV_p)
            else:
                interim_abab = uadc_amplitudes.contract_ladder(adc,r_abab,(eris.Lvv,eris.LVV))

            if isinstance(eris.VVVV_p, np.ndarray):
                interim = np.ascontiguousarray(r_oovv_u_b[:,:,ab_ind_b[0],ab_ind_b[1]]).reshape(nocc_b*nocc_b,-1)
                interim_2 = np.dot(interim,eris.VVVV_p.T).reshape(nocc_b, nocc_b, -1)
                s[s_bbbb:f_bbbb] += interim_2[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)

            elif isinstance(eris.VVVV_p, list) :
                interim_1 = uadc_amplitudes.contract_ladder_antisym(adc,r_oovv_u_b,eris.VVVV_p)
                s[s_bbbb:f_bbbb] += interim_1[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
            else:
                interim_1 = uadc_amplitudes.contract_ladder_antisym(adc,r_oovv_u_b,eris.LVV)
                s[s_bbbb:f_bbbb] += interim_1[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)

            interim_a = lib.einsum('imae,jbem->ijab', r_oovv_u_a, eris.ovvo, optimize = True)
            interim_a -= lib.einsum('imae,mjbe->ijab', r_oovv_u_a, eris.oovv, optimize = True)
            interim_a += lib.einsum('imae,jbem->ijab', r_abab, eris.ovVO, optimize = True)

            interim_abab += lib.einsum('imae,jbem->ijab', r_abab, eris.OVVO, optimize = True)
            interim_abab -= lib.einsum('imae,mjbe->ijab', r_abab, eris.OOVV, optimize = True)
            interim_abab += lib.einsum('imae,mebj->ijab', r_oovv_u_a, eris.ovVO, optimize = True)
            
            interim_b = lib.einsum('imae,jbem->ijab', r_oovv_u_b, eris.OVVO, optimize = True)
            interim_b -= lib.einsum('imae,mjbe->ijab', r_oovv_u_b, eris.OOVV, optimize = True)
            interim_b += lib.einsum('miea,mebj->ijab', r_abab, eris.ovVO, optimize = True)

            interim_a -= lib.einsum('jmae,ibem->ijab', r_oovv_u_a, eris.ovvo, optimize = True)
            interim_a += lib.einsum('jmae,mibe->ijab', r_oovv_u_a, eris.oovv, optimize = True)
            interim_a -= lib.einsum('jmae,ibem->ijab', r_abab, eris.ovVO, optimize = True)

            interim_abab -= lib.einsum('mjae,mibe->ijab', r_abab, eris.ooVV, optimize = True)

            interim_b -= lib.einsum('jmae,ibem->ijab', r_oovv_u_b, eris.OVVO, optimize = True)
            interim_b += lib.einsum('jmae,mibe->ijab', r_oovv_u_b, eris.OOVV, optimize = True)
            interim_b -= lib.einsum('mjea,mebi->ijab', r_abab, eris.ovVO, optimize = True)

            interim_a += 0.5*lib.einsum('mnab,minj->ijab', r_oovv_u_a, eris.oooo, optimize = True)
            interim_a -= 0.5*lib.einsum('mnab,mjni->ijab', r_oovv_u_a, eris.oooo, optimize = True)

            interim_abab += lib.einsum('mnab,minj->ijab', r_abab, eris.ooOO, optimize = True)

            interim_b += 0.5*lib.einsum('mnab,minj->ijab', r_oovv_u_b, eris.OOOO, optimize = True)
            interim_b -= 0.5*lib.einsum('mnab,mjni->ijab', r_oovv_u_b, eris.OOOO, optimize = True)

            interim_a -= lib.einsum('imbe,jaem->ijab', r_oovv_u_a, eris.ovvo, optimize = True)
            interim_a += lib.einsum('imbe,jmea->ijab', r_oovv_u_a, eris.oovv, optimize = True)
            interim_a -= lib.einsum('imbe,jaem->ijab', r_abab, eris.ovVO, optimize = True)
            
            interim_abab -= lib.einsum('imeb,jmea->ijab', r_abab, eris.OOvv, optimize = True)

            interim_b -= lib.einsum('imbe,jaem->ijab', r_oovv_u_b, eris.OVVO, optimize = True)
            interim_b += lib.einsum('imbe,jmea->ijab', r_oovv_u_b, eris.OOVV, optimize = True)
            interim_b -= lib.einsum('mieb,meaj->ijab', r_abab, eris.ovVO, optimize = True)

            interim_a += lib.einsum('jmbe,iaem->ijab', r_oovv_u_a, eris.ovvo, optimize = True)
            interim_a -= lib.einsum('jmbe,imea->ijab', r_oovv_u_a, eris.oovv, optimize = True)
            interim_a += lib.einsum('jmbe,iaem->ijab', r_abab, eris.ovVO, optimize = True)

            interim_abab += lib.einsum('mjeb,iaem->ijab', r_abab, eris.ovvo, optimize = True)
            interim_abab -= lib.einsum('mjeb,imea->ijab', r_abab, eris.oovv, optimize = True)
            interim_abab += lib.einsum('jmbe,iaem->ijab', r_oovv_u_b, eris.ovVO, optimize = True)
            
            interim_b += lib.einsum('jmbe,iaem->ijab', r_oovv_u_b, eris.OVVO, optimize = True)
            interim_b -= lib.einsum('jmbe,imea->ijab', r_oovv_u_b, eris.OOVV, optimize = True)
            interim_b += lib.einsum('mjeb,meai->ijab', r_abab, eris.ovVO, optimize = True)

            interim_a = interim_a[:,:,ab_ind_a[0],ab_ind_a[1]]
            s[s_aaaa:f_aaaa] += interim_a[ij_ind_a[0],ij_ind_a[1]].reshape(n_doubles_aaaa)

            s[s_abab:f_ab] += interim_abab.reshape(-1)

            interim_b = interim_b[:,:,ab_ind_b[0],ab_ind_b[1]]
            s[s_bbbb:f_bbbb] += interim_b[ij_ind_b[0],ij_ind_b[1]].reshape(n_doubles_bbbb)
            
        return s

    return sigma_

@profile
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

    t1_2_a, t1_2_b = adc.t1[0]
    t2_1_a = t2[0][0][:]
    t2_1_ab = t2[0][1][:]
    t2_1_b = t2[0][2][:]

    t2_2_a = t2[1][0][:]
    t2_2_ab = t2[1][1][:]
    t2_2_b = t2[1][2][:]

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
        Y_a = U[r][:f_a].reshape(nocc_a, nvir_a).copy()
        Y_b = U[r][f_a:f_b].reshape(nocc_b, nvir_b).copy()


        Y_aaaa = U[r][s_aaaa:f_aaaa]
        Y_abab = U[r][s_abab:f_ab]
        Y_bbbb = U[r][s_bbbb:f_bbbb]

        Y_aaaa = Y_aaaa.reshape(int((nocc_a * (nocc_a - 1))/2),int((nvir_a * (nvir_a - 1))/2)).copy()
        Y1_abab = Y_abab.reshape(nocc_a, nocc_b, nvir_a, nvir_b)
        Y_bbbb = Y_bbbb.reshape(int((nocc_b * (nocc_b - 1))/2),int((nvir_b * (nvir_b - 1))/2)).copy()

        Y_vv_u_a = None
        Y_vv_u_a = np.zeros((int((nocc_a * (nocc_a - 1))/2),nvir_a, nvir_a))
        Y_vv_u_a[:,ab_ind_a[0],ab_ind_a[1]]= Y_aaaa.copy()
        Y_vv_u_a[:,ab_ind_a[1],ab_ind_a[0]]= -Y_aaaa.copy()
        Y1_oovv_u_a = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        Y1_oovv_u_a[ij_ind_a[0],ij_ind_a[1],:,:]= Y_vv_u_a.copy()
        Y1_oovv_u_a[ij_ind_a[1],ij_ind_a[0],:,:]= -Y_vv_u_a.copy()

        Y_vv_u_b = None
        Y_vv_u_b = np.zeros((int((nocc_b * (nocc_b - 1))/2),nvir_b, nvir_b))
        Y_vv_u_b[:,ab_ind_b[0],ab_ind_b[1]]= Y_bbbb.copy()
        Y_vv_u_b[:,ab_ind_b[1],ab_ind_b[0]]= -Y_bbbb.copy()
        Y1_oovv_u_b = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        Y1_oovv_u_b[ij_ind_b[0],ij_ind_b[1],:,:]= Y_vv_u_b.copy()
        Y1_oovv_u_b[ij_ind_b[1],ij_ind_b[0],:,:]= -Y_vv_u_b.copy()

######################################################################################
#        # T_U1_Y0 qp block
#       K TY[:nocc_so,:nocc_so] = -np.einsum('pg,qg->qp',Y,t1_1,optimize=True)
#----
#        # T_U2_Y0 qp block
#      K  TY[:nocc_so,:nocc_so] += np.einsum('xg,ph,qxgh->qp',Y,t1_1,t2_1,optimize=True)
#      K  TY[:nocc_so,:nocc_so] -= 0.5*np.einsum('pg,xh,qxgh->qp',Y,t1_1,t2_1,optimize=True)
#----
#
#        # T_U1_Y0 rv block
#     K   TY[nocc_so:,nocc_so:] = np.einsum('xr,xv->rv',Y,t1_1,optimize=True)
##
##        # T_U2_Y0 rv block
#     K   TY[nocc_so:,nocc_so:] -= np.einsum('xg,yr,xyvg->rv',Y,t1_1,t2_1,optimize=True)
#     K    TY[nocc_so:,nocc_so:] += 0.5*np.einsum('xr,yg,xyvg->rv',Y,t1_1,t2_1,optimize=True)
##
##
##
##        # T_U2_Y0 vp block
#     K   TY[nocc_so:,:nocc_so] -= 0.5*np.einsum('pg,xg,xv->vp',Y,t1_1,t1_1,optimize=True)
#     K   TY[nocc_so:,:nocc_so] -= 0.5*np.einsum('xv,xg,pg->vp',Y,t1_1,t1_1,optimize=True)
##
##
##
##        # T_U2_Y0 pv block
#     K   TY[:nocc_so,nocc_so:] -= np.einsum('xg,pg,xv->pv',Y,t1_1,t1_1,optimize=True)
#
#
#
#
#        if (method == "adc(2)-e"):
#            # T_U2_Y1 vp block
#           K TY[nocc_so:,:nocc_so] += 0.25*np.einsum('pxgh,yv,xygh->vp',Y1,t1_1,t2_1,optimize=True)
#           K TY[nocc_so:,:nocc_so] += 0.25*np.einsum('xyvg,ph,xygh->vp',Y1,t1_1,t2_1,optimize=True)
#            # T_U2_Y1 pv block
#          K  TY[:nocc_so,nocc_so:] -= 0.5*np.einsum('xygh,pg,xyvh->pv',Y1,t1_1,t2_1,optimize=True)
#          k  TY[:nocc_so,nocc_so:] += 0.5*np.einsum('xygh,yv,pxgh->pv',Y1,t1_1,t2_1,optimize=True)
######################################################################################


        # T_U2_Y0 qp block
        TY_a[:nocc_a,:nocc_a] = -lib.einsum('pg,qg->qp',Y_a,t1_2_a,optimize=True)
        TY_b[:nocc_b,:nocc_b] = -lib.einsum('pg,qg->qp',Y_b,t1_2_b,optimize=True)

        TY_a[:nocc_a,:nocc_a] += lib.einsum('xg,ph,qxgh->qp',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('xg,ph,qxhg->qp',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_b[:nocc_b,:nocc_b] += lib.einsum('xg,ph,qxgh->qp',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('xg,ph,xqgh->qp',Y_a,t1_1_b,t2_1_ab,optimize=True)

        TY_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('pg,xh,qxgh->qp',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('pg,xh,qxgh->qp',Y_a,t1_1_b,t2_1_ab,optimize=True)

        TY_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('pg,xh,qxgh->qp',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('pg,xh,xqhg->qp',Y_b,t1_1_a,t2_1_ab,optimize=True)


       # T_U1_Y0 qp block
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('pg,qg->qp',Y_a,t1_1_a,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('pg,qg->qp',Y_b,t1_1_b,optimize=True)

        # T_U1_Y1 qp block
        TY_a[:nocc_a,:nocc_a] -= 0.5*lib.einsum('pxgh,qxgh->qp',Y1_oovv_u_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,:nocc_a] -= lib.einsum('pxgh,qxgh->qp',Y1_abab,t2_1_ab,optimize=True)

        TY_b[:nocc_b,:nocc_b] -= 0.5*lib.einsum('pxgh,qxgh->qp',Y1_oovv_u_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,:nocc_b] -= lib.einsum('xpgh,xqgh->qp',Y1_abab,t2_1_ab,optimize=True)


        # T_U2_Y0 rv block
        TY_a[nocc_a:,nocc_a:] = lib.einsum('xr,xv->rv',Y_a,t1_2_a,optimize=True)
        TY_b[nocc_b:,nocc_b:] = lib.einsum('xr,xv->rv',Y_b,t1_2_b,optimize=True)

        # T_U1_Y1 rv block
        TY_a[nocc_a:,nocc_a:] += 0.5*lib.einsum('xyrg,xyvg->rv',Y1_oovv_u_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] += lib.einsum('xyrg,xyvg->rv',Y1_abab,t2_1_ab,optimize=True)

        TY_b[nocc_b:,nocc_b:] += 0.5*lib.einsum('xyrg,xyvg->rv',Y1_oovv_u_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xygr,xygv->rv',Y1_abab,t2_1_ab,optimize=True)

        # T_U1_Y0 rv block
        TY_a[nocc_a:,nocc_a:] += lib.einsum('xr,xv->rv',Y_a,t1_1_a,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xr,xv->rv',Y_b,t1_1_b,optimize=True)

        TY_a[nocc_a:,nocc_a:] -= lib.einsum('xg,yr,xyvg->rv',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] += lib.einsum('xg,yr,yxvg->rv',Y_b,t1_1_a,t2_1_ab,optimize=True)
        TY_b[nocc_b:,nocc_b:] -= lib.einsum('xg,yr,xyvg->rv',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] += lib.einsum('xg,yr,xygv->rv',Y_a,t1_1_b,t2_1_ab,optimize=True)

        TY_a[nocc_a:,nocc_a:] += 0.5*np.einsum('xr,yg,xyvg->rv',Y_a,t1_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,nocc_a:] += 0.5*np.einsum('xr,yg,xyvg->rv',Y_a,t1_1_b,t2_1_ab,optimize=True)

        TY_b[nocc_b:,nocc_b:] += 0.5*np.einsum('xr,yg,xyvg->rv',Y_b,t1_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,nocc_b:] += 0.5*np.einsum('xr,yg,yxgv->rv',Y_b,t1_1_a,t2_1_ab,optimize=True)

        # T_U0_Y0 vp block
        TY_a[nocc_a:,:nocc_a] = Y_a.T.copy()
        TY_b[nocc_b:,:nocc_b] = Y_b.T.copy()


        # T_U2_Y0 vp block
        TY_a[nocc_a:,:nocc_a] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_a,t2_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,:nocc_a] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_a,t2_1_ab,t2_1_ab,optimize=True)
        TY_a[nocc_a:,:nocc_a] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_b,t2_1_b,t2_1_ab,optimize=True)
        TY_a[nocc_a:,:nocc_a] += 0.5*lib.einsum('xg,yxhg,pyvh->vp',Y_b,t2_1_ab,t2_1_a,optimize=True)

        TY_b[nocc_b:,:nocc_b] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,:nocc_b] += 0.5*lib.einsum('xg,yxhg,yphv->vp',Y_b,t2_1_ab,t2_1_ab,optimize=True)
        TY_b[nocc_b:,:nocc_b] += 0.5*lib.einsum('xg,xygh,yphv->vp',Y_a,t2_1_a,t2_1_ab,optimize=True)
        TY_b[nocc_b:,:nocc_b] += 0.5*lib.einsum('xg,xygh,pyvh->vp',Y_a,t2_1_ab,t2_1_b,optimize=True)


        TY_a[nocc_a:,:nocc_a] -= 0.25*lib.einsum('pg,xygh,xyvh->vp',Y_a,t2_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,:nocc_a] -= 0.5*lib.einsum('pg,xygh,xyvh->vp',Y_a,t2_1_ab,t2_1_ab,optimize=True)


        TY_b[nocc_b:,:nocc_b] -= 0.25*lib.einsum('pg,xygh,xyvh->vp',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.5*lib.einsum('pg,xyhg,xyhv->vp',Y_b,t2_1_ab,t2_1_ab,optimize=True)

        TY_a[nocc_a:,:nocc_a] -= 0.25*lib.einsum('xv,xygh,pygh->vp',Y_a,t2_1_a,t2_1_a,optimize=True)
        TY_a[nocc_a:,:nocc_a] -= 0.5*lib.einsum('xv,xygh,pygh->vp',Y_a,t2_1_ab,t2_1_ab,optimize=True)


        TY_b[nocc_b:,:nocc_b] -= 0.25*lib.einsum('xv,xygh,pygh->vp',Y_b,t2_1_b,t2_1_b,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.5*lib.einsum('xv,yxgh,ypgh->vp',Y_b,t2_1_ab,t2_1_ab,optimize=True)

        TY_a[nocc_a:,:nocc_a] -= 0.5*np.einsum('pg,xg,xv->vp',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.5*np.einsum('pg,xg,xv->vp',Y_b,t1_1_b,t1_1_b,optimize=True)

        TY_a[nocc_a:,:nocc_a] -= 0.5*np.einsum('xv,xg,pg->vp',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[nocc_b:,:nocc_b] -= 0.5*np.einsum('xv,xg,pg->vp',Y_b,t1_1_b,t1_1_b,optimize=True)



        # T_U1_Y0 pv block
        TY_a[:nocc_a,nocc_a:] = lib.einsum('xg,pxvg->pv',Y_a,t2_1_a,optimize=True)
        TY_a[:nocc_a,nocc_a:] += lib.einsum('xg,pxvg->pv',Y_b,t2_1_ab,optimize=True)

        TY_b[:nocc_b,nocc_b:] = lib.einsum('xg,pxvg->pv',Y_b,t2_1_b,optimize=True)
        TY_b[:nocc_b,nocc_b:] += lib.einsum('xg,xpgv->pv',Y_a,t2_1_ab,optimize=True)

        # T_U2_Y0 pv block
        TY_a[:nocc_a,nocc_a:] += lib.einsum('xg,pxvg->pv',Y_a,t2_2_a,optimize=True)
        TY_a[:nocc_a,nocc_a:] += lib.einsum('xg,pxvg->pv',Y_b,t2_2_ab,optimize=True)

        TY_b[:nocc_b,nocc_b:] += lib.einsum('xg,pxvg->pv',Y_b,t2_2_b,optimize=True)
        TY_b[:nocc_b,nocc_b:] += lib.einsum('xg,xpgv->pv',Y_a,t2_2_ab,optimize=True)

        TY_a[:nocc_a,nocc_a:] -= lib.einsum('xg,pg,xv->pv',Y_a,t1_1_a,t1_1_a,optimize=True)
        TY_b[:nocc_b,nocc_b:] -= lib.einsum('xg,pg,xv->pv',Y_b,t1_1_b,t1_1_b,optimize=True)


        if (method == "adc(2)-x"):
            # T_U2_Y1 qp block
            TY_a[:nocc_a,:nocc_a] -= 0.5*lib.einsum('pxgh,qxgh->qp',Y1_oovv_u_a,t2_2_a,optimize=True)
            TY_a[:nocc_a,:nocc_a] -= lib.einsum('pxgh,qxgh->qp',Y1_abab,t2_2_ab,optimize=True)

            TY_b[:nocc_b,:nocc_b] -= 0.5*lib.einsum('pxgh,qxgh->qp',Y1_oovv_u_b,t2_2_b,optimize=True)
            TY_b[:nocc_b,:nocc_b] -= lib.einsum('xpgh,xqgh->qp',Y1_abab,t2_2_ab,optimize=True)

            # T_U2_Y1 rv block
            TY_a[nocc_a:,nocc_a:] += 0.5*lib.einsum('xyrg,xyvg->rv',Y1_oovv_u_a,t2_2_a,optimize=True)
            TY_a[nocc_a:,nocc_a:] += lib.einsum('xyrg,xyvg->rv',Y1_abab,t2_2_ab,optimize=True)

            TY_b[nocc_b:,nocc_b:] += 0.5*lib.einsum('xyrg,xyvg->rv',Y1_oovv_u_b,t2_2_b,optimize=True)
            TY_b[nocc_b:,nocc_b:] += lib.einsum('xygr,xygv->rv',Y1_abab,t2_2_ab,optimize=True)

 #           # T_U2_Y1 vp block
            TY_a[nocc_a:,:nocc_a] += 0.25*lib.einsum('pxgh,yv,xygh->vp',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[nocc_a:,:nocc_a] -= 0.5*lib.einsum('pxgh,yv,yxgh->vp',Y1_abab,t1_1_a,t2_1_ab,optimize=True)

            TY_b[nocc_b:,:nocc_b] += 0.25*lib.einsum('pxgh,yv,xygh->vp',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[nocc_b:,:nocc_b] -= 0.5*lib.einsum('xpgh,yv,xygh->vp',Y1_abab,t1_1_b,t2_1_ab,optimize=True)

            TY_a[nocc_a:,:nocc_a] += 0.25*np.einsum('xyvg,ph,xygh->vp',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[nocc_a:,:nocc_a] -= 0.5*np.einsum('xyvg,ph,xyhg->vp',Y1_abab,t1_1_a,t2_1_ab,optimize=True)

            TY_b[nocc_b:,:nocc_b] += 0.25*np.einsum('xyvg,ph,xygh->vp',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[nocc_b:,:nocc_b] -= 0.5*np.einsum('xygv,ph,xygh->vp',Y1_abab,t1_1_b,t2_1_ab,optimize=True)

            TY_a[:nocc_a,nocc_a:] -= 0.5*np.einsum('xygh,pg,xyvh->pv',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[:nocc_a,nocc_a:] -= np.einsum('xygh,pg,xyvh->pv',Y1_abab,t1_1_a,t2_1_ab,optimize=True)

            TY_b[:nocc_b,nocc_b:] -= 0.5*np.einsum('xygh,pg,xyvh->pv',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[:nocc_b,nocc_b:] -= np.einsum('xyhg,pg,xyhv->pv',Y1_abab,t1_1_b,t2_1_ab,optimize=True)

            TY_a[:nocc_a,nocc_a:] += 0.5*np.einsum('xygh,yv,pxgh->pv',Y1_oovv_u_a,t1_1_a,t2_1_a,optimize=True)
            TY_a[:nocc_a,nocc_a:] -= np.einsum('yxgh,yv,pxgh->pv',Y1_abab,t1_1_a,t2_1_ab,optimize=True)

            TY_b[:nocc_b,nocc_b:] += 0.5*np.einsum('xygh,yv,pxgh->pv',Y1_oovv_u_b,t1_1_b,t2_1_b,optimize=True)
            TY_b[:nocc_b,nocc_b:] -= np.einsum('xyhg,yv,xphg->pv',Y1_abab,t1_1_b,t2_1_ab,optimize=True)



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
    

    P = np.square(dx.T)*adc.E*(2/3)
    P = P[0] + P[1] + P[2]


    return P, TY


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
        diag = self.get_diag(imds,eris)
        matvec = self.matvec(imds, eris)
        #matvec = lambda x: self.matvec()
        return matvec, diag

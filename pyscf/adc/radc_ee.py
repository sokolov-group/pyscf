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
# Author: Terrenc Stahl <terrencestahl4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Restricted algebraic diagrammatic construction
'''
import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc
from pyscf.adc import radc_ao2mo, radc_amplitudes
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
from pyscf import symm
import sys

def get_imds(adc, eris=None):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_ccee = t2[0][:]

    t2_ce = t1[0]

    t2_ccee = t2[1][:]

    einsum = lib.einsum
    einsum_type = True

    eris_ovvo = eris.ovvo
    ncore = adc._nocc
    nextern = adc._nvir

    n_singles = ncore * nextern

    e_core = adc.mo_energy[:ncore].copy()
    e_extern = adc.mo_energy[ncore:].copy()


    if eris is None:
        eris = adc.transform_integrals()

    v_ccee = eris.oovv
    v_cece = eris.ovvo
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_ccce = eris.ooov
    v_cecc = eris.ovoo
    v_ceee = eris.ovvv

    occ_list = np.array(range(ncore))
    vir_list = np.array(range(nextern))
    M_ab = np.zeros((ncore*nextern, ncore*nextern))

    ####000#####################
    d_ai_a = adc.mo_energy[ncore:][:,None] - adc.mo_energy[:ncore]
    np.fill_diagonal(M_ab, d_ai_a.transpose().reshape(-1))
    M_ab = M_ab.reshape(ncore,nextern,ncore,nextern).copy()
    
    
    ####010#####################

    M_ab -= einsum('ILAD->IDLA', v_ccee, optimize = einsum_type).copy()
    M_ab += einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()
    
    M_ab += einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()

    ####020#####################

    M_ab += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += einsum('iIDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += einsum('iLAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab -= einsum('LAai,IiDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab += 1/2 * einsum('LAai,iIDa->IDLA',v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab += 1/2 * einsum('iAaL,IiDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab -= 1/2 * einsum('iAaL,iIDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab -= einsum('LiAa,IDai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab += 1/2 * einsum('LiAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('iLAa,IDai->IDLA',t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab -= 1/2 * einsum('iLAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab -= einsum('IiDa,LAai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab += 1/2 * einsum('IiDa,iAaL->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab += 1/2 * einsum('iIDa,LAai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##


    M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Iiab,Labi->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] += einsum('Iiab,Lbai->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Liab,Iabi->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] += einsum('Liab,Ibai->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijAa,iDaj->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] += einsum('ijAa,jDai->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijDa,iAaj->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] += einsum('ijDa,jAai->DA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab[occ_list,:,occ_list,:] += einsum('iAaj,ijDa->DA', v_ceec, t1_ccee, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] -= 1/2 * einsum('iAaj,jiDa->DA', v_ceec, t1_ccee, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] += einsum('ijAa,iDaj->DA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] -= 1/2 * einsum('ijAa,jDai->DA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab[:,vir_list,:,vir_list] += einsum('Iabi,Liab->IL', v_ceec, t1_ccee, optimize = einsum_type)##
    M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('Iabi,Liba->IL', v_ceec, t1_ccee, optimize = einsum_type)##
    M_ab[:,vir_list,:,vir_list] += einsum('Iiab,Labi->IL', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('Iiab,Lbai->IL', t1_ccee, v_ceec, optimize = einsum_type) ##
    
    
    M_ab += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab -= einsum('LiAa,IDai->IDLA',t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('LiAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('iLAa,IDai->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#


    #print("M_ab", np.linalg.norm(M_ab))

    if (adc.method == "adc(3)"):
        v_ceee = radc_ao2mo.unpack_eri_1(eris.ovvv, nextern)
        #v_eeee =  np.zeros_like(eris.vvvvv)
        v_eeee =  eris.vvvvv
        v_ccee = eris.oovv.copy()
   #     v_cccc =  np.zeros_like(v_cccc)
    #    v_cece = np.zeros_like(v_cece.copy())
        #/v_ceee =  np.zeros_like(v_ceee)
        #v_ccce =  np.zeros_like(v_ccce)
        h_ce = np.zeros_like(t2_ce)
    #    v_ccee =np.zeros_like(eris.oovv)
    #    v_ceec =np.zeros_like(eris.ovvo)

        M_030_aa  = einsum('Ia,LADa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
        M_030_aa -= einsum('Ia,LaDA->IDLA', t2_ce, v_ceee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IiDa,LAai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa -= einsum('IiDa,iAaL->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('La,IDAa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
        M_030_aa -= einsum('La,IaAD->IDLA', t2_ce, v_ceee, optimize = einsum_type)
        M_030_aa += 2 * einsum('LiAa,IDai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa -= einsum('LiAa,iDaI->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('iA,ILiD->IDLA', t2_ce, v_ccce, optimize = einsum_type)
        M_030_aa -= einsum('iA,iLID->IDLA', t2_ce, v_ccce, optimize = einsum_type)
        M_030_aa += einsum('iD,LIiA->IDLA', t2_ce, v_ccce, optimize = einsum_type)
        M_030_aa -= einsum('iD,iILA->IDLA', t2_ce, v_ccce, optimize = einsum_type)
        M_030_aa -= einsum('iIDa,LAai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('iIDa,iAaL->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa -= einsum('iLAa,IDai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('iLAa,iDaI->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('A,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('A,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('A,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        
        #M_030_aa -= 1/2 * einsum('A,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        
        M_030_aa -= 1/2 * einsum('A,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('A,iIDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('A,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('A,iLAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('D,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('D,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('D,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('D,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('D,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('D,iIDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('D,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('D,iLAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('I,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('I,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('I,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('I,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('I,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('I,iIDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('I,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('I,iLAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('L,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('L,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('L,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        
        #M_030_aa += 1/2 * einsum('L,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        
        M_030_aa += 1/2 * einsum('L,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('L,iIDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('L,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('L,iLAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('a,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('a,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('a,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        

        ##############################################
        #whole = -einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        ##########################################
        
        M_030_aa -= 1/2*einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        #M_030_aa -= 1/2*einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        ################################################

        M_030_aa -= einsum('a,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('a,iIDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('a,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('a,iLAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('i,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('i,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('i,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        
        M_030_aa += 1/2*einsum('i,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
       # M_030_aa += 1/2*einsum('i,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        

       ################
        original = 1/2 * einsum('L,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        original += 1/2*einsum('i,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        original -= 1/2 * einsum('A,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        original -= 1/2*einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        print("original",np.linalg.norm(original))


       #combine term
        combine_neg = -3/8*einsum('LiAa,iIDa->IDLA', v_ccee, t2_ccee, optimize = einsum_type)
        print("combine_neg", np.linalg.norm(combine_neg))
       
        combine_pos = 3/8*einsum('LiAa,iIDa->IDLA', v_ccee, t2_ccee, optimize = einsum_type)
        print("combine_pos",np.linalg.norm(combine_pos))
        exit()

       ###############



        M_030_aa += einsum('i,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('i,iIDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('i,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('i,iLAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Ia,AD,La->IDLA', h_ce, np.identity(nextern), t2_ce, optimize = einsum_type)
        M_030_aa -= einsum('La,AD,Ia->IDLA', h_ce, np.identity(nextern), t2_ce, optimize = einsum_type)
        M_030_aa -= einsum('iA,IL,iD->IDLA', h_ce, np.identity(ncore), t2_ce, optimize = einsum_type)
        M_030_aa -= einsum('iD,IL,iA->IDLA', h_ce, np.identity(ncore), t2_ce, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,Iiab,Labi->IDLA', np.identity(nextern), t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('AD,Iiab,Lbai->IDLA', np.identity(nextern), t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,Liab,Iabi->IDLA', np.identity(nextern), t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('AD,Liab,Ibai->IDLA', np.identity(nextern), t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,ia,ILia->IDLA', np.identity(nextern), t2_ce, v_ccce, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,ia,LIia->IDLA', np.identity(nextern), t2_ce, v_ccce, optimize = einsum_type)
        M_030_aa += einsum('AD,ia,iILa->IDLA', np.identity(nextern), t2_ce, v_ccce, optimize = einsum_type)
        M_030_aa += einsum('AD,ia,iLIa->IDLA', np.identity(nextern), t2_ce, v_ccce, optimize = einsum_type)
        M_030_aa -= einsum('IL,ia,iADa->IDLA', np.identity(ncore), t2_ce, v_ceee, optimize = einsum_type)
        M_030_aa -= einsum('IL,ia,iDAa->IDLA', np.identity(ncore), t2_ce, v_ceee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ia,iaAD->IDLA', np.identity(ncore), t2_ce, v_ceee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ia,iaDA->IDLA', np.identity(ncore), t2_ce, v_ceee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('IL,ijAa,iDaj->IDLA', np.identity(ncore), t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('IL,ijAa,jDai->IDLA', np.identity(ncore), t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa -= 2 * einsum('IL,ijDa,iAaj->IDLA', np.identity(ncore), t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa += einsum('IL,ijDa,jAai->IDLA', np.identity(ncore), t2_ccee, v_cece, optimize = einsum_type)
        M_030_aa -= 2 * einsum('ADab,Iiac,Libc->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ADab,Iiac,Licb->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ADab,Iica,Libc->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('ADab,Iica,Licb->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AabD,Iibc,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AabD,Iibc,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AabD,Iicb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('AabD,Iicb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('Aabc,IiDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Aabc,IiDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Aabc,iIDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Aabc,iIDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('Dabc,LiAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Dabc,LiAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Dabc,iLAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Dabc,iLAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IDaL,ijab,ijAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('IDaL,ijab,jiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IDai,ijab,LjAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IDai,ijab,jLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IDai,ijba,LjAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('IDai,ijba,jLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ILAa,ijab,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('ILAa,ijab,jiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('ILab,ijDa,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('ILab,ijDa,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('ILij,ikAa,jkDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ILij,ikAa,kjDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ILij,kiAa,jkDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('ILij,kiAa,kjDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IabL,ijDb,ijAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IabL,ijDb,jiAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Iabi,ijDb,LjAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Iabi,ijDb,jLAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('Iabi,jiDb,LjAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('Iabi,jiDb,jLAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IiAD,ijab,Ljab->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('IiAD,ijab,Ljba->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('IiAa,Ljab,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IiAa,Ljab,jiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IiAa,Ljba,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IiAa,Ljba,jiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('Iiab,ijDa,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Iiab,ijDa,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Iiab,jiDa,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Iiab,jiDa,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IijL,jkAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IijL,jkAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IijL,kjAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IijL,kjAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('Iijk,LjAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Iijk,LjAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Iijk,jLAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Iijk,jLAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('LADi,ijab,Ijab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('LADi,ijab,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('LAaI,ijab,ijDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('LAaI,ijab,jiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('LAai,ijab,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('LAai,ijab,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('LAai,ijba,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('LAai,ijba,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('LIDa,ijab,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('LIDa,ijab,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Labi,ijAb,IjDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Labi,ijAb,jIDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('Labi,jiAb,IjDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('Labi,jiAb,jIDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('LiDa,Ijab,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('LiDa,Ijab,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('LiDa,Ijba,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('LiDa,Ijba,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('Liab,ijAa,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Liab,ijAa,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Liab,jiAa,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Liab,jiAa,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('Lijk,IjDa,ikAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Lijk,IjDa,kiAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('Lijk,jIDa,ikAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('Lijk,jIDa,kiAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iADI,ijab,Ljab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('iADI,ijab,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iADj,Ljab,Iiab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('iADj,Ljab,Iiba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('iAaI,ijDb,Ljab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iAaI,ijDb,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iAaI,jiDb,Ljab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('iAaI,jiDb,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('iAaj,Ljab,IiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iAaj,Ljab,iIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('iAaj,Ljba,IiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('iAaj,Ljba,iIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('iDaL,ijAb,Ijab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iDaL,ijAb,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iDaL,jiAb,Ijab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('iDaL,jiAb,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('iDaj,Ijab,LiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iDaj,Ijab,iLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('iDaj,Ijba,LiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('iDaj,Ijba,iLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iIDa,ijab,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('iIDa,ijab,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('iIDa,ijba,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('iIDa,ijba,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('iLAD,ijab,Ijab->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('iLAD,ijab,Ijba->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('iLAa,ijab,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('iLAa,ijab,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('iLAa,ijba,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('iLAa,ijba,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 4 * einsum('iabj,LiAa,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('iabj,LiAa,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('iabj,iLAa,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('iabj,iLAa,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('ijAD,Liab,Ijab->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('ijAD,Liab,Ijba->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('ijAa,Liab,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ijAa,Liab,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ijAa,Liba,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('ijAa,Liba,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('ijDa,Iiab,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ijDa,Iiab,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ijDa,Iiba,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('ijDa,Iiba,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('ijab,LjAa,IiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ijab,LjAa,iIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('ijab,jLAa,IiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('ijab,jLAa,iIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('A,IL,ijAa,ijDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('A,IL,ijAa,jiDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('A,IL,ijDa,ijAa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('A,IL,ijDa,jiAa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('D,IL,ijAa,ijDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('D,IL,ijAa,jiDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('D,IL,ijDa,ijAa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 1/2 * einsum('D,IL,ijDa,jiAa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('I,AD,Iiab,Liab->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('I,AD,Iiab,Liba->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('I,AD,Liab,Iiab->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('I,AD,Liab,Iiba->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('L,AD,Iiab,Liab->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('L,AD,Iiab,Liba->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('L,AD,Liab,Iiab->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 1/2 * einsum('L,AD,Liab,Iiba->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('a,AD,Iiab,Liab->IDLA', e_extern, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('a,AD,Iiab,Liba->IDLA', e_extern, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('a,AD,Iiba,Liab->IDLA', e_extern, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('a,AD,Iiba,Liba->IDLA', e_extern, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('a,AD,Liab,Iiab->IDLA', e_extern, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('a,AD,Liab,Iiba->IDLA', e_extern, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('a,AD,Liba,Iiab->IDLA', e_extern, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('a,AD,Liba,Iiba->IDLA', e_extern, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('a,IL,ijAa,ijDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('a,IL,ijAa,jiDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('a,IL,ijDa,ijAa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += einsum('a,IL,ijDa,jiAa->IDLA', e_extern, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('i,AD,Iiab,Liab->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('i,AD,Iiab,Liba->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('i,AD,Liab,Iiab->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('i,AD,Liab,Iiba->IDLA', e_core, np.identity(nextern), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('i,IL,ijAa,ijDa->IDLA', e_core, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('i,IL,ijAa,jiDa->IDLA', e_core, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('i,IL,ijDa,ijAa->IDLA', e_core, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('i,IL,ijDa,jiAa->IDLA', e_core, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('i,IL,jiAa,ijDa->IDLA', e_core, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('i,IL,jiAa,jiDa->IDLA', e_core, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= einsum('i,IL,jiDa,ijAa->IDLA', e_core, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('i,IL,jiDa,jiAa->IDLA', e_core, np.identity(ncore), t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aa -= 4 * einsum('AD,ILab,ijac,ijbc->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,ILab,ijac,jibc->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 4 * einsum('AD,ILij,ikab,jkab->IDLA', np.identity(nextern), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,ILij,ikab,jkba->IDLA', np.identity(nextern), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,IabL,ijbc,ijac->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,IabL,ijbc,jiac->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 4 * einsum('AD,Iabi,ijbc,Ljac->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,Iabi,ijbc,Ljca->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,Iabi,jibc,Ljac->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,Iabi,jibc,Ljca->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,Iiab,ijac,Ljbc->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,Iiab,ijac,Ljcb->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,Iiab,jiac,Ljbc->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,Iiab,jiac,Ljcb->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,IijL,jkab,ikab->IDLA', np.identity(nextern), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('AD,IijL,jkab,ikba->IDLA', np.identity(nextern), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,Iijk,Ljab,ikab->IDLA', np.identity(nextern), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('AD,Iijk,Ljab,ikba->IDLA', np.identity(nextern), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 4 * einsum('AD,Labi,ijbc,Ijac->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,Labi,ijbc,Ijca->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,Labi,jibc,Ijac->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,Labi,jibc,Ijca->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,Liab,ijac,Ijbc->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,Liab,ijac,Ijcb->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,Liab,jiac,Ijbc->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,Liab,jiac,Ijcb->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,Lijk,Ijab,ikab->IDLA', np.identity(nextern), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('AD,Lijk,Ijab,ikba->IDLA', np.identity(nextern), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('AD,abcd,Iiac,Libd->IDLA', np.identity(nextern), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('AD,abcd,Iiac,Lidb->IDLA', np.identity(nextern), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,iabj,Iiac,Ljbc->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,iabj,Iiac,Ljcb->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,iabj,Iica,Ljbc->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 4 * einsum('AD,iabj,Iica,Ljcb->IDLA', np.identity(nextern), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,ijab,Iibc,Ljac->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,ijab,Iibc,Ljca->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('AD,ijab,Iicb,Ljac->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('AD,ijab,Iicb,Ljca->IDLA', np.identity(nextern), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 4 * einsum('IL,ADab,ijac,ijbc->IDLA', np.identity(ncore), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('IL,ADab,ijac,jibc->IDLA', np.identity(ncore), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('IL,AabD,ijbc,ijac->IDLA', np.identity(ncore), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IL,AabD,ijbc,jiac->IDLA', np.identity(ncore), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('IL,Aabc,ijDb,ijac->IDLA', np.identity(ncore), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IL,Aabc,ijDb,jiac->IDLA', np.identity(ncore), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('IL,Dabc,ijAb,ijac->IDLA', np.identity(ncore), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IL,Dabc,ijAb,jiac->IDLA', np.identity(ncore), v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,iADj,jkab,ikab->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,iADj,jkab,ikba->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 4 * einsum('IL,iAaj,jkab,ikDb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,iAaj,jkab,kiDb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,iAaj,jkba,ikDb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,iAaj,jkba,kiDb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 4 * einsum('IL,iDaj,jkab,ikAb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,iDaj,jkab,kiAb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,iDaj,jkba,ikAb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,iDaj,jkba,kiAb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,iabj,ikAa,jkDb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,iabj,ikAa,kjDb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,iabj,kiAa,jkDb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 4 * einsum('IL,iabj,kiAa,kjDb->IDLA', np.identity(ncore), v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 4 * einsum('IL,ijAD,ikab,jkab->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ijAD,ikab,jkba->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ijAa,ikab,jkDb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,ijAa,ikab,kjDb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,ijAa,ikba,jkDb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ijAa,ikba,kjDb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ijDa,ikab,jkAb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,ijDa,ikab,kjAb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,ijDa,ikba,jkAb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ijDa,ikba,kjAb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ijab,jkAa,ikDb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,ijab,jkAa,kiDb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= einsum('IL,ijab,kjAa,ikDb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += 2 * einsum('IL,ijab,kjAa,kiDb->IDLA', np.identity(ncore), v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa -= 2 * einsum('IL,ijkl,ikAa,jlDa->IDLA', np.identity(ncore), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aa += einsum('IL,ijkl,ikAa,ljDa->IDLA', np.identity(ncore), v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)

        M_030_aabb  = einsum('Ia,LADa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('IiDa,LAai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aabb -= einsum('IiDa,iAaL->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aabb += einsum('La,IDAa->IDLA', t2_ce, v_ceee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('LiAa,IDai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aabb -= einsum('LiAa,iDaI->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aabb -= einsum('iA,iLID->IDLA', t2_ce, v_ccce, optimize = einsum_type)
        M_030_aabb -= einsum('iD,iILA->IDLA', t2_ce, v_ccce, optimize = einsum_type)
        M_030_aabb -= einsum('iIDa,LAai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aabb -= einsum('iLAa,IDai->IDLA', t2_ccee, v_cece, optimize = einsum_type)
        M_030_aabb += einsum('A,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 1/2 * einsum('A,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += einsum('A,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 1/2 * einsum('A,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 1/2 * einsum('A,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 1/2 * einsum('A,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += einsum('D,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 1/2 * einsum('D,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += einsum('D,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 1/2 * einsum('D,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 1/2 * einsum('D,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 1/2 * einsum('D,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('I,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('I,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('I,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('I,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('I,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('I,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('L,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('L,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('L,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('L,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('L,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('L,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('a,IiDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('a,IiDa,iLAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('a,LiAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('a,iIDa,LiAa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('a,iLAa,IiDa->IDLA', e_extern, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('i,IiDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += einsum('i,IiDa,iLAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('i,LiAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += einsum('i,LiAa,iIDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += einsum('i,iIDa,LiAa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += einsum('i,iLAa,IiDa->IDLA', e_core, t1_ccee, t2_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('AabD,Iibc,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('AabD,Iibc,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('AabD,Iicb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('Aabc,IiDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('Aabc,IiDb,Lica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('Aabc,iIDb,Liac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('Dabc,LiAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('Dabc,LiAb,Iica->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('Dabc,iLAb,Iiac->IDLA', v_eeee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('IDaL,ijab,ijAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('IDaL,ijab,jiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('IDai,ijab,LjAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('IDai,ijab,jLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('IDai,ijba,LjAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('IDai,ijba,jLAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('IabL,ijDb,jiAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('Iabi,ijDb,LjAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('Iabi,jiDb,LjAa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('IiAa,Ljab,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('IiAa,Ljab,jiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('IiAa,Ljba,ijDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('Iiab,ijDa,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('Iiab,ijDa,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('Iiab,jiDa,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('IijL,jkAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('IijL,jkAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('IijL,kjAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('Iijk,LjAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('Iijk,LjAa,kiDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('Iijk,jLAa,ikDa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('LADi,ijab,Ijab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('LADi,ijab,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('LAaI,ijab,ijDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('LAaI,ijab,jiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('LAai,ijab,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('LAai,ijab,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('LAai,ijba,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('LAai,ijba,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('Labi,ijAb,IjDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('Labi,jiAb,IjDa->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('LiDa,Ijab,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('LiDa,Ijab,jiAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('LiDa,Ijba,ijAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('Liab,ijAa,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('Liab,ijAa,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('Liab,jiAa,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 2 * einsum('Lijk,IjDa,ikAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('Lijk,IjDa,kiAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('Lijk,jIDa,ikAa->IDLA', v_cccc, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('iADI,ijab,Ljab->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('iADI,ijab,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('iADj,Ljab,Iiba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('iAaI,jiDb,Ljba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('iAaj,Ljab,IiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('iAaj,Ljba,IiDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('iDaL,jiAb,Ijba->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('iDaj,Ijab,LiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('iDaj,Ijba,LiAb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('iIDa,ijab,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('iIDa,ijab,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('iIDa,ijba,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= einsum('iLAa,ijab,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('iLAa,ijab,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 1/2 * einsum('iLAa,ijba,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += 4 * einsum('iabj,LiAa,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('iabj,LiAa,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('iabj,iLAa,IjDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('iabj,iLAa,jIDb->IDLA', v_ceec, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('ijAa,Liab,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('ijAa,Liab,jIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('ijAa,Liba,IjDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('ijDa,Iiab,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('ijDa,Iiab,jLAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('ijDa,Iiba,LjAb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb -= 2 * einsum('ijab,LjAa,IiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('ijab,LjAa,iIDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)
        M_030_aabb += einsum('ijab,jLAa,IiDb->IDLA', v_ccee, t1_ccee, t1_ccee, optimize = einsum_type)


    M_ab_ = M_ab + M_030_aa + M_030_aabb
    M_ab = M_ab_.reshape(n_singles, n_singles)

  #  print(np.linalg.norm(M_ab - M_ab.T))
  #  exit()

    #e, h = np.linalg.eigh(M_ab)
    #print(e*27.2114)

    return M_ab


def get_diag(adc,M_ab=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_ = adc.get_imds()

    M_ = M_ab

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir

    dim = n_singles + n_doubles
    diag = np.zeros(dim)

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None]+e_occ
    d_ab = e_vir[:,None]+e_vir

    D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1)).reshape((nocc,nocc,nvir,nvir))
    diag[s2:f2] = D_ijab.reshape(-1)

    diag[s1:f1] = np.diagonal(M_)

    return diag


def matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    
    if M_ab is None:
        M_  = adc.get_imds()

    M_ = M_ab
    
    if eris is None:
        eris = adc.transform_integrals()

    einsum = lib.einsum
    einsum_type = True

    v_ccee = eris.oovv
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_cecc = eris.ovoo
    v_ceee = eris.ovvv

    nocc = adc._nocc
    nvir = adc._nvir

    ij_ind_a = np.tril_indices(nocc, k=-1)
    ab_ind_a = np.tril_indices(nvir, k=-1)

    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir

    dim = n_singles + n_doubles
    diag = np.zeros(dim)

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None]+e_occ
    d_ab = e_vir[:,None]+e_vir

    d_ia = e_occ[:,None]+e_vir
    e_core = adc.mo_energy[:nocc].copy()
    e_extern = adc.mo_energy[nocc:].copy()


    #Calculate sigma vector
    #@profile
    #@profile
    def sigma_(r):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)
       
        r1 = r[s1:f1]

        Y = r1.reshape(nocc, nvir).copy()
        
        r2 = r[s2:f2].reshape(nocc,nocc,nvir,nvir).copy()

        s = np.zeros(dim)

        s[s1:f1] = lib.einsum('ab,b->a',M_,r1, optimize = True)

        D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1)).reshape((nocc,nocc,nvir,nvir))
        s[s2:f2] = (D_ijab.reshape(-1))*r[s2:f2]
        del D_ijab

        if isinstance(eris.ovvv, type(None)):
            M_11Y0 = np.zeros((nocc,nocc,nvir,nvir))
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc,chnk_size):
                v_ceee = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                k = v_ceee.shape[0]
                M_11Y0[:,a:a+k,:,:] += einsum('Ia,JDaC->IJCD', Y, v_ceee, optimize = einsum_type)
                M_11Y0[a:a+k,:,:,:] += einsum('Ja,ICaD->IJCD', Y, v_ceee, optimize = einsum_type)

                s[s1:f1] += -einsum('Iiab,iabD->ID', r2[:,a:a+k,:,:], v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += 2*einsum('Iiab,ibDa->ID', r2[:,a:a+k,:,:], v_ceee, optimize = einsum_type).reshape(-1)
                del v_ceee
                a += k
            s[s2:f2] += M_11Y0.reshape(-1)
            del M_11Y0
        else:
            v_ceee = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            M_11Y0 = einsum('Ia,JDaC->IJCD', Y, v_ceee, optimize = einsum_type)
            M_11Y0 += einsum('Ja,ICaD->IJCD', Y, v_ceee, optimize = einsum_type)
            s[s2:f2] += M_11Y0.reshape(-1)


            M_01Y1 = -einsum('Iiab,iabD->ID', r2, v_ceee, optimize = einsum_type)
            M_01Y1 += 2*einsum('Iiab,ibDa->ID', r2, v_ceee, optimize = einsum_type)
            s[s1:f1] += M_01Y1.reshape(-1)
            del M_11Y0
            del M_01Y1

        M_11Y0 = -einsum('iC,JDIi->IJCD', Y, v_cecc, optimize = einsum_type)
        M_11Y0 -= einsum('iD,ICJi->IJCD', Y, v_cecc, optimize = einsum_type)
        s[s2:f2] += M_11Y0.reshape(-1)

        M_01Y1 = -2*einsum('ijDa,jaiI->ID', r2, v_cecc, optimize = einsum_type)
        M_01Y1 += einsum('ijDa,iajI->ID', r2, v_cecc, optimize = einsum_type)
        s[s1:f1] += M_01Y1.reshape(-1)

        if (adc.method == "adc(2)-x") or (adc.method == "adc(3)"):
            del Y
            Y = r2.copy()

            if isinstance(eris.ovvv, type(None)):
                s[s2:f2] += radc_amplitudes.contract_ladder(adc,Y,eris.Lvv).reshape(-1)
            else:
                v_eeee = eris.vvvv.reshape(nvir, nvir, nvir,nvir)
                M_1Y1_aa  = einsum('IJab,CDab->IJCD', Y, v_eeee, optimize = einsum_type)
                s[s2:f2] += M_1Y1_aa.reshape(-1)
                del M_1Y1_aa
            
                
            M_1Y1_aa = 2 * einsum('IiCa,JDai->IJCD', Y, v_ceec, optimize = einsum_type)
            M_1Y1_aa -= einsum('IiCa,iJDa->IJCD', Y, v_ccee, optimize = einsum_type)
            M_1Y1_aa -= einsum('IiaC,JDai->IJCD', Y, v_ceec, optimize = einsum_type)
            M_1Y1_aa -= einsum('IiaD,iJCa->IJCD', Y, v_ccee, optimize = einsum_type)
            M_1Y1_aa += 2 * einsum('JiDa,ICai->IJCD', Y, v_ceec, optimize = einsum_type)
            M_1Y1_aa -= einsum('JiDa,iICa->IJCD', Y, v_ccee, optimize = einsum_type)
            M_1Y1_aa -= einsum('JiaC,iIDa->IJCD', Y, v_ccee, optimize = einsum_type)
            M_1Y1_aa -= einsum('JiaD,ICai->IJCD', Y, v_ceec, optimize = einsum_type)
            M_1Y1_aa += einsum('ijCD,IiJj->IJCD', Y, v_cccc, optimize = einsum_type)






            s[s2:f2] += M_1Y1_aa.reshape(-1)


        if (adc.method == "adc(3)"):
            t1 = adc.t1
            t2 = adc.t2
            t1_ccee = t2[0][:]
            t2_ce = t1[0]
            t2_ccee = t2[1][:]
            v_eece = v_ceee.copy()

            v_ccce = v_cecc.copy()

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            M_02Y1_aa  = 2 * einsum('IiDa,a,ia->ID', Y, e_extern, t2_ce, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('IiDa,i,ia->ID', Y, e_core, t2_ce, optimize = einsum_type)
            M_02Y1_aa -= einsum('IiaD,a,ia->ID', Y, e_extern, t2_ce, optimize = einsum_type)
            M_02Y1_aa += einsum('IiaD,i,ia->ID', Y, e_core, t2_ce, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('IiDa,ijbc,jbac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += 4 * einsum('IiDa,ijbc,jcab->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= 4 * einsum('IiDa,jkab,kbji->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += 2 * einsum('IiDa,jkab,jbki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += einsum('IiaD,ijbc,jbac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('IiaD,ijbc,jcab->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += 2 * einsum('IiaD,jkab,kbji->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa -= einsum('IiaD,jkab,jbki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += einsum('Iiab,ijac,jDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('Iiab,ijac,jcbD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('Iiab,ijbc,jDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += 4 * einsum('Iiab,ijbc,jcaD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('Iiab,ijca,jDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += einsum('Iiab,ijca,jcbD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += einsum('Iiab,ijcb,jDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('Iiab,ijcb,jcaD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= einsum('Iiab,jkab,kDji->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += 2 * einsum('Iiab,jkab,jDki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('ijDa,ijbc,Ibac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += einsum('ijDa,ijbc,Icab->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += 2 * einsum('ijDa,ikab,kbIj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa -= einsum('ijDa,ikab,Ibkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa -= einsum('ijDa,ikba,kbIj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += 2 * einsum('ijDa,ikba,Ibkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa -= 4 * einsum('ijDa,jkab,kbIi->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += 2 * einsum('ijDa,jkab,Ibki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += 2 * einsum('ijDa,jkba,kbIi->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa -= einsum('ijDa,jkba,Ibki->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += 3 * einsum('ijab,ijac,IDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= 2 * einsum('ijab,ijac,IcbD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= einsum('ijab,ijbc,IDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += einsum('ijab,ijbc,IcaD->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa += 2 * einsum('ijab,ikab,kDIj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa -= 3 * einsum('ijab,ikab,IDkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa -= einsum('ijab,ikba,kDIj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += einsum('ijab,ikba,IDkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            
            M_02Y1_aa += einsum('ijab,ijac,IDbc->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= einsum('ijab,ijbc,IDac->ID', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_02Y1_aa -= einsum('ijab,ikab,IDkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_02Y1_aa += einsum('ijab,ikba,IDkj->ID', Y, t1_ccee, v_ccce, optimize = einsum_type)
            
            
            Y = r1.reshape(nocc, nvir).copy()













            M_12Y0_ab  = -einsum('Ia,JiDb,iaCb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab += 2 * einsum('Ia,JiDb,ibCa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab -= einsum('Ia,iJCb,iaDb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab -= einsum('Ia,iJDb,ibCa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab += einsum('Ia,ijCD,iajJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab -= einsum('Ja,IiCb,iaDb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab += 2 * einsum('Ja,IiCb,ibDa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab -= einsum('Ja,iICb,ibDa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab -= einsum('Ja,iIDb,iaCb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab += einsum('Ja,ijCD,jaiI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab -= einsum('iC,IJab,iaDb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab -= 2 * einsum('iC,JjDa,jaiI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab += einsum('iC,JjDa,iajI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab += einsum('iC,jIDa,iajJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab += einsum('iC,jJDa,jaiI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab -= einsum('iD,IJab,ibCa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab -= 2 * einsum('iD,IjCa,jaiJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab += einsum('iD,IjCa,iajJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab += einsum('iD,jICa,jaiJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab += einsum('iD,jJCa,iajI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab += 2 * einsum('ia,IJCb,iaDb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab -= einsum('ia,IJCb,ibDa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab += einsum('ia,IjCD,jaiJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab -= 2 * einsum('ia,IjCD,iajJ->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab += 2 * einsum('ia,JIDb,iaCb->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab -= einsum('ia,JIDb,ibCa->IJCD', Y, t1_ccee, v_ceee, optimize = einsum_type)
            M_12Y0_ab += einsum('ia,jJCD,jaiI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            M_12Y0_ab -= 2 * einsum('ia,jJCD,iajI->IJCD', Y, t1_ccee, v_ccce, optimize = einsum_type)
            
            s[s1:f1] += M_02Y1_aa.reshape(-1)
            s[s2:f2] += M_12Y0_ab.reshape(-1)
        return s



    return sigma_





def renormalize_eigenvectors(adc, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc * nvir

    U = adc.U
    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nocc,nocc,nvir,nvir)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - 4.*np.dot(U2.ravel(), U2.transpose(0,1,3,2).ravel())
        U[:,I] /= np.sqrt(UdotU)

    return U




def get_X(adc):
    U = renormalize_eigenvectors(adc)

    #U = adc.U
    U = U.T.copy()

    nroots = U.shape[0]

    dip_ints = -adc.mol.intor('int1e_r',comp=3)
    dm = np.zeros_like((dip_ints))
    for i in range(dip_ints.shape[0]):
        dip = dip_ints[i,:,:]
        dm[i,:,:] = np.dot(adc.mo_coeff.T,np.dot(dip,adc.mo_coeff))

   

    x = np.array([])
    
    nocc = adc._nocc
    nvir = adc._nvir

    nmo = nocc + nvir


    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir


    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    t1 = adc.t1
    t2 = adc.t2

    t1_ccee = t2[0][:]
    t2_ccee = t2[1][:]

    t2_ce = t1[0]


    einsum = lib.einsum
    einsum_type = True

    TY = np.zeros((nmo, nmo))

    TY_ = []
    x = np.array([])

    for r in range(U.shape[0]):


        r1 = U[r][s1:f1]

        Y = r1.reshape(nocc, nvir).copy()
        #print(np.linalg.norm(Y))
        #exit()
        
       # print(np.linalg.norm(Y))
       # exit()
        r2 = U[r][s2:f2].reshape(nocc,nocc,nvir,nvir).copy()

        TY[:nocc,nocc:] = Y.copy()

        TY[nocc:,:nocc]  = einsum('ia,LiAa->AL', Y, t1_ccee, optimize = einsum_type)
        TY[nocc:,:nocc] -= einsum('ia,iLAa->AL', Y, t1_ccee, optimize = einsum_type)

        TY[:nocc,:nocc] =- einsum('Iiab,Liab->IL', r2, t1_ccee, optimize = einsum_type)
        TY[:nocc,:nocc] += einsum('Iiab,Liba->IL', r2, t1_ccee, optimize = einsum_type)


        TY[nocc:,nocc:]  = 2 * einsum('ijCa,ijAa->AC', r2, t1_ccee, optimize = einsum_type)
        TY[nocc:,nocc:] -= 2 * einsum('ijCa,jiAa->AC', r2, t1_ccee, optimize = einsum_type)


        TY[:nocc,:nocc] -= einsum('Iiab,Liab->IL', r2, t1_ccee, optimize = einsum_type)
        TY[nocc:,nocc:] += einsum('ijCa,ijAa->AC', r2, t1_ccee, optimize = einsum_type)


        TY[:nocc,:nocc] -= einsum('Ia,La->IL', Y, t2_ce, optimize = einsum_type)
        TY[nocc:,nocc:] += einsum('iC,iA->AC', Y, t2_ce, optimize = einsum_type)

        TY[:nocc,nocc:] -= einsum('Ia,ijab,ijCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('Ia,ijab,jiCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= einsum('iC,ijab,Ijab->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('iC,ijab,Ijba->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += einsum('ia,ijab,IjCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijab,jICb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijba,IjCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('ia,ijba,jICb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)

        TY[nocc:,:nocc] += einsum('ia,LiAa->AL', Y, t2_ccee, optimize = einsum_type)
        TY[nocc:,:nocc] -= einsum('ia,iLAa->AL', Y, t2_ccee, optimize = einsum_type)


        dx = lib.einsum("rqp,qp->r", dm, TY, optimize = True)

        TY_ = np.append(TY_,TY)

        x = np.append(x,dx)
    x = x.reshape(nroots, 3)

    return TY_, x

#def analyze_spec_factor(adc):




def get_properties(adc,nroots):

    TY, dx  = adc.get_X()


    X = TY.reshape(nroots,-1)

    P = np.square(dx.T)*adc.E*(2/3)
    P = P[0] + P[1] + P[2]

    return P, X


def analyze(myadc):

    if myadc.compute_properties:

        header = ("\n*************************************************************"
                  "\n            Spectroscopic factors analysis summary"
                  "\n*************************************************************")
        logger.info(myadc, header)

        myadc.analyze_spec_factor()

def compute_dyson_mo(myadc):

    X = myadc.X

    if X is None:
        nroots = myadc.U.shape[1]
        P,X = myadc.get_properties(nroots)

    nroots = X.shape[1]
    dyson_mo = np.dot(myadc.mo_coeff,X)

    return dyson_mo


class RADCEE(radc.RADC):
    '''restricted ADC for EA energies and spectroscopic amplitudes

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
            Space size to hold trial vectors for Davidson iterative
            diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcea = adc.RADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ip : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''
    def __init__(self, adc):
        self.mol = adc.mol
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.imds = adc.imds
        self.e_corr = adc.e_corr
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments
        self.E = None
        self.U = None
        self.P = None
        self.X = None
        self.evec_print_tol = adc.evec_print_tol
        self.spec_factor_print_tol = adc.spec_factor_print_tol

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
                    'mo_energy', 'max_memory', 't1', 'max_space', 't2',
                    'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = radc.kernel
    get_imds = get_imds
    matvec = matvec
    get_diag = get_diag
    #get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors
    get_X = get_X
    get_properties = get_properties
#    analyze_spec_factor = analyze_spec_factor
 #   analyze_eigenvector = analyze_eigenvector
 #   analyze = analyze
 #   compute_dyson_mo = compute_dyson_mo

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


    def gen_matvec(self, imds=None, eris=None):
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(imds, eris)
        matvec = self.matvec(imds, eris)
        return matvec, diag


def contract_r_vvvv(myadc,r2,vvvv):

    nocc = myadc._nocc
    nvir = myadc._nvir

    r2_vvvv = np.zeros((nocc,nocc,nvir,nvir))
    r2 = np.ascontiguousarray(r2.reshape(nocc*nocc,-1))
    chnk_size = radc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv, list):
        for dataset in vvvv:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc,-1,nvir)
            del dataset
            a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv_p.T).reshape(nocc*nocc,-1)
            del vvvv_p
            a += k
    else:
        raise Exception("Unknown vvvv type")

    r2_vvvv = r2_vvvv.reshape(-1)

    return r2_vvvv

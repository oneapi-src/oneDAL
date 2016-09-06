/* file: vmlvsl.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  VML/VSL function declarations
//--
*/


#ifndef __VMLVSL_H__
#define __VMLVSL_H__


#if defined(__cplusplus)
extern "C" {
#endif


typedef void * DAAL_VSLSSTaskPtr;


#if defined(_WIN64) || defined(__x86_64__)

void fpk_vml_sLn_EXHAynn(const int , const float* , float* );
void fpk_vml_sLn_EXLAynn(const int , const float* , float* );
void fpk_vml_sLn_EXEPnnn(const int , const float* , float* );
void fpk_vml_dLn_EXHAynn(const int , const double* , double* );
void fpk_vml_dLn_EXLAynn(const int , const double* , double* );
void fpk_vml_dLn_EXEPnnn(const int , const double* , double* );

void fpk_vml_sLn_U8HAynn(const int , const float* , float* );
void fpk_vml_sLn_U8LAynn(const int , const float* , float* );
void fpk_vml_sLn_U8EPnnn(const int , const float* , float* );
void fpk_vml_dLn_U8HAynn(const int , const double* , double* );
void fpk_vml_dLn_U8LAynn(const int , const double* , double* );
void fpk_vml_dLn_U8EPnnn(const int , const double* , double* );

void fpk_vml_sLn_H8HAynn(const int , const float* , float* );
void fpk_vml_sLn_H8LAynn(const int , const float* , float* );
void fpk_vml_sLn_H8EPnnn(const int , const float* , float* );
void fpk_vml_dLn_H8HAynn(const int , const double* , double* );
void fpk_vml_dLn_H8LAynn(const int , const double* , double* );
void fpk_vml_dLn_H8EPnnn(const int , const double* , double* );

void fpk_vml_sLn_E9HAynn(const int , const float* , float* );
void fpk_vml_sLn_E9LAynn(const int , const float* , float* );
void fpk_vml_sLn_E9EPnnn(const int , const float* , float* );
void fpk_vml_dLn_E9HAynn(const int , const double* , double* );
void fpk_vml_dLn_E9LAynn(const int , const double* , double* );
void fpk_vml_dLn_E9EPnnn(const int , const double* , double* );

void fpk_vml_sLn_L9HAynn(const int , const float* , float* );
void fpk_vml_sLn_L9LAynn(const int , const float* , float* );
void fpk_vml_sLn_L9EPnnn(const int , const float* , float* );
void fpk_vml_dLn_L9HAynn(const int , const double* , double* );
void fpk_vml_dLn_L9LAynn(const int , const double* , double* );
void fpk_vml_dLn_L9EPnnn(const int , const double* , double* );

void fpk_vml_sLn_B3HAynn(const int , const float* , float* );
void fpk_vml_sLn_B3LAynn(const int , const float* , float* );
void fpk_vml_sLn_B3EPnnn(const int , const float* , float* );
void fpk_vml_dLn_B3HAynn(const int , const double* , double* );
void fpk_vml_dLn_B3LAynn(const int , const double* , double* );
void fpk_vml_dLn_B3EPnnn(const int , const double* , double* );

void fpk_vml_sLn_Z0HAynn(const int , const float* , float* );
void fpk_vml_sLn_Z0LAynn(const int , const float* , float* );
void fpk_vml_sLn_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dLn_Z0HAynn(const int , const double* , double* );
void fpk_vml_dLn_Z0LAynn(const int , const double* , double* );
void fpk_vml_dLn_Z0EPnnn(const int , const double* , double* );


void fpk_vml_sExp_EXHAynn(const int , const float* , float* );
void fpk_vml_sExp_EXLAynn(const int , const float* , float* );
void fpk_vml_sExp_EXEPnnn(const int , const float* , float* );
void fpk_vml_dExp_EXHAynn(const int , const double* , double* );
void fpk_vml_dExp_EXLAynn(const int , const double* , double* );
void fpk_vml_dExp_EXEPnnn(const int , const double* , double* );

void fpk_vml_sExp_U8HAynn(const int , const float* , float* );
void fpk_vml_sExp_U8LAynn(const int , const float* , float* );
void fpk_vml_sExp_U8EPnnn(const int , const float* , float* );
void fpk_vml_dExp_U8HAynn(const int , const double* , double* );
void fpk_vml_dExp_U8LAynn(const int , const double* , double* );
void fpk_vml_dExp_U8EPnnn(const int , const double* , double* );

void fpk_vml_sExp_H8HAynn(const int , const float* , float* );
void fpk_vml_sExp_H8LAynn(const int , const float* , float* );
void fpk_vml_sExp_H8EPnnn(const int , const float* , float* );
void fpk_vml_dExp_H8HAynn(const int , const double* , double* );
void fpk_vml_dExp_H8LAynn(const int , const double* , double* );
void fpk_vml_dExp_H8EPnnn(const int , const double* , double* );

void fpk_vml_sExp_E9HAynn(const int , const float* , float* );
void fpk_vml_sExp_E9LAynn(const int , const float* , float* );
void fpk_vml_sExp_E9EPnnn(const int , const float* , float* );
void fpk_vml_dExp_E9HAynn(const int , const double* , double* );
void fpk_vml_dExp_E9LAynn(const int , const double* , double* );
void fpk_vml_dExp_E9EPnnn(const int , const double* , double* );

void fpk_vml_sExp_L9HAynn(const int , const float* , float* );
void fpk_vml_sExp_L9LAynn(const int , const float* , float* );
void fpk_vml_sExp_L9EPnnn(const int , const float* , float* );
void fpk_vml_dExp_L9HAynn(const int , const double* , double* );
void fpk_vml_dExp_L9LAynn(const int , const double* , double* );
void fpk_vml_dExp_L9EPnnn(const int , const double* , double* );

void fpk_vml_sExp_B3HAynn(const int , const float* , float* );
void fpk_vml_sExp_B3LAynn(const int , const float* , float* );
void fpk_vml_sExp_B3EPnnn(const int , const float* , float* );
void fpk_vml_dExp_B3HAynn(const int , const double* , double* );
void fpk_vml_dExp_B3LAynn(const int , const double* , double* );
void fpk_vml_dExp_B3EPnnn(const int , const double* , double* );

void fpk_vml_sExp_Z0HAynn(const int , const float* , float* );
void fpk_vml_sExp_Z0LAynn(const int , const float* , float* );
void fpk_vml_sExp_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dExp_Z0HAynn(const int , const double* , double* );
void fpk_vml_dExp_Z0LAynn(const int , const double* , double* );
void fpk_vml_dExp_Z0EPnnn(const int , const double* , double* );


void fpk_vml_sErf_EXHAynn(const int , const float* , float* );
void fpk_vml_sErf_EXLAynn(const int , const float* , float* );
void fpk_vml_sErf_EXEPnnn(const int , const float* , float* );
void fpk_vml_dErf_EXHAynn(const int , const double* , double* );
void fpk_vml_dErf_EXLAynn(const int , const double* , double* );
void fpk_vml_dErf_EXEPnnn(const int , const double* , double* );

void fpk_vml_sErf_U8HAynn(const int , const float* , float* );
void fpk_vml_sErf_U8LAynn(const int , const float* , float* );
void fpk_vml_sErf_U8EPnnn(const int , const float* , float* );
void fpk_vml_dErf_U8HAynn(const int , const double* , double* );
void fpk_vml_dErf_U8LAynn(const int , const double* , double* );
void fpk_vml_dErf_U8EPnnn(const int , const double* , double* );

void fpk_vml_sErf_H8HAynn(const int , const float* , float* );
void fpk_vml_sErf_H8LAynn(const int , const float* , float* );
void fpk_vml_sErf_H8EPnnn(const int , const float* , float* );
void fpk_vml_dErf_H8HAynn(const int , const double* , double* );
void fpk_vml_dErf_H8LAynn(const int , const double* , double* );
void fpk_vml_dErf_H8EPnnn(const int , const double* , double* );

void fpk_vml_sErf_E9HAynn(const int , const float* , float* );
void fpk_vml_sErf_E9LAynn(const int , const float* , float* );
void fpk_vml_sErf_E9EPnnn(const int , const float* , float* );
void fpk_vml_dErf_E9HAynn(const int , const double* , double* );
void fpk_vml_dErf_E9LAynn(const int , const double* , double* );
void fpk_vml_dErf_E9EPnnn(const int , const double* , double* );

void fpk_vml_sErf_L9HAynn(const int , const float* , float* );
void fpk_vml_sErf_L9LAynn(const int , const float* , float* );
void fpk_vml_sErf_L9EPnnn(const int , const float* , float* );
void fpk_vml_dErf_L9HAynn(const int , const double* , double* );
void fpk_vml_dErf_L9LAynn(const int , const double* , double* );
void fpk_vml_dErf_L9EPnnn(const int , const double* , double* );

void fpk_vml_sErf_B3HAynn(const int , const float* , float* );
void fpk_vml_sErf_B3LAynn(const int , const float* , float* );
void fpk_vml_sErf_B3EPnnn(const int , const float* , float* );
void fpk_vml_dErf_B3HAynn(const int , const double* , double* );
void fpk_vml_dErf_B3LAynn(const int , const double* , double* );
void fpk_vml_dErf_B3EPnnn(const int , const double* , double* );

void fpk_vml_sErf_Z0HAynn(const int , const float* , float* );
void fpk_vml_sErf_Z0LAynn(const int , const float* , float* );
void fpk_vml_sErf_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dErf_Z0HAynn(const int , const double* , double* );
void fpk_vml_dErf_Z0LAynn(const int , const double* , double* );
void fpk_vml_dErf_Z0EPnnn(const int , const double* , double* );


void fpk_vml_sErfInv_EXHAynn(const int , const float* , float* );
void fpk_vml_sErfInv_EXLAynn(const int , const float* , float* );
void fpk_vml_sErfInv_EXEPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_EXHAynn(const int , const double* , double* );
void fpk_vml_dErfInv_EXLAynn(const int , const double* , double* );
void fpk_vml_dErfInv_EXEPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_U8HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_U8LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_U8EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_U8HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_U8LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_U8EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_H8HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_H8LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_H8EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_H8HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_H8LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_H8EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_E9HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_E9LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_E9EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_E9HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_E9LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_E9EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_L9HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_L9LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_L9EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_L9HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_L9LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_L9EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_B3HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_B3LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_B3EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_B3HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_B3LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_B3EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_Z0HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_Z0LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_Z0HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_Z0LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_Z0EPnnn(const int , const double* , double* );


void fpk_vml_sCeil_EXHAynn(const int , const float* , float* );
void fpk_vml_sCeil_EXLAynn(const int , const float* , float* );
void fpk_vml_sCeil_EXEPnnn(const int , const float* , float* );
void fpk_vml_dCeil_EXHAynn(const int , const double* , double* );
void fpk_vml_dCeil_EXLAynn(const int , const double* , double* );
void fpk_vml_dCeil_EXEPnnn(const int , const double* , double* );

void fpk_vml_sCeil_U8HAynn(const int , const float* , float* );
void fpk_vml_sCeil_U8LAynn(const int , const float* , float* );
void fpk_vml_sCeil_U8EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_U8HAynn(const int , const double* , double* );
void fpk_vml_dCeil_U8LAynn(const int , const double* , double* );
void fpk_vml_dCeil_U8EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_H8HAynn(const int , const float* , float* );
void fpk_vml_sCeil_H8LAynn(const int , const float* , float* );
void fpk_vml_sCeil_H8EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_H8HAynn(const int , const double* , double* );
void fpk_vml_dCeil_H8LAynn(const int , const double* , double* );
void fpk_vml_dCeil_H8EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_E9HAynn(const int , const float* , float* );
void fpk_vml_sCeil_E9LAynn(const int , const float* , float* );
void fpk_vml_sCeil_E9EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_E9HAynn(const int , const double* , double* );
void fpk_vml_dCeil_E9LAynn(const int , const double* , double* );
void fpk_vml_dCeil_E9EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_L9HAynn(const int , const float* , float* );
void fpk_vml_sCeil_L9LAynn(const int , const float* , float* );
void fpk_vml_sCeil_L9EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_L9HAynn(const int , const double* , double* );
void fpk_vml_dCeil_L9LAynn(const int , const double* , double* );
void fpk_vml_dCeil_L9EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_B3HAynn(const int , const float* , float* );
void fpk_vml_sCeil_B3LAynn(const int , const float* , float* );
void fpk_vml_sCeil_B3EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_B3HAynn(const int , const double* , double* );
void fpk_vml_dCeil_B3LAynn(const int , const double* , double* );
void fpk_vml_dCeil_B3EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_Z0HAynn(const int , const float* , float* );
void fpk_vml_sCeil_Z0LAynn(const int , const float* , float* );
void fpk_vml_sCeil_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_Z0HAynn(const int , const double* , double* );
void fpk_vml_dCeil_Z0LAynn(const int , const double* , double* );
void fpk_vml_dCeil_Z0EPnnn(const int , const double* , double* );


void fpk_vml_sPowx_EXHAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_EXLAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_EXEPnnn(const int , const float*  , const float  , float* );
void fpk_vml_dPowx_EXHAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_EXLAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_EXEPnnn(const int , const double* , const double , double* );

void fpk_vml_sPowx_U8HAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_U8LAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_U8EPnnn(const int , const float*  , const float  , float* );
void fpk_vml_dPowx_U8HAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_U8LAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_U8EPnnn(const int , const double* , const double , double* );

void fpk_vml_sPowx_H8HAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_H8LAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_H8EPnnn(const int , const float*  , const float  , float* );
void fpk_vml_dPowx_H8HAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_H8LAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_H8EPnnn(const int , const double* , const double , double* );

void fpk_vml_sPowx_E9HAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_E9LAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_E9EPnnn(const int , const float*  , const float  , float* );
void fpk_vml_dPowx_E9HAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_E9LAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_E9EPnnn(const int , const double* , const double , double* );

void fpk_vml_sPowx_L9HAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_L9LAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_L9EPnnn(const int , const float*  , const float  , float* );
void fpk_vml_dPowx_L9HAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_L9LAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_L9EPnnn(const int , const double* , const double , double* );

void fpk_vml_sPowx_B3HAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_B3LAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_B3EPnnn(const int , const float*  , const float  , float* );
void fpk_vml_dPowx_B3HAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_B3LAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_B3EPnnn(const int , const double* , const double , double* );

void fpk_vml_sPowx_Z0HAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_Z0LAynn(const int , const float*  , const float  , float* );
void fpk_vml_sPowx_Z0EPnnn(const int , const float*  , const float  , float* );
void fpk_vml_dPowx_Z0HAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_Z0LAynn(const int , const double* , const double , double* );
void fpk_vml_dPowx_Z0EPnnn(const int , const double* , const double , double* );


void fpk_vml_sSqrt_EXHAynn(const int , const float* , float* );
void fpk_vml_sSqrt_EXLAynn(const int , const float* , float* );
void fpk_vml_sSqrt_EXEPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_EXHAynn(const int , const double* , double* );
void fpk_vml_dSqrt_EXLAynn(const int , const double* , double* );
void fpk_vml_dSqrt_EXEPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_U8HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_U8LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_U8EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_U8HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_U8LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_U8EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_H8HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_H8LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_H8EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_H8HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_H8LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_H8EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_E9HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_E9LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_E9EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_E9HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_E9LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_E9EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_L9HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_L9LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_L9EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_L9HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_L9LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_L9EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_B3HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_B3LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_B3EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_B3HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_B3LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_B3EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_Z0HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_Z0LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_Z0HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_Z0LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_Z0EPnnn(const int , const double* , double* );


void fpk_vml_sTanh_EXHAynn(const int , const float* , float* );
void fpk_vml_sTanh_EXLAynn(const int , const float* , float* );
void fpk_vml_sTanh_EXEPnnn(const int , const float* , float* );
void fpk_vml_dTanh_EXHAynn(const int , const double* , double* );
void fpk_vml_dTanh_EXLAynn(const int , const double* , double* );
void fpk_vml_dTanh_EXEPnnn(const int , const double* , double* );

void fpk_vml_sTanh_U8HAynn(const int , const float* , float* );
void fpk_vml_sTanh_U8LAynn(const int , const float* , float* );
void fpk_vml_sTanh_U8EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_U8HAynn(const int , const double* , double* );
void fpk_vml_dTanh_U8LAynn(const int , const double* , double* );
void fpk_vml_dTanh_U8EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_H8HAynn(const int , const float* , float* );
void fpk_vml_sTanh_H8LAynn(const int , const float* , float* );
void fpk_vml_sTanh_H8EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_H8HAynn(const int , const double* , double* );
void fpk_vml_dTanh_H8LAynn(const int , const double* , double* );
void fpk_vml_dTanh_H8EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_E9HAynn(const int , const float* , float* );
void fpk_vml_sTanh_E9LAynn(const int , const float* , float* );
void fpk_vml_sTanh_E9EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_E9HAynn(const int , const double* , double* );
void fpk_vml_dTanh_E9LAynn(const int , const double* , double* );
void fpk_vml_dTanh_E9EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_L9HAynn(const int , const float* , float* );
void fpk_vml_sTanh_L9LAynn(const int , const float* , float* );
void fpk_vml_sTanh_L9EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_L9HAynn(const int , const double* , double* );
void fpk_vml_dTanh_L9LAynn(const int , const double* , double* );
void fpk_vml_dTanh_L9EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_B3HAynn(const int , const float* , float* );
void fpk_vml_sTanh_B3LAynn(const int , const float* , float* );
void fpk_vml_sTanh_B3EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_B3HAynn(const int , const double* , double* );
void fpk_vml_dTanh_B3LAynn(const int , const double* , double* );
void fpk_vml_dTanh_B3EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_Z0HAynn(const int , const float* , float* );
void fpk_vml_sTanh_Z0LAynn(const int , const float* , float* );
void fpk_vml_sTanh_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_Z0HAynn(const int , const double* , double* );
void fpk_vml_dTanh_Z0LAynn(const int , const double* , double* );
void fpk_vml_dTanh_Z0EPnnn(const int , const double* , double* );


void fpk_vml_sLog1p_EXHAynn(const int , const float* , float* );
void fpk_vml_sLog1p_EXLAynn(const int , const float* , float* );
void fpk_vml_sLog1p_EXEPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_EXHAynn(const int , const double* , double* );
void fpk_vml_dLog1p_EXLAynn(const int , const double* , double* );
void fpk_vml_dLog1p_EXEPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_U8HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_U8LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_U8EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_U8HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_U8LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_U8EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_H8HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_H8LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_H8EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_H8HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_H8LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_H8EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_E9HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_E9LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_E9EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_E9HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_E9LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_E9EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_L9HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_L9LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_L9EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_L9HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_L9LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_L9EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_B3HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_B3LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_B3EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_B3HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_B3LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_B3EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_Z0HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_Z0LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_Z0HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_Z0LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_Z0EPnnn(const int , const double* , double* );


void fpk_vml_sCdfNormInv_EXHAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_EXLAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_EXEPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_EXHAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_EXLAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_EXEPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_U8HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_U8LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_U8EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_U8HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_U8LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_U8EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_H8HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_H8LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_H8EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_H8HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_H8LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_H8EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_E9HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_E9LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_E9EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_E9HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_E9LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_E9EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_L9HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_L9LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_L9EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_L9HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_L9LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_L9EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_B3HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_B3LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_B3EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_B3HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_B3LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_B3EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_Z0HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_Z0LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_Z0EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_Z0HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_Z0LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_Z0EPnnn(const int , const double* , double* );


int fpk_vsl_sub_kernel_ex_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_ex_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_ex_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_ex_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_ex_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_ex_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_ex_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_ex_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_ex_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_ex_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_ex_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_ex_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_ex_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_ex_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_ex_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_ex_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_ex_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_ex_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_ex_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_u8_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_u8_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_u8_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_u8_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_u8_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_u8_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_u8_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_u8_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_u8_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_u8_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_u8_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_u8_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_u8_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_u8_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_u8_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_u8_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_u8_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_u8_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_u8_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_h8_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_h8_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_h8_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_h8_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_h8_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_h8_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_h8_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_h8_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_h8_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_h8_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_h8_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_h8_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_h8_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_h8_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_h8_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_h8_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_h8_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_h8_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_h8_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_e9_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_e9_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_e9_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_e9_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_e9_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_e9_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_e9_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_e9_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_e9_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_e9_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_e9_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_e9_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_e9_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_e9_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_e9_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_e9_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_e9_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_e9_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_e9_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_l9_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_l9_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_l9_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_l9_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_l9_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_l9_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_l9_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_l9_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_l9_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_l9_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_l9_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_l9_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_l9_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_l9_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_l9_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_l9_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_l9_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_l9_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_l9_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_b3_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_b3_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_b3_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_b3_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_b3_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_b3_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_b3_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_b3_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_b3_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_b3_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_b3_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_b3_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_b3_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_b3_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_b3_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_b3_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_b3_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_b3_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_b3_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_z0_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_z0_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_z0_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_z0_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_z0_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_z0_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_z0_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_z0_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_z0_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_z0_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_z0_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_z0_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_z0_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_z0_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_z0_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_z0_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_z0_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_z0_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_z0_vslLoadStreamM(void *, const char *);

#else

void fpk_vml_sLn_W7HAynn(const int , const float* , float* );
void fpk_vml_sLn_W7LAynn(const int , const float* , float* );
void fpk_vml_sLn_W7EPnnn(const int , const float* , float* );
void fpk_vml_dLn_W7HAynn(const int , const double* , double* );
void fpk_vml_dLn_W7LAynn(const int , const double* , double* );
void fpk_vml_dLn_W7EPnnn(const int , const double* , double* );

void fpk_vml_sLn_V8HAynn(const int , const float* , float* );
void fpk_vml_sLn_V8LAynn(const int , const float* , float* );
void fpk_vml_sLn_V8EPnnn(const int , const float* , float* );
void fpk_vml_dLn_V8HAynn(const int , const double* , double* );
void fpk_vml_dLn_V8LAynn(const int , const double* , double* );
void fpk_vml_dLn_V8EPnnn(const int , const double* , double* );

void fpk_vml_sLn_N8HAynn(const int , const float* , float* );
void fpk_vml_sLn_N8LAynn(const int , const float* , float* );
void fpk_vml_sLn_N8EPnnn(const int , const float* , float* );
void fpk_vml_dLn_N8HAynn(const int , const double* , double* );
void fpk_vml_dLn_N8LAynn(const int , const double* , double* );
void fpk_vml_dLn_N8EPnnn(const int , const double* , double* );

void fpk_vml_sLn_G9HAynn(const int , const float* , float* );
void fpk_vml_sLn_G9LAynn(const int , const float* , float* );
void fpk_vml_sLn_G9EPnnn(const int , const float* , float* );
void fpk_vml_dLn_G9HAynn(const int , const double* , double* );
void fpk_vml_dLn_G9LAynn(const int , const double* , double* );
void fpk_vml_dLn_G9EPnnn(const int , const double* , double* );

void fpk_vml_sLn_S9HAynn(const int , const float* , float* );
void fpk_vml_sLn_S9LAynn(const int , const float* , float* );
void fpk_vml_sLn_S9EPnnn(const int , const float* , float* );
void fpk_vml_dLn_S9HAynn(const int , const double* , double* );
void fpk_vml_dLn_S9LAynn(const int , const double* , double* );
void fpk_vml_dLn_S9EPnnn(const int , const double* , double* );

void fpk_vml_sLn_A3HAynn(const int , const float* , float* );
void fpk_vml_sLn_A3LAynn(const int , const float* , float* );
void fpk_vml_sLn_A3EPnnn(const int , const float* , float* );
void fpk_vml_dLn_A3HAynn(const int , const double* , double* );
void fpk_vml_dLn_A3LAynn(const int , const double* , double* );
void fpk_vml_dLn_A3EPnnn(const int , const double* , double* );

void fpk_vml_sLn_X0HAynn(const int , const float* , float* );
void fpk_vml_sLn_X0LAynn(const int , const float* , float* );
void fpk_vml_sLn_X0EPnnn(const int , const float* , float* );
void fpk_vml_dLn_X0HAynn(const int , const double* , double* );
void fpk_vml_dLn_X0LAynn(const int , const double* , double* );
void fpk_vml_dLn_X0EPnnn(const int , const double* , double* );


void fpk_vml_sExp_W7HAynn(const int , const float* , float* );
void fpk_vml_sExp_W7LAynn(const int , const float* , float* );
void fpk_vml_sExp_W7EPnnn(const int , const float* , float* );
void fpk_vml_dExp_W7HAynn(const int , const double* , double* );
void fpk_vml_dExp_W7LAynn(const int , const double* , double* );
void fpk_vml_dExp_W7EPnnn(const int , const double* , double* );

void fpk_vml_sExp_V8HAynn(const int , const float* , float* );
void fpk_vml_sExp_V8LAynn(const int , const float* , float* );
void fpk_vml_sExp_V8EPnnn(const int , const float* , float* );
void fpk_vml_dExp_V8HAynn(const int , const double* , double* );
void fpk_vml_dExp_V8LAynn(const int , const double* , double* );
void fpk_vml_dExp_V8EPnnn(const int , const double* , double* );

void fpk_vml_sExp_N8HAynn(const int , const float* , float* );
void fpk_vml_sExp_N8LAynn(const int , const float* , float* );
void fpk_vml_sExp_N8EPnnn(const int , const float* , float* );
void fpk_vml_dExp_N8HAynn(const int , const double* , double* );
void fpk_vml_dExp_N8LAynn(const int , const double* , double* );
void fpk_vml_dExp_N8EPnnn(const int , const double* , double* );

void fpk_vml_sExp_G9HAynn(const int , const float* , float* );
void fpk_vml_sExp_G9LAynn(const int , const float* , float* );
void fpk_vml_sExp_G9EPnnn(const int , const float* , float* );
void fpk_vml_dExp_G9HAynn(const int , const double* , double* );
void fpk_vml_dExp_G9LAynn(const int , const double* , double* );
void fpk_vml_dExp_G9EPnnn(const int , const double* , double* );

void fpk_vml_sExp_S9HAynn(const int , const float* , float* );
void fpk_vml_sExp_S9LAynn(const int , const float* , float* );
void fpk_vml_sExp_S9EPnnn(const int , const float* , float* );
void fpk_vml_dExp_S9HAynn(const int , const double* , double* );
void fpk_vml_dExp_S9LAynn(const int , const double* , double* );
void fpk_vml_dExp_S9EPnnn(const int , const double* , double* );

void fpk_vml_sExp_A3HAynn(const int , const float* , float* );
void fpk_vml_sExp_A3LAynn(const int , const float* , float* );
void fpk_vml_sExp_A3EPnnn(const int , const float* , float* );
void fpk_vml_dExp_A3HAynn(const int , const double* , double* );
void fpk_vml_dExp_A3LAynn(const int , const double* , double* );
void fpk_vml_dExp_A3EPnnn(const int , const double* , double* );

void fpk_vml_sExp_X0HAynn(const int , const float* , float* );
void fpk_vml_sExp_X0LAynn(const int , const float* , float* );
void fpk_vml_sExp_X0EPnnn(const int , const float* , float* );
void fpk_vml_dExp_X0HAynn(const int , const double* , double* );
void fpk_vml_dExp_X0LAynn(const int , const double* , double* );
void fpk_vml_dExp_X0EPnnn(const int , const double* , double* );


void fpk_vml_sErf_W7HAynn(const int , const float* , float* );
void fpk_vml_sErf_W7LAynn(const int , const float* , float* );
void fpk_vml_sErf_W7EPnnn(const int , const float* , float* );
void fpk_vml_dErf_W7HAynn(const int , const double* , double* );
void fpk_vml_dErf_W7LAynn(const int , const double* , double* );
void fpk_vml_dErf_W7EPnnn(const int , const double* , double* );

void fpk_vml_sErf_V8HAynn(const int , const float* , float* );
void fpk_vml_sErf_V8LAynn(const int , const float* , float* );
void fpk_vml_sErf_V8EPnnn(const int , const float* , float* );
void fpk_vml_dErf_V8HAynn(const int , const double* , double* );
void fpk_vml_dErf_V8LAynn(const int , const double* , double* );
void fpk_vml_dErf_V8EPnnn(const int , const double* , double* );

void fpk_vml_sErf_N8HAynn(const int , const float* , float* );
void fpk_vml_sErf_N8LAynn(const int , const float* , float* );
void fpk_vml_sErf_N8EPnnn(const int , const float* , float* );
void fpk_vml_dErf_N8HAynn(const int , const double* , double* );
void fpk_vml_dErf_N8LAynn(const int , const double* , double* );
void fpk_vml_dErf_N8EPnnn(const int , const double* , double* );

void fpk_vml_sErf_G9HAynn(const int , const float* , float* );
void fpk_vml_sErf_G9LAynn(const int , const float* , float* );
void fpk_vml_sErf_G9EPnnn(const int , const float* , float* );
void fpk_vml_dErf_G9HAynn(const int , const double* , double* );
void fpk_vml_dErf_G9LAynn(const int , const double* , double* );
void fpk_vml_dErf_G9EPnnn(const int , const double* , double* );

void fpk_vml_sErf_S9HAynn(const int , const float* , float* );
void fpk_vml_sErf_S9LAynn(const int , const float* , float* );
void fpk_vml_sErf_S9EPnnn(const int , const float* , float* );
void fpk_vml_dErf_S9HAynn(const int , const double* , double* );
void fpk_vml_dErf_S9LAynn(const int , const double* , double* );
void fpk_vml_dErf_S9EPnnn(const int , const double* , double* );

void fpk_vml_sErf_A3HAynn(const int , const float* , float* );
void fpk_vml_sErf_A3LAynn(const int , const float* , float* );
void fpk_vml_sErf_A3EPnnn(const int , const float* , float* );
void fpk_vml_dErf_A3HAynn(const int , const double* , double* );
void fpk_vml_dErf_A3LAynn(const int , const double* , double* );
void fpk_vml_dErf_A3EPnnn(const int , const double* , double* );

void fpk_vml_sErf_X0HAynn(const int , const float* , float* );
void fpk_vml_sErf_X0LAynn(const int , const float* , float* );
void fpk_vml_sErf_X0EPnnn(const int , const float* , float* );
void fpk_vml_dErf_X0HAynn(const int , const double* , double* );
void fpk_vml_dErf_X0LAynn(const int , const double* , double* );
void fpk_vml_dErf_X0EPnnn(const int , const double* , double* );


void fpk_vml_sErfInv_W7HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_W7LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_W7EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_W7HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_W7LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_W7EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_V8HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_V8LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_V8EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_V8HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_V8LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_V8EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_N8HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_N8LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_N8EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_N8HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_N8LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_N8EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_G9HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_G9LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_G9EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_G9HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_G9LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_G9EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_S9HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_S9LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_S9EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_S9HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_S9LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_S9EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_A3HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_A3LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_A3EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_A3HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_A3LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_A3EPnnn(const int , const double* , double* );

void fpk_vml_sErfInv_X0HAynn(const int , const float* , float* );
void fpk_vml_sErfInv_X0LAynn(const int , const float* , float* );
void fpk_vml_sErfInv_X0EPnnn(const int , const float* , float* );
void fpk_vml_dErfInv_X0HAynn(const int , const double* , double* );
void fpk_vml_dErfInv_X0LAynn(const int , const double* , double* );
void fpk_vml_dErfInv_X0EPnnn(const int , const double* , double* );


void fpk_vml_sCeil_W7HAynn(const int , const float* , float* );
void fpk_vml_sCeil_W7LAynn(const int , const float* , float* );
void fpk_vml_sCeil_W7EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_W7HAynn(const int , const double* , double* );
void fpk_vml_dCeil_W7LAynn(const int , const double* , double* );
void fpk_vml_dCeil_W7EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_V8HAynn(const int , const float* , float* );
void fpk_vml_sCeil_V8LAynn(const int , const float* , float* );
void fpk_vml_sCeil_V8EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_V8HAynn(const int , const double* , double* );
void fpk_vml_dCeil_V8LAynn(const int , const double* , double* );
void fpk_vml_dCeil_V8EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_N8HAynn(const int , const float* , float* );
void fpk_vml_sCeil_N8LAynn(const int , const float* , float* );
void fpk_vml_sCeil_N8EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_N8HAynn(const int , const double* , double* );
void fpk_vml_dCeil_N8LAynn(const int , const double* , double* );
void fpk_vml_dCeil_N8EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_G9HAynn(const int , const float* , float* );
void fpk_vml_sCeil_G9LAynn(const int , const float* , float* );
void fpk_vml_sCeil_G9EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_G9HAynn(const int , const double* , double* );
void fpk_vml_dCeil_G9LAynn(const int , const double* , double* );
void fpk_vml_dCeil_G9EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_S9HAynn(const int , const float* , float* );
void fpk_vml_sCeil_S9LAynn(const int , const float* , float* );
void fpk_vml_sCeil_S9EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_S9HAynn(const int , const double* , double* );
void fpk_vml_dCeil_S9LAynn(const int , const double* , double* );
void fpk_vml_dCeil_S9EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_A3HAynn(const int , const float* , float* );
void fpk_vml_sCeil_A3LAynn(const int , const float* , float* );
void fpk_vml_sCeil_A3EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_A3HAynn(const int , const double* , double* );
void fpk_vml_dCeil_A3LAynn(const int , const double* , double* );
void fpk_vml_dCeil_A3EPnnn(const int , const double* , double* );

void fpk_vml_sCeil_X0HAynn(const int , const float* , float* );
void fpk_vml_sCeil_X0LAynn(const int , const float* , float* );
void fpk_vml_sCeil_X0EPnnn(const int , const float* , float* );
void fpk_vml_dCeil_X0HAynn(const int , const double* , double* );
void fpk_vml_dCeil_X0LAynn(const int , const double* , double* );
void fpk_vml_dCeil_X0EPnnn(const int , const double* , double* );


void fpk_vml_sPowx_W7HAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_W7LAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_W7EPnnn(const int , const float* , const float  , float* );
void fpk_vml_dPowx_W7HAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_W7LAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_W7EPnnn(const int , const double*, const double , double* );

void fpk_vml_sPowx_V8HAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_V8LAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_V8EPnnn(const int , const float* , const float  , float* );
void fpk_vml_dPowx_V8HAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_V8LAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_V8EPnnn(const int , const double*, const double , double* );

void fpk_vml_sPowx_N8HAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_N8LAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_N8EPnnn(const int , const float* , const float  , float* );
void fpk_vml_dPowx_N8HAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_N8LAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_N8EPnnn(const int , const double*, const double , double* );

void fpk_vml_sPowx_G9HAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_G9LAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_G9EPnnn(const int , const float* , const float  , float* );
void fpk_vml_dPowx_G9HAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_G9LAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_G9EPnnn(const int , const double*, const double , double* );

void fpk_vml_sPowx_S9HAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_S9LAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_S9EPnnn(const int , const float* , const float  , float* );
void fpk_vml_dPowx_S9HAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_S9LAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_S9EPnnn(const int , const double*, const double , double* );

void fpk_vml_sPowx_A3HAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_A3LAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_A3EPnnn(const int , const float* , const float  , float* );
void fpk_vml_dPowx_A3HAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_A3LAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_A3EPnnn(const int , const double*, const double , double* );

void fpk_vml_sPowx_X0HAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_X0LAynn(const int , const float* , const float  , float* );
void fpk_vml_sPowx_X0EPnnn(const int , const float* , const float  , float* );
void fpk_vml_dPowx_X0HAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_X0LAynn(const int , const double*, const double , double* );
void fpk_vml_dPowx_X0EPnnn(const int , const double*, const double , double* );


void fpk_vml_sSqrt_W7HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_W7LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_W7EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_W7HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_W7LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_W7EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_V8HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_V8LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_V8EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_V8HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_V8LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_V8EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_N8HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_N8LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_N8EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_N8HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_N8LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_N8EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_G9HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_G9LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_G9EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_G9HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_G9LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_G9EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_S9HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_S9LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_S9EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_S9HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_S9LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_S9EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_A3HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_A3LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_A3EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_A3HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_A3LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_A3EPnnn(const int , const double* , double* );

void fpk_vml_sSqrt_X0HAynn(const int , const float* , float* );
void fpk_vml_sSqrt_X0LAynn(const int , const float* , float* );
void fpk_vml_sSqrt_X0EPnnn(const int , const float* , float* );
void fpk_vml_dSqrt_X0HAynn(const int , const double* , double* );
void fpk_vml_dSqrt_X0LAynn(const int , const double* , double* );
void fpk_vml_dSqrt_X0EPnnn(const int , const double* , double* );


void fpk_vml_sTanh_W7HAynn(const int , const float* , float* );
void fpk_vml_sTanh_W7LAynn(const int , const float* , float* );
void fpk_vml_sTanh_W7EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_W7HAynn(const int , const double* , double* );
void fpk_vml_dTanh_W7LAynn(const int , const double* , double* );
void fpk_vml_dTanh_W7EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_V8HAynn(const int , const float* , float* );
void fpk_vml_sTanh_V8LAynn(const int , const float* , float* );
void fpk_vml_sTanh_V8EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_V8HAynn(const int , const double* , double* );
void fpk_vml_dTanh_V8LAynn(const int , const double* , double* );
void fpk_vml_dTanh_V8EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_N8HAynn(const int , const float* , float* );
void fpk_vml_sTanh_N8LAynn(const int , const float* , float* );
void fpk_vml_sTanh_N8EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_N8HAynn(const int , const double* , double* );
void fpk_vml_dTanh_N8LAynn(const int , const double* , double* );
void fpk_vml_dTanh_N8EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_G9HAynn(const int , const float* , float* );
void fpk_vml_sTanh_G9LAynn(const int , const float* , float* );
void fpk_vml_sTanh_G9EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_G9HAynn(const int , const double* , double* );
void fpk_vml_dTanh_G9LAynn(const int , const double* , double* );
void fpk_vml_dTanh_G9EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_S9HAynn(const int , const float* , float* );
void fpk_vml_sTanh_S9LAynn(const int , const float* , float* );
void fpk_vml_sTanh_S9EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_S9HAynn(const int , const double* , double* );
void fpk_vml_dTanh_S9LAynn(const int , const double* , double* );
void fpk_vml_dTanh_S9EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_A3HAynn(const int , const float* , float* );
void fpk_vml_sTanh_A3LAynn(const int , const float* , float* );
void fpk_vml_sTanh_A3EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_A3HAynn(const int , const double* , double* );
void fpk_vml_dTanh_A3LAynn(const int , const double* , double* );
void fpk_vml_dTanh_A3EPnnn(const int , const double* , double* );

void fpk_vml_sTanh_X0HAynn(const int , const float* , float* );
void fpk_vml_sTanh_X0LAynn(const int , const float* , float* );
void fpk_vml_sTanh_X0EPnnn(const int , const float* , float* );
void fpk_vml_dTanh_X0HAynn(const int , const double* , double* );
void fpk_vml_dTanh_X0LAynn(const int , const double* , double* );
void fpk_vml_dTanh_X0EPnnn(const int , const double* , double* );


void fpk_vml_sLog1p_W7HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_W7LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_W7EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_W7HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_W7LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_W7EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_V8HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_V8LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_V8EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_V8HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_V8LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_V8EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_N8HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_N8LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_N8EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_N8HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_N8LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_N8EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_G9HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_G9LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_G9EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_G9HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_G9LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_G9EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_S9HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_S9LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_S9EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_S9HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_S9LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_S9EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_A3HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_A3LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_A3EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_A3HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_A3LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_A3EPnnn(const int , const double* , double* );

void fpk_vml_sLog1p_X0HAynn(const int , const float* , float* );
void fpk_vml_sLog1p_X0LAynn(const int , const float* , float* );
void fpk_vml_sLog1p_X0EPnnn(const int , const float* , float* );
void fpk_vml_dLog1p_X0HAynn(const int , const double* , double* );
void fpk_vml_dLog1p_X0LAynn(const int , const double* , double* );
void fpk_vml_dLog1p_X0EPnnn(const int , const double* , double* );


void fpk_vml_sCdfNormInv_W7HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_W7LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_W7EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_W7HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_W7LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_W7EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_V8HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_V8LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_V8EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_V8HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_V8LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_V8EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_N8HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_N8LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_N8EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_N8HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_N8LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_N8EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_G9HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_G9LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_G9EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_G9HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_G9LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_G9EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_S9HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_S9LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_S9EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_S9HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_S9LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_S9EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_A3HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_A3LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_A3EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_A3HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_A3LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_A3EPnnn(const int , const double* , double* );

void fpk_vml_sCdfNormInv_X0HAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_X0LAynn(const int , const float* , float* );
void fpk_vml_sCdfNormInv_X0EPnnn(const int , const float* , float* );
void fpk_vml_dCdfNormInv_X0HAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_X0LAynn(const int , const double* , double* );
void fpk_vml_dCdfNormInv_X0EPnnn(const int , const double* , double* );


int fpk_vsl_sub_kernel_w7_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_w7_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_w7_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_w7_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_w7_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_w7_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_w7_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_w7_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_w7_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_w7_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_w7_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_w7_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_w7_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_w7_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_w7_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_w7_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_w7_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_w7_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_w7_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_v8_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_v8_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_v8_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_v8_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_v8_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_v8_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_v8_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_v8_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_v8_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_v8_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_v8_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_v8_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_v8_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_v8_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_v8_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_v8_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_v8_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_v8_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_v8_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_n8_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_n8_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_n8_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_n8_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_n8_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_n8_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_n8_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_n8_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_n8_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_n8_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_n8_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_n8_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_n8_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_n8_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_n8_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_n8_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_n8_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_n8_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_n8_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_g9_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_g9_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_g9_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_g9_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_g9_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_g9_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_g9_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_g9_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_g9_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_g9_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_g9_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_g9_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_g9_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_g9_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_g9_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_g9_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_g9_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_g9_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_g9_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_s9_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_s9_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_s9_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_s9_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_s9_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_s9_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_s9_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_s9_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_s9_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_s9_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_s9_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_s9_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_s9_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_s9_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_s9_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_s9_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_s9_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_s9_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_s9_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_a3_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_a3_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_a3_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_a3_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_a3_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_a3_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_a3_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_a3_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_a3_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_a3_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_a3_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_a3_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_a3_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_a3_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_a3_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_a3_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_a3_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_a3_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_a3_vslLoadStreamM(void *, const char *);

int fpk_vsl_sub_kernel_x0_vsldSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const double *, const double *, const __int64 *, const int );
int fpk_vsl_sub_kernel_x0_vslsSSNewTask(DAAL_VSLSSTaskPtr *, const __int64 *, const __int64 *, const __int64 *, const float *, const float *, const __int64 *, const int );
int fpk_vsl_sub_kernel_x0_vsldSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const double *);
int fpk_vsl_sub_kernel_x0_vslsSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const float *);
int fpk_vsl_sub_kernel_x0_vsliSSEditTask(DAAL_VSLSSTaskPtr , const __int64 , const __int64 *);
int fpk_vsl_sub_kernel_x0_vslSSDeleteTask(DAAL_VSLSSTaskPtr *);
int fpk_vsl_kernel_x0_dSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSBasic(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSOutliersDetection(void* , const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_sub_kernel_x0_vslsSSEditOutDetect(void *, const __int64 *, const float *, const float *);
int fpk_vsl_sub_kernel_x0_vsldSSEditOutDetect(void *, const __int64 *, const double *, const double *);
int fpk_vsl_kernel_x0_dSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSMahDistance(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSSort(void *, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSStreamQuantiles(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_dSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_sSSMissingValues(void*, const __int64 , const __int64 , const struct ThreadingFuncs *);
int fpk_vsl_kernel_x0_iRngUniform(const int , void * , const int , const int [] , const int , const int );
int fpk_vsl_kernel_x0_sRngUniform(const int , void * , const int , const float [] , const float , const float );
int fpk_vsl_kernel_x0_dRngUniform(const int , void * , const int , const double [] , const double , const double );
int fpk_vsl_kernel_x0_iRngBernoulli(const int , void * , const int , const int [] , const double );
int fpk_vsl_sub_kernel_x0_vslNewStreamEx(void *, const int , const int , const unsigned __int32 []);
int fpk_vsl_sub_kernel_x0_vslDeleteStream(void *);
int fpk_vsl_sub_kernel_x0_vslGetStreamSize(const void *);
int fpk_vsl_sub_kernel_x0_vslSaveStreamM(const void *, char *);
int fpk_vsl_sub_kernel_x0_vslLoadStreamM(void *, const char *);

#endif


#if defined(__cplusplus)
}
#endif


#endif

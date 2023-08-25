/* file: service_stat_rng_ref
.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Template wrappers for common REF functions.
//--
*/

#ifndef __SERVICE_STAT_RNG_REF_H__
#define __SERVICE_STAT_RNG_REF_H__

#include "src/externals/service_stat_rng_ref.h"

#if !defined(__DAAL_CONCAT2)
    #define __DAAL_CONCAT2(a, b) a##b
#endif

#define __DAAL_VSLFN(f_pref, f_name)                          __DAAL_CONCAT2(f_pref, f_name)
#define __DAAL_VSLFN_CALL_NR(f_pref, f_name, f_args, errcode) __DAAL_VSLFN_CALL(f_pref, f_name, f_args, errcode)
#define __DAAL_VSLFN_CALL_NR_WHILE(f_pref, f_name, f_args, errcode) \
    {                                                               \
        __DAAL_VSLFN_CALL(f_pref, f_name, f_args, errcode);         \
    }

#define __DAAL_VSLFN_CALL(f_pref, f_name, f_args, errcode) \
    {                                                      \
        errcode = __DAAL_VSLFN(f_pref, f_name) f_args;     \
    }

#endif

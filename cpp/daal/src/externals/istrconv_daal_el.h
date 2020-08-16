/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

//
// Abstract:
//
// External header file for libistrconv.
//
// =============================================================================

#ifndef _ISTRCONV_EL_H_
#define _ISTRCONV_EL_H_

#if defined(__cplusplus)
    #define _ISTRCONV_EXTERN_C extern "C"
#else
    #define _ISTRCONV_EXTERN_C extern
#endif

_ISTRCONV_EXTERN_C int __FPK_string_to_int_generic(const char * nptr, char ** endptr);

#endif /*_ISTRCONV_H_*/

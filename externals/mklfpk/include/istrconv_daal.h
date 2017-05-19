// File: istrconv.h
/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef _ISTRCONV_H_
#define _ISTRCONV_H_

#if defined(__cplusplus)
#define _ISTRCONV_EXTERN_C extern "C"
#else
#define _ISTRCONV_EXTERN_C extern
#endif

_ISTRCONV_EXTERN_C float __FPK_string_to_float(const char * nptr,
                                               char ** endptr);

_ISTRCONV_EXTERN_C double __FPK_string_to_double(const char * nptr,
                                                 char ** endptr);

_ISTRCONV_EXTERN_C int __FPK_string_to_int_generic(const char * nptr,
                                                   char ** endptr);

_ISTRCONV_EXTERN_C unsigned int __FPK_string_to_uint_generic(const char * nptr,
                                                             char ** endptr);

_ISTRCONV_EXTERN_C long long __FPK_string_to_int64_generic(const char * nptr,
                                                           char ** endptr);

_ISTRCONV_EXTERN_C unsigned long long __FPK_string_to_uint64_generic(const char * nptr,
                                                                     char ** endptr);

_ISTRCONV_EXTERN_C int __FPK_string_to_int_sse4(const char * nptr,
                                                char ** endptr);

_ISTRCONV_EXTERN_C unsigned int __FPK_string_to_uint_sse4(const char * nptr,
                                                          char ** endptr);

_ISTRCONV_EXTERN_C long long __FPK_string_to_int64_sse4(const char * nptr,
                                                        char ** endptr);

_ISTRCONV_EXTERN_C unsigned long long __FPK_string_to_uint64_sse4(const char * nptr,
                                                                  char ** endptr);

#endif /*_ISTRCONV_H_*/

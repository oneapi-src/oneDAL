/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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

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

_ISTRCONV_EXTERN_C float __FPK_string_to_float(const char * nptr, char ** endptr);

_ISTRCONV_EXTERN_C double __FPK_string_to_double(const char * nptr, char ** endptr);

_ISTRCONV_EXTERN_C int __FPK_double_to_string_f(char * str, size_t n, double x);

_ISTRCONV_EXTERN_C int __FPK_int_to_string(char * str, size_t n, int x);


#endif /*_ISTRCONV_H_*/

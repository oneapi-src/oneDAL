/* file: service_service_mkl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Declaration of math service functions
//--
*/

#ifndef __SERVICE_SERVICE_MKL_H__
#define __SERVICE_SERVICE_MKL_H__

#include "services/daal_defines.h"
#include "mkl_daal.h"
#include "istrconv_daal.h"
#include "istrconv_daal_el.h"

namespace daal
{
namespace internal
{
namespace mkl
{
struct MklService
{
    static void * serv_malloc(size_t size, size_t alignment) { return fpk_serv_malloc(size, alignment); }

    static void serv_free(void * ptr) { fpk_serv_free(ptr); }

    static void serv_free_buffers() { fpk_serv_free_buffers(); }

    static int serv_memcpy_s(void * dest, size_t destSize, const void * src, size_t srcSize)
    {
        return fpk_serv_memcpy_s(dest, destSize, src, srcSize);
    }

    static int serv_memmove_s(void * dest, size_t destSize, const void * src, size_t smax) { return fpk_serv_memmove_s(dest, destSize, src, smax); }

    static int serv_get_ht() { return fpk_serv_get_ht(); }

    static int serv_get_ncpus() { return fpk_serv_get_ncpus(); }

    static int serv_get_ncorespercpu() { return fpk_serv_get_ncorespercpu(); }

    static int serv_set_memory_limit(int type, size_t limit) { return fpk_serv_set_memory_limit(type, limit); }

    // Added for interface compatibility - not expected to be called
    static size_t serv_strnlen_s(const char * src, size_t slen)
    {
        size_t i = 0;
        for (; i < slen && src[i] != '\0'; ++i)
            ;
        return i;
    }

    static int serv_strncpy_s(char * dest, size_t dmax, const char * src, size_t slen) { return fpk_serv_strncpy_s(dest, dmax, src, slen); }

    static int serv_strncat_s(char * dest, size_t dmax, const char * src, size_t slen) { return fpk_serv_strncat_s(dest, dmax, src, slen); }

    static float serv_string_to_float(const char * nptr, char ** endptr) { return __FPK_string_to_float(nptr, endptr); }

    static double serv_string_to_double(const char * nptr, char ** endptr) { return __FPK_string_to_double(nptr, endptr); }

    static int serv_string_to_int(const char * nptr, char ** endptr) { return __FPK_string_to_int_generic(nptr, endptr); }

    static int serv_int_to_string(char * buffer, size_t n, int value) { return __FPK_int_to_string(buffer, n, value); }

    static int serv_double_to_string(char * buffer, size_t n, double value) { return __FPK_double_to_string_f(buffer, n, value); }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif

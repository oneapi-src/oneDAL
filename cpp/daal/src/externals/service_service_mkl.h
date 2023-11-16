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
// #include "mkl_daal.h"
#include <mkl.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <errno.h>

namespace daal
{
namespace internal
{
namespace mkl
{
struct MklService
{
    static void * serv_malloc(size_t size, size_t alignment) { return aligned_alloc(size, alignment); }

    static void serv_free(void * ptr) { free(ptr); }

    static void serv_free_buffers() { mkl_free_buffers(); }

    static int serv_memcpy_s(void * dest, size_t destSize, const void * src, size_t srcSize)
    {
        if (destSize < srcSize) return static_cast<int>(ENOMEM);
        memcpy(dest, src, srcSize);
        return 0;
    }

    static int serv_memmove_s(void * dest, size_t destSize, const void * src, size_t smax)
    {
        if (destSize < smax) return static_cast<int>(ENOMEM);
        memmove(dest, src, smax);
        return 0;
    }

    static int serv_get_ht() { return 0; }

    static int serv_get_ncpus() { return 224; }

    static int serv_get_ncorespercpu() { return 1; }

    static int serv_set_memory_limit(int type, size_t limit) { return 0; }

    // Added for interface compatibility - not expected to be called
    static size_t serv_strnlen_s(const char * src, size_t slen)
    {
        size_t i = 0;
        for (; i < slen && src[i] != '\0'; ++i)
            ;
        return i;
    }

    static int serv_strncpy_s(char * dest, size_t dmax, const char * src, size_t slen)
    {
        if (dmax < slen) return static_cast<int>(ENOMEM);
        strncpy(dest, src, slen);
        return 0;
        // TODO: safe funtion
        // return strncpy_s(dest, dmax, src, slen);
    }

    static int serv_strncat_s(char * dest, size_t dmax, const char * src, size_t slen)
    {
        if (dmax < slen) return static_cast<int>(ENOMEM);
        strncat(dest, src, slen);
        return 0;
        // TODO: safe funtion
        // return strncat_s(dest, dmax, src, slen);
    }

    static double serv_string_to_double(const char * nptr, char ** endptr)
    {
        const char * cur = nptr;
        for (; isdigit(*cur) || *cur == '-' || *cur == 'e' || *cur == 'E' || *cur == '.'; ++cur)
            ;
        if (endptr) *endptr = const_cast<char *>(cur);
        size_t size = cur - nptr;
        // TODO replace with static buffer
        char * buffer = static_cast<char *>(malloc(size + 1));
        for (size_t i = 0; i < size; ++i) buffer[i] = nptr[i];
        buffer[size] = '\0';
        double val   = atof(buffer);
        free(buffer);
        return val;
    }

    static float serv_string_to_float(const char * nptr, char ** endptr) { return static_cast<float>(serv_string_to_double(nptr, endptr)); }

    static int serv_string_to_int(const char * nptr, char ** endptr)
    {
        const char * cur = nptr;
        for (; isdigit(*cur) || *cur == '-'; ++cur)
            ;
        if (endptr) *endptr = const_cast<char *>(cur);
        size_t size = cur - nptr;
        // TODO replace with static buffer
        char * buffer = static_cast<char *>(malloc(size + 1));
        for (size_t i = 0; i < size; ++i) buffer[i] = nptr[i];
        buffer[size] = '\0';
        int val      = atoi(buffer);
        free(buffer);
        return val;
    }

    static int serv_int_to_string(char * buffer, size_t n, int value)
    {
        return snprintf(buffer, n, "%d", value);
        // TODO: safe funtion
        // return snprintf_s(buffer, n, "%d", value);
    }

    static int serv_double_to_string(char * buffer, size_t n, double value)
    {
        return snprintf(buffer, n, "%E", value);
        // TODO: safe funtion
        // return snprintf_s(buffer, n, "%E", value);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif

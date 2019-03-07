/* file: service_service_mkl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

/*
//++
//  Declaration of math service functions
//--
*/

#ifndef __SERVICE_SERVICE_MKL_H__
#define __SERVICE_SERVICE_MKL_H__


#include "mkl_daal.h"
#include "istrconv_daal.h"


namespace daal
{
namespace internal
{
namespace mkl
{

struct MklService
{
    static void * serv_malloc(size_t size, size_t alignment)
    {
        return fpk_serv_malloc(size, alignment);
    }

    static void serv_free(void *ptr)
    {
        fpk_serv_free(ptr);
    }

    static void serv_free_buffers()
    {
        fpk_serv_free_buffers();
    }

    static int serv_memcpy_s(void *dest, size_t destSize, const void *src, size_t srcSize)
    {
        return fpk_serv_memcpy_s(dest, destSize, src, srcSize);
    }

    static int serv_get_ht()
    {
        return fpk_serv_get_ht();
    }

    static int serv_get_ncpus()
    {
        return fpk_serv_get_ncpus();
    }

    static int serv_get_ncorespercpu()
    {
        return fpk_serv_get_ncorespercpu();
    }

    static int serv_set_memory_limit(int type, size_t limit)
    {
        return fpk_serv_set_memory_limit(type, limit);
    }

    static int serv_strncpy_s(char *dest, size_t dmax, const char *src, size_t slen)
    {
        return fpk_serv_strncpy_s(dest, dmax, src, slen);
    }

    static int serv_strncat_s(char *dest, size_t dmax, const char *src, size_t slen)
    {
        return fpk_serv_strncat_s(dest, dmax, src, slen);
    }

    static float serv_string_to_float(const char * nptr, char ** endptr) {
        return __FPK_string_to_float(nptr, endptr);
    }

    static double serv_string_to_double(const char * nptr, char ** endptr) {
        return __FPK_string_to_double(nptr, endptr);
    }

    static int serv_int_to_string(char * buffer, size_t n, int value) {
        return __FPK_int_to_string(buffer, n, value);
    }

    static int serv_double_to_string(char * buffer, size_t n, double value) {
        return __FPK_double_to_string_f(buffer, n, value);
    }
};

}
}
}

#endif

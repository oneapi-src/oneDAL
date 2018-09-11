/* file: service_service.h */
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

/*
//++
//  Template wrappers for math functions.
//--
*/

#ifndef __SERVICE_SERVICE_H__
#define __SERVICE_SERVICE_H__

#include "service_service_mkl.h"
#include "daal_memory.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<class _impl=mkl::MklService>
struct Service
{
    static void * serv_malloc(size_t size, size_t alignment)
    {
        return _impl::serv_malloc(size, alignment);
    }

    static void serv_free(void *ptr)
    {
        _impl::serv_free(ptr);
    }

    static void serv_free_buffers()
    {
        _impl::serv_free_buffers();
    }

    static int serv_memcpy_s(void *dest, size_t destSize, const void *src, size_t srcSize)
    {
        return _impl::serv_memcpy_s(dest, destSize, src, srcSize);
    }

    static int serv_get_ht()
    {
        return _impl::serv_get_ht();
    }

    static int serv_get_ncpus()
    {
        return _impl::serv_get_ncpus();
    }

    static int serv_get_ncorespercpu()
    {
        return _impl::serv_get_ncorespercpu();
    }

    static int serv_set_memory_limit(int type, size_t limit)
    {
        return _impl::serv_set_memory_limit(type, limit);
    }

    static int serv_strncpy_s(char *dest, size_t dmax, const char *src, size_t slen)
    {
        return _impl::serv_strncpy_s(dest, dmax, src, slen);
    }

    static int serv_strncat_s(char *dest, size_t dmax, const char *src, size_t slen)
    {
        return _impl::serv_strncat_s(dest, dmax, src, slen);
    }

    static float serv_string_to_float(const char * nptr, char ** endptr) {
        return _impl::serv_string_to_float(nptr, endptr);
    }

    static double serv_string_to_double(const char * nptr, char ** endptr) {
        return _impl::serv_string_to_double(nptr, endptr);
    }
};

} // namespace internal
} // namespace daal

#endif

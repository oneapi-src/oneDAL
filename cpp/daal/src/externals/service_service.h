/* file: service_service.h */
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
//  Template wrappers for math functions.
//--
*/

#ifndef __SERVICE_SERVICE_H__
#define __SERVICE_SERVICE_H__

#include "services/daal_memory.h"
#include "src/externals/config.h"

namespace daal
{
namespace internal
{
/*
// Template functions definition
*/
template <class _impl>
struct Service
{
    static void * serv_malloc(size_t size, size_t alignment) { return _impl::serv_malloc(size, alignment); }

    static void serv_free(void * ptr) { _impl::serv_free(ptr); }

    static void serv_free_buffers() { _impl::serv_free_buffers(); }

    static int serv_memcpy_s(void * dest, size_t destSize, const void * src, size_t srcSize)
    {
        return _impl::serv_memcpy_s(dest, destSize, src, srcSize);
    }

    static int serv_memmove_s(void * dest, size_t destSize, const void * src, size_t smax)
    {
        return _impl::serv_memmove_s(dest, destSize, src, smax);
    }

    static int serv_get_ht() { return _impl::serv_get_ht(); }

    static int serv_get_ncpus() { return _impl::serv_get_ncpus(); }

    static int serv_get_ncorespercpu() { return _impl::serv_get_ncorespercpu(); }

    static int serv_set_memory_limit(int type, size_t limit) { return _impl::serv_set_memory_limit(type, limit); }

    static size_t serv_strnlen_s(const char * src, size_t slen) { return _impl::serv_strnlen_s(src, slen); }

    static int serv_strncpy_s(char * dest, size_t dmax, const char * src, size_t slen) { return _impl::serv_strncpy_s(dest, dmax, src, slen); }

    static int serv_strncat_s(char * dest, size_t dmax, const char * src, size_t slen) { return _impl::serv_strncat_s(dest, dmax, src, slen); }

    static float serv_string_to_float(const char * nptr, char ** endptr) { return _impl::serv_string_to_float(nptr, endptr); }

    static double serv_string_to_double(const char * nptr, char ** endptr) { return _impl::serv_string_to_double(nptr, endptr); }

    static int serv_string_to_int(const char * nptr, char ** endptr) { return _impl::serv_string_to_int(nptr, endptr); }

    static int serv_int_to_string(char * buffer, size_t n, int value) { return _impl::serv_int_to_string(buffer, n, value); }

    static int serv_double_to_string(char * buffer, size_t n, double value) { return _impl::serv_double_to_string(buffer, n, value); }
};

} // namespace internal
} // namespace daal

namespace daal
{
namespace internal
{
using ServiceInst = Service<ServiceBackend>;
} // namespace internal
} // namespace daal

#endif

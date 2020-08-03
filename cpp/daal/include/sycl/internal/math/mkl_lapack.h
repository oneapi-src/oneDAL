/* file: mkl_lapack.h */
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

/*
//++
//  Wrappers for LAPACK functions.
//--
*/

#ifndef __ONEAPI_INTERNAL_MKL_LAPACK_H__
#define __ONEAPI_INTERNAL_MKL_LAPACK_H__

#include "services/buffer.h"
#include "mkl_dal_sycl.hpp"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{
namespace interface1
{
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLPOTRF"></a>
 *  \brief Adapter for MKL POTRF routine
 */
template <typename algorithmFPType>
struct MKLPotrf
{
    MKLPotrf(cl::sycl::queue & queue) : _queue(queue) {}

    services::Status operator()(const math::UpLo uplo, const size_t n, services::Buffer<algorithmFPType> & a, const size_t lda,
                                cl::sycl::buffer<algorithmFPType, 1> & scratchpad)
    {
        services::Status status;
        const ::oneapi::fpk::uplo uplomkl                = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;
        cl::sycl::buffer<algorithmFPType, 1> a_sycl_buff = a.toSycl();

        {
            using namespace daal::services;
            const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrf_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, lda);
            if (scratchpad.get_count() < minimalScratchpadSize) return Status(ErrorID::ErrorMemoryAllocationFailed);
        }

        ::oneapi::fpk::lapack::potrf(_queue, uplomkl, n, a_sycl_buff, lda, scratchpad, scratchpad.get_count());

        _queue.wait();
        /** TODO: Check info buffer for containing errors. Now it is not supported.:
         *  https://software.intel.com/en-us/oneapi-mkl-dpcpp-developer-reference-potrf
        */
        return status;
    }

    services::Status operator()(const math::UpLo uplo, const size_t n, services::Buffer<algorithmFPType> & a, const size_t lda,
                                const std::int64_t scratchpadSize)
    {
        cl::sycl::buffer<algorithmFPType, 1> scratchpad_buffer { cl::sycl::range<1>(scratchpadSize) };
        return this->operator()(uplo, n, a, lda, scratchpad_buffer);
    }

    services::Status operator()(const math::UpLo uplo, const size_t n, services::Buffer<algorithmFPType> & a, const size_t lda)
    {
        const ::oneapi::fpk::uplo uplomkl        = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;
        const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrf_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, lda);
        return this->operator()(uplo, n, a, lda, minimalScratchpadSize);
    }

private:
    cl::sycl::queue & _queue;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLPOTRS></a>
 *  \brief Adapter for MKL POTRS routine
 */
template <typename algorithmFPType>
struct MKLPotrs
{
    MKLPotrs(cl::sycl::queue & queue) : _queue(queue) {}

    services::Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, services::Buffer<algorithmFPType> & a, const size_t lda,
                                services::Buffer<algorithmFPType> & b, const size_t ldb, cl::sycl::buffer<algorithmFPType, 1> & scratchpad)
    {
        services::Status status;
        const ::oneapi::fpk::uplo uplomkl                = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;
        cl::sycl::buffer<algorithmFPType, 1> a_sycl_buff = a.toSycl();
        cl::sycl::buffer<algorithmFPType, 1> b_sycl_buff = b.toSycl();

        {
            using namespace daal::services;
            const std::int64_t minimalScratchpadSize =
                ::oneapi::fpk::lapack::potrs_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, ny, lda, ldb);
            if (scratchpad.get_count() < minimalScratchpadSize) return Status(ErrorID::ErrorMemoryAllocationFailed);
        }

        ::oneapi::fpk::lapack::potrs(_queue, uplomkl, n, ny, a_sycl_buff, lda, b_sycl_buff, ldb, scratchpad, scratchpad.get_count());

        _queue.wait();
        /** TODO: Check info buffer for containing errors. Now it is not supported.:
         *  https://software.intel.com/en-us/oneapi-mkl-dpcpp-developer-reference-potrs
        */
        return status;
    }

    services::Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, services::Buffer<algorithmFPType> & a, const size_t lda,
                                services::Buffer<algorithmFPType> & b, const size_t ldb, const std::int64_t scratchpadSize)
    {
        cl::sycl::buffer<algorithmFPType, 1> scratchpad_buffer { cl::sycl::range<1>(scratchpadSize) };
        return this->operator()(uplo, n, ny, a, lda, b, ldb, scratchpad_buffer);
    }

    services::Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, services::Buffer<algorithmFPType> & a, const size_t lda,
                                services::Buffer<algorithmFPType> & b, const size_t ldb)
    {
        const ::oneapi::fpk::uplo uplomkl        = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;
        const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrs_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, ny, lda, ldb);
        return this->operator()(uplo, n, ny, a, lda, b, ldb, minimalScratchpadSize);
    }

private:
    cl::sycl::queue & _queue;
};

/** @} */

} // namespace interface1

using interface1::MKLPotrf;
using interface1::MKLPotrs;

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif

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

#include "services/internal/buffer.h"
#include "mkl_dal_sycl.hpp"

namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
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

    services::Status operator()(const math::UpLo uplo, const size_t n, services::internal::Buffer<algorithmFPType> & a, const size_t lda)
    {
        const ::oneapi::fpk::uplo uplomkl        = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;
        const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrf_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, lda);
        return this->operator()(uplo, n, a, lda, minimalScratchpadSize);
    }

private:
    services::Status operator()(const math::UpLo uplo, const size_t n, services::internal::Buffer<algorithmFPType> & a, const size_t lda,
                                const std::int64_t scratchpadSize)
    {
        using namespace daal::services;

        services::Status status;
        const ::oneapi::fpk::uplo uplomkl = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;

        if (a.isUSMBacked())
        {
            auto a_ptr = a.toUSM().get();
            auto scratchpad = cl::sycl::malloc<algorithmFPType>(scratchpadSize, _queue, cl::sycl::usm::alloc::shared);

            if (scratchpad == nullptr) return Status(ErrorID::ErrorMemoryAllocationFailed);

            ::oneapi::fpk::lapack::potrf(_queue, uplomkl, n, a_ptr, lda, scratchpad, scratchpadSize);
            cl::sycl::free(scratchpad, _queue);
        }
        else
        {
            auto a_sycl_buff = a.toSycl();
            cl::sycl::buffer<algorithmFPType> scratchpad { cl::sycl::range<1>(scratchpadSize) };

            if (scratchpad.get_count() < scratchpadSize) return Status(ErrorID::ErrorMemoryAllocationFailed);

            ::oneapi::fpk::lapack::potrf(_queue, uplomkl, n, a_sycl_buff, lda, scratchpad, scratchpadSize);
        }
        _queue.wait();
        /** TODO: Check info buffer for containing errors. Now it is not supported.:
         *  https://software.intel.com/en-us/oneapi-mkl-dpcpp-developer-reference-potrf
        */
        return status;
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

    services::Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, services::internal::Buffer<algorithmFPType> & a,
                                const size_t lda, services::internal::Buffer<algorithmFPType> & b, const size_t ldb)
    {
        const ::oneapi::fpk::uplo uplomkl        = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;
        const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrs_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, ny, lda, ldb);
        return this->operator()(uplo, n, ny, a, lda, b, ldb, minimalScratchpadSize);
    }

private:

    services::Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, services::internal::Buffer<algorithmFPType> & a,
                                const size_t lda, services::internal::Buffer<algorithmFPType> & b, const size_t ldb,
                                const std::int64_t scratchpadSize)
    {
        using namespace daal::services;

        services::Status status;
        const ::oneapi::fpk::uplo uplomkl = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;

        if (a.isUSMBacked())
        {
            auto a_ptr = a.toUSM().get();
            auto b_ptr = b.toUSM().get();
            auto scratchpad = cl::sycl::malloc<algorithmFPType>(scratchpadSize, _queue, cl::sycl::usm::alloc::shared);

            if (scratchpad == nullptr) return Status(ErrorID::ErrorMemoryAllocationFailed);

            ::oneapi::fpk::lapack::potrs(_queue, uplomkl, n, ny, a_ptr, lda, b_ptr, ldb, scratchpad, scratchpadSize);
            cl::sycl::free(scratchpad, _queue);
        }
        else
        {
            auto a_sycl_buff = a.toSycl();
            auto b_sycl_buff = b.toSycl();

            cl::sycl::buffer<algorithmFPType> scratchpad { cl::sycl::range<1>(scratchpadSize) };
            if (scratchpad.get_count() < scratchpadSize) return Status(ErrorID::ErrorMemoryAllocationFailed);

            ::oneapi::fpk::lapack::potrs(_queue, uplomkl, n, ny, a_sycl_buff, lda, b_sycl_buff, ldb, scratchpad, scratchpadSize);
        }

        _queue.wait();
        /** TODO: Check info buffer for containing errors. Now it is not supported.:
         *  https://software.intel.com/en-us/oneapi-mkl-dpcpp-developer-reference-potrs
        */
        return status;
    }

private:
    cl::sycl::queue & _queue;
};

/** @} */

} // namespace interface1

using interface1::MKLPotrf;
using interface1::MKLPotrs;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif

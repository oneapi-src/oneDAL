/* file: mkl_lapack.h */
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
//  Wrappers for LAPACK functions.
//--
*/

#ifndef __ONEAPI_INTERNAL_MKL_LAPACK_H__
#define __ONEAPI_INTERNAL_MKL_LAPACK_H__

#include "services/internal/buffer.h"
#include "services/internal/sycl/math/mkl_dal_utils.h"

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
 *  \brief Adapter for Intel(R) MKL POTRF routine
 */
template <typename algorithmFPType>
struct MKLPotrf
{
    MKLPotrf(::sycl::queue & queue) : _queue(queue) {}

    Status operator()(const math::UpLo uplo, const size_t n, Buffer<algorithmFPType> & a, const size_t lda)
    {
        const auto uplomkl                       = to_fpk_uplo(uplo);
        const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrf_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, lda);
        return this->operator()(uplo, n, a, lda, minimalScratchpadSize);
    }

private:
    Status operator()(const math::UpLo uplo, const size_t n, Buffer<algorithmFPType> & a, const size_t lda, const std::int64_t scratchpadSize)
    {
        using namespace daal::services;

        Status status;
        const auto uplomkl = to_fpk_uplo(uplo);

#ifdef DAAL_SYCL_INTERFACE_USM
        auto a_usm = a.toUSM(_queue, data_management::readWrite, status);
        DAAL_CHECK_STATUS_VAR(status);

        algorithmFPType * scratchpad = nullptr;
        if (scratchpadSize > 0)
        {
            scratchpad = ::sycl::malloc_device<algorithmFPType>(scratchpadSize, _queue);
            if (scratchpad == nullptr) return ErrorMemoryAllocationFailed;
        }

        status |= catchSyclExceptions([&]() mutable {
            ::oneapi::fpk::lapack::potrf(_queue, uplomkl, n, a_usm.get(), lda, scratchpad, scratchpadSize);
            _queue.wait_and_throw();
        });

        if (scratchpadSize > 0) ::sycl::free(scratchpad, _queue);

        scratchpad = nullptr;
#else
        static_assert(false, "USM support required");
#endif
        return status;
    }

private:
    ::sycl::queue & _queue;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLPOTRS></a>
 *  \brief Adapter for Intel(R) MKL POTRS routine
 */
template <typename algorithmFPType>
struct MKLPotrs
{
    MKLPotrs(::sycl::queue & queue) : _queue(queue) {}

    Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, Buffer<algorithmFPType> & a, const size_t lda,
                      Buffer<algorithmFPType> & b, const size_t ldb)
    {
        const auto uplomkl                       = to_fpk_uplo(uplo);
        const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrs_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, ny, lda, ldb);
        return this->operator()(uplo, n, ny, a, lda, b, ldb, minimalScratchpadSize);
    }

private:
    Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, Buffer<algorithmFPType> & a, const size_t lda,
                      Buffer<algorithmFPType> & b, const size_t ldb, const std::int64_t scratchpadSize)
    {
        using namespace daal::services;

        services::Status status;
        const auto uplomkl = to_fpk_uplo(uplo);

#ifdef DAAL_SYCL_INTERFACE_USM
        auto a_usm = a.toUSM(_queue, data_management::readWrite, status);
        DAAL_CHECK_STATUS_VAR(status);

        auto b_usm = b.toUSM(_queue, data_management::readWrite, status);
        DAAL_CHECK_STATUS_VAR(status);

        algorithmFPType * scratchpad = nullptr;
        if (scratchpadSize > 0)
        {
            scratchpad = ::sycl::malloc_device<algorithmFPType>(scratchpadSize, _queue);
            if (scratchpad == nullptr) return ErrorMemoryAllocationFailed;
        }

        status |= catchSyclExceptions([&]() mutable {
            ::oneapi::fpk::lapack::potrs(_queue, uplomkl, n, ny, a_usm.get(), lda, b_usm.get(), ldb, scratchpad, scratchpadSize);
            _queue.wait_and_throw();
        });

        if (scratchpadSize > 0) ::sycl::free(scratchpad, _queue);

        scratchpad = nullptr;
#else
        static_assert(false, "USM support required");
#endif
        return status;
    }

private:
    ::sycl::queue & _queue;
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

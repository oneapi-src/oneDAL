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
#include "services/internal/sycl/math/mkl_dal.h"

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

    Status operator()(const math::UpLo uplo, const size_t n, Buffer<algorithmFPType> & a, const size_t lda)
    {
        const ::oneapi::fpk::uplo uplomkl        = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;
        const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrf_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, lda);
        return this->operator()(uplo, n, a, lda, minimalScratchpadSize);
    }

private:
    Status operator()(const math::UpLo uplo, const size_t n, Buffer<algorithmFPType> & a, const size_t lda, const std::int64_t scratchpadSize)
    {
        using namespace daal::services;

        Status status;
        const ::oneapi::fpk::uplo uplomkl = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;

#ifdef DAAL_SYCL_INTERFACE_USM
        if (a.isUSMBacked())
        {
            auto a_usm      = a.toUSM(status);
            DAAL_CHECK_STATUS_VAR(status);

            auto scratchpad = cl::sycl::malloc<algorithmFPType>(scratchpadSize, _queue, cl::sycl::usm::alloc::shared);
            if (scratchpad == nullptr) return ErrorMemoryAllocationFailed;

            status |= catchSyclExceptions([&]() mutable {
                ::oneapi::fpk::lapack::potrf(_queue, uplomkl, n, a_usm.get(), lda, scratchpad, scratchpadSize);
                _queue.wait_and_throw();
            });

            cl::sycl::free(scratchpad, _queue);
        }
        else
#endif
        {
            auto a_sycl_buff = a.toSycl(status);
            DAAL_CHECK_STATUS_VAR(status);

            status |= catchSyclExceptions([&]() mutable {
                cl::sycl::buffer<algorithmFPType, 1> scratchpad { cl::sycl::range<1>(scratchpadSize) };

                const size_t minimalScratchpadSize = size_t(::oneapi::fpk::lapack::potrf_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, lda));
                _queue.wait_and_throw();
                if (scratchpad.get_count() < minimalScratchpadSize) return ErrorMemoryAllocationFailed;

                ::oneapi::fpk::lapack::potrf(_queue, uplomkl, n, a_sycl_buff, lda, scratchpad, scratchpad.get_count());
                _queue.wait_and_throw();
            });
        }

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

    Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, Buffer<algorithmFPType> & a, const size_t lda,
                      Buffer<algorithmFPType> & b, const size_t ldb)
    {
        const ::oneapi::fpk::uplo uplomkl        = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;
        const std::int64_t minimalScratchpadSize = ::oneapi::fpk::lapack::potrs_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, ny, lda, ldb);
        return this->operator()(uplo, n, ny, a, lda, b, ldb, minimalScratchpadSize);
    }

private:
    Status operator()(const math::UpLo uplo, const size_t n, const size_t ny, Buffer<algorithmFPType> & a, const size_t lda,
                      Buffer<algorithmFPType> & b, const size_t ldb, const std::int64_t scratchpadSize)
    {
        using namespace daal::services;

        services::Status status;
        const ::oneapi::fpk::uplo uplomkl = uplo == math::UpLo::Upper ? ::oneapi::fpk::uplo::upper : ::oneapi::fpk::uplo::lower;

#ifdef DAAL_SYCL_INTERFACE_USM
        if (a.isUSMBacked())
        {
            auto a_usm      = a.toUSM(status);
            DAAL_CHECK_STATUS_VAR(status);

            auto b_usm      = b.toUSM(status);
            DAAL_CHECK_STATUS_VAR(status);

            auto scratchpad = cl::sycl::malloc<algorithmFPType>(scratchpadSize, _queue, cl::sycl::usm::alloc::shared);
            if (scratchpad == nullptr) return ErrorMemoryAllocationFailed;

            status |= catchSyclExceptions([&]() mutable {
                ::oneapi::fpk::lapack::potrs(_queue, uplomkl, n, ny, a_usm.get(), lda, b_usm.get(), ldb, scratchpad, scratchpadSize);
                _queue.wait_and_throw();
            });

            cl::sycl::free(scratchpad, _queue);
        }
        else
#endif
        {
            cl::sycl::buffer<algorithmFPType, 1> a_sycl_buff = a.toSycl(status);
            DAAL_CHECK_STATUS_VAR(status);

            cl::sycl::buffer<algorithmFPType, 1> b_sycl_buff = b.toSycl(status);
            DAAL_CHECK_STATUS_VAR(status);

            status |= catchSyclExceptions([&]() mutable {
                cl::sycl::buffer<algorithmFPType, 1> scratchpad { cl::sycl::range<1>(scratchpadSize) };

                const size_t minimalScratchpadSize =
                    size_t(::oneapi::fpk::lapack::potrs_scratchpad_size<algorithmFPType>(_queue, uplomkl, n, ny, lda, ldb));
                _queue.wait_and_throw();
                if (scratchpad.get_count() < minimalScratchpadSize) return ErrorMemoryAllocationFailed;

                :oneapi::fpk::lapack::potrs(_queue, uplomkl, n, ny, a_sycl_buff, lda, b_sycl_buff, ldb, scratchpad, scratchpad.get_count());
                _queue.wait_and_throw();
            });
        }

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

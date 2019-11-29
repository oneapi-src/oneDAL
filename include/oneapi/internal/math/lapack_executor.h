/* file: lapack_executor.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#ifndef __ONEAPI_INTERNAL_MATH_LAPACK_EXECUTOR_H__
#define __ONEAPI_INTERNAL_MATH_LAPACK_EXECUTOR_H__

/*
//++
//  Executors for LAPACK functions
//--
*/

#ifdef ONEAPI_DAAL_USE_MKL_GPU_FUNC
    #if !(defined(__clang__))
        #undef ONEAPI_DAAL_USE_MKL_GPU_FUNC
    #endif
#endif

#ifdef ONEAPI_DAAL_USE_MKL_GPU_FUNC
    #include "mkl_lapack.h"
#endif

#include "oneapi/internal/types_utils.h"
#include "services/internal/error_handling_helpers.h"
#include "reference_lapack.h"
#include "types.h"

#include <CL/sycl.hpp>

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
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__POTRFEXECUTOR"></a>
 *  \brief Adapter for POTRF routine
 */
class PotrfExecutor
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        const math::UpLo uplo;
        const size_t n;
        UniversalBuffer & a_buffer;
        const size_t lda;
        services::Status * status;

        explicit Execute(cl::sycl::queue & queue, const math::UpLo uplo, const size_t n, UniversalBuffer & a_buffer, const size_t lda,
                         services::Status * status)
            : queue(queue), uplo(uplo), n(n), a_buffer(a_buffer), lda(lda), status(status)
        {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            auto a_buffer_t = a_buffer.template get<T>();

#ifdef ONEAPI_DAAL_USE_MKL_GPU_FUNC
            MKLPotrf<T> functor(queue);
#else
            ReferencePotrf<T> functor;
#endif

            services::internal::tryAssignStatus(status, functor(uplo, n, a_buffer_t, lda));
        }
    };

public:
    static void run(cl::sycl::queue & queue, const math::UpLo uplo, const size_t n, UniversalBuffer & a_buffer, const size_t lda,
                    services::Status * status)
    {
        Execute op(queue, uplo, n, a_buffer, lda, status);
        TypeDispatcher::floatDispatch(a_buffer.type(), op);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__POTRSEXECUTOR"></a>
 *  \brief Adapter for POTRS routine
 */
class PotrsExecutor
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        const math::UpLo uplo;
        const size_t n;
        const size_t ny;
        UniversalBuffer & a_buffer;
        const size_t lda;
        UniversalBuffer & b_buffer;
        const size_t ldb;
        services::Status * status;

        explicit Execute(cl::sycl::queue & queue, const math::UpLo uplo, const size_t n, const size_t ny, UniversalBuffer & a_buffer,
                         const size_t lda, UniversalBuffer & b_buffer, const size_t ldb, services::Status * status)
            : queue(queue), uplo(uplo), ny(ny), n(n), a_buffer(a_buffer), lda(lda), b_buffer(b_buffer), ldb(ldb), status(status)
        {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            auto a_buffer_t = a_buffer.template get<T>();
            auto b_buffer_t = b_buffer.template get<T>();

#ifdef ONEAPI_DAAL_USE_MKL_GPU_FUNC
            MKLPotrs<T> functor(queue);
#else
            ReferencePotrs<T> functor;
#endif

            services::internal::tryAssignStatus(status, functor(uplo, n, ny, a_buffer_t, lda, b_buffer_t, ldb));
        }
    };

public:
    static void run(cl::sycl::queue & queue, const math::UpLo uplo, const size_t n, const size_t ny, UniversalBuffer & a_buffer, const size_t lda,
                    UniversalBuffer & b_buffer, const size_t ldb, services::Status * status)
    {
        Execute op(queue, uplo, n, ny, a_buffer, lda, b_buffer, ldb, status);
        TypeDispatcher::floatDispatch(a_buffer.type(), op);
    }
};

/** @} */
} // namespace interface1

using interface1::PotrfExecutor;
using interface1::PotrsExecutor;

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif

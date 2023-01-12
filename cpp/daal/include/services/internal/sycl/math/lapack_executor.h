/* file: lapack_executor.h */
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

#ifndef __ONEAPI_INTERNAL_MATH_LAPACK_EXECUTOR_H__
#define __ONEAPI_INTERNAL_MATH_LAPACK_EXECUTOR_H__

/*
//++
//  Executors for LAPACK functions
//--
*/

#include <sycl/sycl.hpp>

#if (!defined(ONEAPI_DAAL_NO_MKL_GPU_FUNC) && defined(__SYCL_COMPILER_VERSION))
    #include "services/internal/sycl/math/mkl_lapack.h"
#endif

#include "services/internal/sycl/types_utils.h"
#include "services/internal/sycl/math/types.h"
#include "services/internal/sycl/math/reference_lapack.h"

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
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__POTRFEXECUTOR"></a>
 *  \brief Adapter for POTRF routine
 */
class PotrfExecutor
{
private:
    struct Execute
    {
        ::sycl::queue & queue;
        const math::UpLo uplo;
        const size_t n;
        UniversalBuffer & a_buffer;
        const size_t lda;
        explicit Execute(::sycl::queue & queue, const math::UpLo uplo, const size_t n, UniversalBuffer & a_buffer, const size_t lda)
            : queue(queue), uplo(uplo), n(n), a_buffer(a_buffer), lda(lda)
        {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER(a_buffer, T, n * lda);

            auto a_buffer_t = a_buffer.template get<T>();

#ifdef ONEAPI_DAAL_NO_MKL_GPU_FUNC
            ReferencePotrf<T> functor;
#else
            MKLPotrf<T> functor(queue);
#endif
            status |= functor(uplo, n, a_buffer_t, lda);
        }
    };

public:
    static void run(::sycl::queue & queue, const math::UpLo uplo, const size_t n, UniversalBuffer & a_buffer, const size_t lda, Status & status)
    {
        DAAL_ASSERT(!a_buffer.empty());

        Execute op(queue, uplo, n, a_buffer, lda);
        TypeDispatcher::floatDispatch(a_buffer.type(), op, status);
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
        ::sycl::queue & queue;
        const math::UpLo uplo;
        const size_t n;
        const size_t ny;
        UniversalBuffer & a_buffer;
        const size_t lda;
        UniversalBuffer & b_buffer;
        const size_t ldb;

        explicit Execute(::sycl::queue & queue, const math::UpLo uplo, const size_t n, const size_t ny, UniversalBuffer & a_buffer, const size_t lda,
                         UniversalBuffer & b_buffer, const size_t ldb)
            : queue(queue), uplo(uplo), n(n), ny(ny), a_buffer(a_buffer), lda(lda), b_buffer(b_buffer), ldb(ldb)
        {}

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER(a_buffer, T, n * lda);
            DAAL_ASSERT_UNIVERSAL_BUFFER(b_buffer, T, ny * ldb);

            auto a_buffer_t = a_buffer.template get<T>();
            auto b_buffer_t = b_buffer.template get<T>();

#ifdef ONEAPI_DAAL_NO_MKL_GPU_FUNC
            ReferencePotrs<T> functor;
#else
            MKLPotrs<T> functor(queue);
#endif

            status |= functor(uplo, n, ny, a_buffer_t, lda, b_buffer_t, ldb);
        }
    };

public:
    static void run(::sycl::queue & queue, const math::UpLo uplo, const size_t n, const size_t ny, UniversalBuffer & a_buffer, const size_t lda,
                    UniversalBuffer & b_buffer, const size_t ldb, Status & status)
    {
        DAAL_ASSERT(!a_buffer.empty());
        DAAL_ASSERT(!b_buffer.empty());
        DAAL_ASSERT(a_buffer.type() == b_buffer.type());

        Execute op(queue, uplo, n, ny, a_buffer, lda, b_buffer, ldb);
        TypeDispatcher::floatDispatch(a_buffer.type(), op, status);
    }
};

/** @} */
} // namespace interface1

using interface1::PotrfExecutor;
using interface1::PotrsExecutor;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif

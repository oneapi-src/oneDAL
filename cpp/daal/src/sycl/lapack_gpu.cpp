/* file: lapack_gpu.cpp */
/*******************************************************************************
* Copyright 2015 Intel Corporation
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

#include "services/internal/sycl/math/reference_lapack.h"
#include "src/externals/service_lapack.h"
#include "services/error_handling.h"
#include "src/sycl/blas_gpu.h"
#include "src/sycl/cl_kernels/kernel_blas.cl"

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
using namespace daal::internal;

template <typename algorithmFPType>
services::Status ReferencePotrf<algorithmFPType>::operator()(const math::UpLo uplo, const size_t n,
                                                             services::internal::Buffer<algorithmFPType> & a_buffer, const size_t lda)
{
    services::Status status;

    char up = uplo == math::UpLo::Upper ? 'U' : 'L';

    DAAL_INT info;

    DAAL_INT nInt   = static_cast<DAAL_INT>(n);
    DAAL_INT ldaInt = static_cast<DAAL_INT>(lda);

    services::SharedPtr<algorithmFPType> aPtr = a_buffer.toHost(data_management::ReadWriteMode::readWrite, status);
    DAAL_CHECK_STATUS_VAR(status);

    LapackAutoDispatch<algorithmFPType>::xpotrf(&up, &nInt, aPtr.get(), &ldaInt, &info);

    DAAL_CHECK(info == 0, services::ErrorID::ErrorNormEqSystemSolutionFailed);
    return status;
}

template <typename algorithmFPType>
services::Status ReferencePotrs<algorithmFPType>::operator()(const math::UpLo uplo, const size_t n, const size_t ny,
                                                             services::internal::Buffer<algorithmFPType> & a_buffer, const size_t lda,
                                                             services::internal::Buffer<algorithmFPType> & b_buffer, const size_t ldb)
{
    services::Status status;

    char up = uplo == math::UpLo::Upper ? 'U' : 'L';

    DAAL_INT info;

    DAAL_INT nInt   = static_cast<DAAL_INT>(n);
    DAAL_INT nyInt  = static_cast<DAAL_INT>(ny);
    DAAL_INT ldaInt = static_cast<DAAL_INT>(lda);
    DAAL_INT ldbInt = static_cast<DAAL_INT>(ldb);

    services::SharedPtr<algorithmFPType> aPtr = a_buffer.toHost(data_management::ReadWriteMode::readWrite, status);
    DAAL_CHECK_STATUS_VAR(status);

    services::SharedPtr<algorithmFPType> bPtr = b_buffer.toHost(data_management::ReadWriteMode::readWrite, status);
    DAAL_CHECK_STATUS_VAR(status);

    LapackAutoDispatch<algorithmFPType>::xpotrs(&up, &nInt, &nyInt, aPtr.get(), &ldaInt, bPtr.get(), &ldbInt, &info);

    DAAL_CHECK(info == 0, services::ErrorID::ErrorNormEqSystemSolutionFailed);
    return status;
}

template class ReferencePotrf<float>;
template class ReferencePotrf<double>;

template class ReferencePotrs<float>;
template class ReferencePotrs<double>;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

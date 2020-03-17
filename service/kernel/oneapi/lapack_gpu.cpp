/* file: lapack_gpu.cpp */
/*******************************************************************************
* Copyright 2015-2020 Intel Corporation
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

#include "oneapi/internal/math/reference_lapack.h"
#include "externals/service_lapack.h"
#include "services/error_handling.h"
#include "service/kernel/oneapi/blas_gpu.h"
#include "service/kernel/oneapi/cl_kernels/kernel_blas.cl"
#include "service/kernel/service_arrays.h"
#include "service/kernel/service_data_utils.h"

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
using namespace daal::internal;

template <typename algorithmFPType>
services::Status ReferencePotrf<algorithmFPType>::operator()(const math::UpLo uplo, const size_t n, services::Buffer<algorithmFPType> & a_buffer,
                                                             const size_t lda)
{
    services::Status status;

    char up = uplo == math::UpLo::Upper ? 'U' : 'L';

    DAAL_INT info;

    DAAL_INT nInt   = static_cast<DAAL_INT>(n);
    DAAL_INT ldaInt = static_cast<DAAL_INT>(lda);

    services::SharedPtr<algorithmFPType> aPtr = a_buffer.toHost(data_management::ReadWriteMode::readWrite);

    LapackAutoDispatch<algorithmFPType>::xpotrf(&up, &nInt, aPtr.get(), &ldaInt, &info);

    DAAL_CHECK(info == 0, services::ErrorID::ErrorNormEqSystemSolutionFailed);
    return status;
}

template <typename algorithmFPType>
services::Status ReferencePotrs<algorithmFPType>::operator()(const math::UpLo uplo, const size_t n, const size_t ny,
                                                             services::Buffer<algorithmFPType> & a_buffer, const size_t lda,
                                                             services::Buffer<algorithmFPType> & b_buffer, const size_t ldb)
{
    services::Status status;

    char up = uplo == math::UpLo::Upper ? 'U' : 'L';

    DAAL_INT info;

    DAAL_INT nInt   = static_cast<DAAL_INT>(n);
    DAAL_INT nyInt  = static_cast<DAAL_INT>(ny);
    DAAL_INT ldaInt = static_cast<DAAL_INT>(lda);
    DAAL_INT ldbInt = static_cast<DAAL_INT>(ldb);

    services::SharedPtr<algorithmFPType> aPtr = a_buffer.toHost(data_management::ReadWriteMode::readWrite);
    services::SharedPtr<algorithmFPType> bPtr = b_buffer.toHost(data_management::ReadWriteMode::readWrite);

    LapackAutoDispatch<algorithmFPType>::xpotrs(&up, &nInt, &nyInt, aPtr.get(), &ldaInt, bPtr.get(), &ldbInt, &info);

    DAAL_CHECK(info == 0, services::ErrorID::ErrorNormEqSystemSolutionFailed);
    return status;
}

template <typename algorithmFPType>
services::Status ReferenceXsyevd<algorithmFPType>::operator()(const math::Job jobz, const math::UpLo uplo, const int64_t n,
                                                              services::Buffer<algorithmFPType> & a, const int64_t lda,
                                                              services::Buffer<algorithmFPType> & w, services::Buffer<algorithmFPType> & work,
                                                              const int64_t lwork, services::Buffer<int64_t> & iwork, const int64_t liwork)
{
    services::Status status;

    char up;
    switch (uplo)
    {
    case math::UpLo::Upper: up = 'U'; break;
    case math::UpLo::Lower: up = 'L'; break;

    default: status.add(services::UnknownError); break;
    }

    char job;
    switch (jobz)
    {
    case math::Job::novec: job = 'N'; break;
    case math::Job::vec: job = 'V'; break;
    case math::Job::updatevec: job = 'U'; break;
    case math::Job::allvec: job = 'A'; break;
    case math::Job::somevec: job = 'S'; break;
    case math::Job::overwritevec: job = 'O'; break;

    default: status.add(services::UnknownError); break;
    }
    DAAL_CHECK_STATUS_VAR(status);

    if (n > services::internal::MaxVal<DAAL_INT>::get() || lda > services::internal::MaxVal<DAAL_INT>::get()
        || lwork > services::internal::MaxVal<DAAL_INT>::get() || liwork > services::internal::MaxVal<DAAL_INT>::get())
    {
        status.add(services::ErrorID::ErrorIncorrectSizeOfArray);
        return status;
    }
    DAAL_INT nInt      = static_cast<DAAL_INT>(n);
    DAAL_INT ldaInt    = static_cast<DAAL_INT>(lda);
    DAAL_INT lworkInt  = static_cast<DAAL_INT>(lwork);
    DAAL_INT liworkInt = static_cast<DAAL_INT>(liwork);

    services::SharedPtr<algorithmFPType> aPtr    = a.toHost(data_management::ReadWriteMode::readWrite);
    services::SharedPtr<algorithmFPType> wPtr    = w.toHost(data_management::ReadWriteMode::readWrite);
    services::SharedPtr<algorithmFPType> workPtr = work.toHost(data_management::ReadWriteMode::readWrite);
    services::SharedPtr<int64_t> iworkSharedPtr  = iwork.toHost(data_management::ReadWriteMode::readWrite);

    DAAL_INT info;

    services::internal::TArray<DAAL_INT, CpuType::sse2> iWorkTmp(iwork.size());

    LapackAutoDispatch<algorithmFPType>::xsyevd(&job, &up, &nInt, aPtr.get(), &ldaInt, wPtr.get(), workPtr.get(), &lworkInt, iWorkTmp.get(),
                                                &liworkInt, &info);
    DAAL_CHECK(info == 0, services::ErrorID::ErrorNormEqSystemSolutionFailed);

    int64_t * iworkPtr = iworkSharedPtr.get();
    for (size_t i = 0; i < iwork.size(); ++i)
    {
        iworkPtr[i] = static_cast<int64_t>(iWorkTmp[i]);
    }

    return status;
}

template class ReferencePotrf<float>;
template class ReferencePotrf<double>;

template class ReferencePotrs<float>;
template class ReferencePotrs<double>;

template class ReferenceXsyevd<float>;
template class ReferenceXsyevd<double>;

} // namespace interface1
} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

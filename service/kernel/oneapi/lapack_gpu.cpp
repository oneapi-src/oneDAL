/* file: lapack_gpu.cpp */
/*******************************************************************************
* Copyright 2015-2019 Intel Corporation
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
#include "service_lapack.h"
#include "daal_kernel_defines.h"
#include "error_handling.h"
#include "blas_gpu.h"
#include "cl_kernels/kernel_blas.cl"

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

#define CALL_CPU_FUNCTION(fptype, cpuId, className, funcName, ...)                                                                                         \
    switch (cpuId) {                                                                                                                                       \
    DAAL_KERNEL_SSSE3_ONLY_CODE(case daal::CpuType::ssse3: className<fptype, daal::CpuType::ssse3>::funcName(__VA_ARGS__); break;)                         \
    DAAL_KERNEL_SSE42_ONLY_CODE(case daal::CpuType::sse42: className<fptype, daal::CpuType::sse42>::funcName(__VA_ARGS__); break;)                         \
    DAAL_KERNEL_AVX_ONLY_CODE(case daal::CpuType::avx:   className<fptype, daal::CpuType::avx>::funcName(__VA_ARGS__); break;)                             \
    DAAL_KERNEL_AVX2_ONLY_CODE(case daal::CpuType::avx2:  className<fptype, daal::CpuType::avx2>::funcName(__VA_ARGS__); break;)                           \
    DAAL_KERNEL_AVX512_ONLY_CODE(case daal::CpuType::avx512:  className<fptype, daal::CpuType::avx512>::funcName(__VA_ARGS__); break;)                     \
    DAAL_KERNEL_AVX512_MIC_ONLY_CODE(case daal::CpuType::avx512_mic_e1: className<fptype, daal::CpuType::avx512_mic_e1>::funcName(__VA_ARGS__); break;)    \
    default: className<fptype, daal::CpuType::sse2>::funcName(__VA_ARGS__); break;                                                                         \
    }


template<typename algorithmFPType>
services::Status ReferencePotrf<algorithmFPType>::operator()(const math::UpLo uplo, const size_t n, services::Buffer<algorithmFPType> & a_buffer,
                                                             const size_t lda)
{
    services::Status status;
    const daal::CpuType cpuId = static_cast<daal::CpuType>(services::Environment::getInstance()->getCpuId());

    char up = uplo == math::UpLo::Upper ? 'U' : 'L';

    DAAL_INT info;

    DAAL_INT nInt   = static_cast<DAAL_INT>(n);
    DAAL_INT ldaInt = static_cast<DAAL_INT>(lda);

    services::SharedPtr<algorithmFPType> aPtr = a_buffer.toHost(data_management::ReadWriteMode::readWrite);

    CALL_CPU_FUNCTION(algorithmFPType, cpuId, Lapack, xpotrf, &up, &nInt, aPtr.get(), &ldaInt, &info);

    DAAL_CHECK(info == 0, services::ErrorID::ErrorNormEqSystemSolutionFailed);
    return status;
}

template <typename algorithmFPType>
services::Status ReferencePotrs<algorithmFPType>::operator()(const math::UpLo uplo, const size_t n, const size_t ny,
                                                             services::Buffer<algorithmFPType> & a_buffer, const size_t lda,
                                                             services::Buffer<algorithmFPType> & b_buffer, const size_t ldb)
{
    services::Status status;
    const daal::CpuType cpuId = static_cast<daal::CpuType>(services::Environment::getInstance()->getCpuId());

    char up = uplo == math::UpLo::Upper ? 'U' : 'L';

    DAAL_INT info;

    DAAL_INT nInt   = static_cast<DAAL_INT>(n);
    DAAL_INT nyInt  = static_cast<DAAL_INT>(ny);
    DAAL_INT ldaInt = static_cast<DAAL_INT>(lda);
    DAAL_INT ldbInt = static_cast<DAAL_INT>(ldb);

    services::SharedPtr<algorithmFPType> aPtr = a_buffer.toHost(data_management::ReadWriteMode::readWrite);
    services::SharedPtr<algorithmFPType> bPtr = b_buffer.toHost(data_management::ReadWriteMode::readWrite);

    CALL_CPU_FUNCTION(algorithmFPType, cpuId, Lapack, xpotrs, &up, &nInt, &nyInt, aPtr.get(), &ldaInt, bPtr.get(), &ldbInt, &info);

    DAAL_CHECK(info == 0, services::ErrorID::ErrorNormEqSystemSolutionFailed);
    return status;
}

template class ReferencePotrf<float>;
template class ReferencePotrf<double>;

template class ReferencePotrs<float>;
template class ReferencePotrs<double>;

} // namespace interface1
} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

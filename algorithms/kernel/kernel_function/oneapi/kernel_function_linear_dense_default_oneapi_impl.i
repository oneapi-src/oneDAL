/* file: kernel_function_linear_dense_default_oneapi_impl.i */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Linear kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_DENSE_DEFAULT_ONEAPI_IMPL_I__
#define __KERNEL_FUNCTION_LINEAR_DENSE_DEFAULT_ONEAPI_IMPL_I__

#include "algorithms/kernel_function/kernel_function_types_linear.h"

#include "externals/service_blas.h"
#include "externals/service_stat.h"
#include "algorithms/threading/threading.h"
#include "algorithms/kernel/service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace internal
{
template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<defaultDense, algorithmFPType>::computeInternalVectorVector(const NumericTable & a1, const NumericTable & a2,
                                                                                                    NumericTable & r, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<defaultDense, algorithmFPType>::computeInternalMatrixVector(const NumericTable & a1, const NumericTable & a2,
                                                                                                    NumericTable & r, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<defaultDense, algorithmFPType>::computeInternalMatrixMatrix(const NumericTable & a1, const NumericTable * a2,
                                                                                                    NumericTable & r, const ParameterBase * par)
{
    services::Status status;

    auto & context    = oneapi::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    const size_t nVectors1 = a1.getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();

    const size_t nFeatures1 = a1->getNumberOfColumns();
    const size_t nFeatures2 = a2->getNumberOfColumns();
    DAAL_ASSERT(nFeatures1 == nFeatures2);

    const Parameter * linPar = static_cast<const Parameter *>(par);
    algorithmFPType alpha    = algorithmFPType(linPar->k);
    algorithmFPType beta     = algorithmFPType(linPar->b);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelLinearOneAPI.gemm);

        BlockDescriptor<algorithmFPType> a1BD;
        BlockDescriptor<algorithmFPType> a2BD;
        BlockDescriptor<algorithmFPType> rBD;

        // TODO: Need block GEMM to avoid copying

        DAAL_CHECK_STATUS(status, a1.getBlockOfRows(0, nVectors1, ReadWriteMode::readOnly, a1BD));
        DAAL_CHECK_STATUS(status, a2.getBlockOfRows(0, nVectors2, ReadWriteMode::readOnly, a2BD));

        DAAL_CHECK_STATUS(status, r.getBlockOfRows(0, nVectors1, ReadWriteMode::readWrite, rBD));

        const services::Buffer<algorithmFPType> a1Buf = a1BD.getBuffer();
        const services::Buffer<algorithmFPType> a2Buf = a2BD.getBuffer();

        services::Buffer<algorithmFPType> rBuf = rBD.getBuffer();

        if (beta != 0.0)
        {
            context.fill(rBuf, 1.0, &status);
        }

        status = BlasGpu<algorithmFPType>::xgemm(math::Layout::ColMajor, math::Transpose::Trans, math::Transpose::NoTrans, nVectors2, nVectors1,
                                                 nFeatures1, alpha, a2Buf, nFeatures1, 0, a1Buf, nFeatures1, 0, beta, rBuf, nVectors2, 0);
    }
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

} // namespace internal

} // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif

/* file: pca_transform_dense_default_batch_oneapi.h */
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
//  Common functions of PCA transformation on GPU
//--
*/

#ifndef __PCA_TRANSFORM_DENSE_DEFAULT_BATCH_ONEAPI_H__
#define __PCA_TRANSFORM_DENSE_DEFAULT_BATCH_ONEAPI_H__

#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "algorithms/pca/pca_types.h"
#include "service/kernel/oneapi/blas_gpu.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
namespace oneapi
{
namespace internal
{
template <typename algorithmFPType, transform::Method method>
class TransformKernelOneAPI : public Kernel
{
public:
    services::Status compute(data_management::NumericTable & data, data_management::NumericTable & eigenvectors,
                             data_management::NumericTable * pMeans, data_management::NumericTable * pVariances,
                             data_management::NumericTable * pEigenvalues, data_management::NumericTable & transformedData);

    void computeTransformedBlock(uint32_t numRows, uint32_t numFeatures, uint32_t numComponents, daal::oneapi::internal::UniversalBuffer & dataBlock,
                                 const services::Buffer<algorithmFPType> & eigenvectors, const services::Buffer<algorithmFPType> & resultBlock);

private:
    services::Status allocateBuffer(daal::oneapi::internal::ExecutionContextIface & context, daal::oneapi::internal::UniversalBuffer & returnBuffer,
                                    uint32_t bufferSize);

    services::Status copyBuffer(daal::oneapi::internal::ExecutionContextIface & context, daal::oneapi::internal::UniversalBuffer & returnBuffer,
                                data_management::NumericTable & data, const uint32_t nRows, const uint32_t nCols);

    services::Status buildKernel(daal::oneapi::internal::ExecutionContextIface & context, daal::oneapi::internal::ClKernelFactoryIface & factory);

    services::Status checkVariances(data_management::NumericTable & pVariances, uint32_t numRows);

    services::Status computeInvSigmas(daal::oneapi::internal::ExecutionContextIface & context, data_management::NumericTable * variances,
                                      const services::Buffer<algorithmFPType> & invSigmas, const uint32_t numFeatures);

    services::Status normalize(daal::oneapi::internal::ExecutionContextIface & context, daal::oneapi::internal::UniversalBuffer & copyBlock,
                               daal::oneapi::internal::UniversalBuffer & rawMeans, daal::oneapi::internal::UniversalBuffer & invSigmas,
                               unsigned char hasMeans, unsigned char hasInvSigmas, const uint32_t numFeatures, const uint32_t numVectors);

    services::Status whitening(daal::oneapi::internal::ExecutionContextIface & context, const services::Buffer<algorithmFPType> & transformedBlock,
                               daal::oneapi::internal::UniversalBuffer & invEigenvalues, const uint32_t numComponents, const uint32_t numVectors);

    services::Status initBuffers(daal::oneapi::internal::ExecutionContextIface & ctx, data_management::NumericTable & data,
                                 const uint32_t numFeatures, const uint32_t numComponents, const uint32_t numVectors);

private:
    const unsigned int maxWorkItemsPerGroup = 256;
    daal::oneapi::internal::UniversalBuffer invSigmas;
    daal::oneapi::internal::UniversalBuffer invEigenvalues;
    daal::oneapi::internal::UniversalBuffer rawMeans;
    daal::oneapi::internal::UniversalBuffer copyBlock;
};

} // namespace internal
} // namespace oneapi
} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif

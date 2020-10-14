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

#include "services/internal/sycl/types.h"
#include "services/internal/sycl/execution_context.h"
#include "algorithms/pca/pca_types.h"
#include "algorithms/pca/transform/pca_transform_types.h"
#include "src/sycl/blas_gpu.h"

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

    void computeTransformedBlock(uint32_t numRows, uint32_t numFeatures, uint32_t numComponents,
                                 daal::services::internal::sycl::UniversalBuffer & dataBlock,
                                 const services::internal::Buffer<algorithmFPType> & eigenvectors,
                                 const services::internal::Buffer<algorithmFPType> & resultBlock);

private:
    services::Status allocateBuffer(daal::services::internal::sycl::ExecutionContextIface & context,
                                    daal::services::internal::sycl::UniversalBuffer & returnBuffer, uint32_t bufferSize);

    services::Status copyBuffer(daal::services::internal::sycl::ExecutionContextIface & context,
                                daal::services::internal::sycl::UniversalBuffer & returnBuffer, data_management::NumericTable & data,
                                const uint32_t nRows, const uint32_t nCols);

    services::Status buildKernel(daal::services::internal::sycl::ExecutionContextIface & context,
                                 daal::services::internal::sycl::ClKernelFactoryIface & factory);

    services::Status checkVariances(data_management::NumericTable & pVariances, uint32_t numRows);

    services::Status computeInvSigmas(daal::services::internal::sycl::ExecutionContextIface & context, data_management::NumericTable * variances,
                                      const services::internal::Buffer<algorithmFPType> & invSigmas, const uint32_t numFeatures);

    services::Status normalize(daal::services::internal::sycl::ExecutionContextIface & context,
                               daal::services::internal::sycl::UniversalBuffer & copyBlock,
                               daal::services::internal::sycl::UniversalBuffer & rawMeans,
                               daal::services::internal::sycl::UniversalBuffer & invSigmas, bool hasMeans, bool hasInvSigmas,
                               const uint32_t numFeatures, const uint32_t numVectors);

    services::Status whitening(daal::services::internal::sycl::ExecutionContextIface & context,
                               const services::internal::Buffer<algorithmFPType> & transformedBlock,
                               daal::services::internal::sycl::UniversalBuffer & invEigenvalues, const uint32_t numComponents,
                               const uint32_t numVectors);

    services::Status initBuffers(daal::services::internal::sycl::ExecutionContextIface & ctx, data_management::NumericTable & data,
                                 const uint32_t numFeatures, const uint32_t numComponents, const uint32_t numVectors);

private:
    const unsigned int maxWorkItemsPerGroup = 256;
    daal::services::internal::sycl::UniversalBuffer invSigmas;
    daal::services::internal::sycl::UniversalBuffer invEigenvalues;
    daal::services::internal::sycl::UniversalBuffer rawMeans;
    daal::services::internal::sycl::UniversalBuffer copyBlock;
};

} // namespace internal
} // namespace oneapi
} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif

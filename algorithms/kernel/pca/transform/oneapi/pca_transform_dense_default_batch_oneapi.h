/* file: pca_dense_correlation_batch_kernel_ucapi.h */
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

/*
//++
//  Common functions of PCA transformation on GPU
//--
*/

#ifndef __PCA_TRANSFORM_DENSE_DEFAULT_BATCH_ONEAPI_H__
#define __PCA_TRANSFORM_DENSE_DEFAULT_BATCH_ONEAPI_H__

#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "pca_types.h"
#include "oneapi/blas_gpu.h"

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
    services::Status compute
            (data_management::NumericTable& data,
             data_management::NumericTable& eigenvectors,
             data_management::NumericTable *pMeans,
             data_management::NumericTable *pVariances,
             data_management::NumericTable *pEigenvalues,
             data_management::NumericTable &transformedData);

    void computeTransformedBlock
            (uint32_t numRows, uint32_t numFeatures, uint32_t numComponents,
             daal::oneapi::internal::UniversalBuffer & dataBlock,
             daal::oneapi::internal::UniversalBuffer & eigenvectors,
             const services::Buffer<algorithmFPType> & resultBlock);

private:
    services::Status computeInvSigmas(daal::oneapi::internal::ExecutionContextIface& context,
                                      const daal::oneapi::internal::KernelPtr& computeInvSigmasKernel,
                                      data_management::NumericTable* variances,
                                      const services::Buffer<algorithmFPType>& invSigmas,
                                      const size_t numFeatures);

    services::Status normalize(daal::oneapi::internal::ExecutionContextIface& context,
                               const daal::oneapi::internal::KernelPtr & normalizeKernel,
                               daal::oneapi::internal::UniversalBuffer & copyBlock,
                               daal::oneapi::internal::UniversalBuffer & rawMeans,
                               daal::oneapi::internal::UniversalBuffer & invSigmas,
                               size_t numMeans, size_t numInvSigmas, const unsigned int maxWorkItemsPerGroup,
                               const size_t numFeatures, const unsigned int workItemsPerGroup, uint32_t numVectors);

    services::Status whitening(daal::oneapi::internal::ExecutionContextIface& context,
                               const daal::oneapi::internal::KernelPtr & whiteningKernel,
                               daal::oneapi::internal::UniversalBuffer transformedBlock,
                               daal::oneapi::internal::UniversalBuffer invEigenvalues,
                               uint32_t numComponents, uint32_t numVectors);
};

} // namespace internal
} // namespace oneapi
} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif

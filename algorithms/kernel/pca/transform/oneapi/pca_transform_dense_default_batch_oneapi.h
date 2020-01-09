/* file: pca_transform_dense_default_batch_oneapi.h */
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

using namespace daal::services;
using namespace daal::data_management;
using namespace daal::oneapi::internal;

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
            (DAAL_INT numRows, DAAL_INT numFeatures, DAAL_INT numComponents,
             daal::oneapi::internal::UniversalBuffer & dataBlock,
             daal::oneapi::internal::UniversalBuffer & eigenvectors,
             services::Buffer<algorithmFPType> resultBlock);

private:
    services::Status computeInvSigmas(daal::oneapi::internal::ExecutionContextIface& context,
                                        const daal::oneapi::internal::KernelPtr& computeInvSigmasKernel,
                                        data_management::NumericTable* variances,
                                        const services::Buffer<algorithmFPType>& invSigmas,
                                        const size_t numFeatures);

};

} // namespace internal
} // namespace oneapi
} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif

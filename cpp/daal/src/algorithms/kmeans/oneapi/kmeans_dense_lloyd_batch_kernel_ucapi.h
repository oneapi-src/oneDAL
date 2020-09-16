/* file: kmeans_dense_lloyd_batch_kernel_ucapi.h */
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
//  Implementation of K-means Batch Kernel for GPU.
//--
*/

#ifndef __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_H__
#define __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_H__

#include "sycl/internal/types.h"
#include "sycl/internal/execution_context.h"
#include "algorithms/kmeans/kmeans_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/kmeans/oneapi/kmeans_dense_lloyd_kernel_base_ucapi.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
template <typename algorithmFPType>
class KMeansDenseLloydBatchKernelUCAPI : public KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>
{
public:
    services::Status compute(const NumericTable * const * a, const NumericTable * const * r, const Parameter * par);
};

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif

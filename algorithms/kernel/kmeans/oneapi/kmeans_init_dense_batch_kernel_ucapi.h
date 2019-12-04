/* file: kmeans_init_dense_batch_kernel_ucapi.h */
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
//  Declaration of template function that computes K-means.
//--
*/

#ifndef _KMEANS_INIT_DENSE_BATCH_KERNEL_UCAPI_H
#define _KMEANS_INIT_DENSE_BATCH_KERNEL_UCAPI_H

#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "kmeans_init_types.h"
#include "kernel.h"
#include "numeric_table.h"
#include "memory_block.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{
template <Method method, typename algorithmFPType>
class KMeansInitDenseBatchKernelUCAPI : public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable * const * a, size_t nr, const NumericTable * const * r, const Parameter * par,
                             engines::BatchBase & engine);

private:
    services::Status init(size_t p, size_t n, size_t nRowsTotal, size_t nClusters, NumericTable * ntClusters, NumericTable * ntData,
                          unsigned int seed, engines::BatchBase & engine, size_t & clustersFound);

    void gatherRandom(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::KernelPtr & kernel_gather_random,
                      const services::Buffer<algorithmFPType> & data, const services::Buffer<algorithmFPType> & clusters,
                      oneapi::internal::UniversalBuffer & indices, uint32_t nRows, uint32_t nClusters, uint32_t nFeatures, services::Status * st);

    uint32_t getWorkgroupsCount(uint32_t rows);

    const uint32_t _maxWorkItemsPerGroup = 256; // should be a power of two for interal needs
};

} // namespace internal
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif

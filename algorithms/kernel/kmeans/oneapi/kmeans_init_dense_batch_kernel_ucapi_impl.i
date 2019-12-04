/* file: kmeans_init_dense_batch_kernel_ucapi_impl.i */
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
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "cl_kernels/kmeans_init_cl_kernels.cl"
#include "execution_context.h"
#include "oneapi/service_defines_oneapi.h"
#include "oneapi/internal/types.h"

#include "numeric_table.h"
#include "daal_defines.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "uniform_kernel.h"
#include "uniform_impl.i"

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
using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::oneapi::internal;
using namespace daal::data_management;
using namespace daal::algorithms::distributions::uniform::internal;

template <Method method, typename algorithmFPType>
Status KMeansInitDenseBatchKernelUCAPI<method, algorithmFPType>::init(size_t p, size_t n, size_t nRowsTotal, size_t nClusters,
                                                                      NumericTable * ntClusters, NumericTable * ntData, unsigned int seed,
                                                                      engines::BatchBase & engine, size_t & clustersFound)
{
    Status st;

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add("-cl-std=CL1.2 -D LOCAL_SUM_SIZE=256"); // should be equal to _maxWorkitemsPerGroup

    services::String cachekey("__daal_algorithms_kmeans_init_dense_batch_");
    cachekey.add(fptype_name);

    kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), kmeans_init_cl_kernels, build_options.c_str());

    if (method == deterministicDense)
    {
        BlockDescriptor<algorithmFPType> dataRows;
        ntData->getBlockOfRows(0, nClusters, readOnly, dataRows);
        auto data = dataRows.getBuffer();

        BlockDescriptor<algorithmFPType> clustersRows;
        ntClusters->getBlockOfRows(0, nClusters, writeOnly, clustersRows);
        auto clusters = clustersRows.getBuffer();

        context.copy(clusters, 0, data, 0, nClusters * p, &st);
        DAAL_CHECK_STATUS_VAR(st);

        ntData->releaseBlockOfRows(dataRows);
        ntClusters->releaseBlockOfRows(clustersRows);

        clustersFound = nClusters;

        return st;
    }

    if (method == randomDense)
    {
        auto gather_random = kernel_factory.getKernel("gather_random");

        auto indices = context.allocate(TypeIds::id<int>(), nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);

        {
            auto indicesHostPtr = indices.get<int>().toHost(data_management::readWrite);
            auto * indicesHost  = indicesHostPtr.get();

            size_t k = 0;
            Status s;
            for (size_t i = 0; i < nClusters; i++)
            {
                DAAL_CHECK_STATUS(s, (UniformKernelDefault<int, sse2>::compute(i, (int)nRowsTotal, engine, 1, indicesHost + k)));
                size_t c    = (size_t)indicesHost[k];
                int & value = indicesHost[k];
                for (size_t j = k; j > 0; j--)
                {
                    if (value == indicesHost[j - 1])
                    {
                        c     = (size_t)(j - 1);
                        value = c;
                    }
                }
                if (c >= n) continue;
                k++;
            }

            clustersFound = k;
        }

        BlockDescriptor<algorithmFPType> dataRows;
        ntData->getBlockOfRows(0, nRowsTotal, readOnly, dataRows);
        auto data = dataRows.getBuffer();

        BlockDescriptor<algorithmFPType> clustersRows;
        ntClusters->getBlockOfRows(0, clustersFound, writeOnly, clustersRows);
        auto clusters = clustersRows.getBuffer();

        gatherRandom(context, gather_random, data, clusters, indices, nRowsTotal, clustersFound, p, &st);
        DAAL_CHECK_STATUS_VAR(st);

        ntData->releaseBlockOfRows(dataRows);
        ntClusters->releaseBlockOfRows(clustersRows);

        return st;
    }

    DAAL_ASSERT(false && "should never happen");
    return Status();
}

template <Method method, typename algorithmFPType>
services::Status KMeansInitDenseBatchKernelUCAPI<method, algorithmFPType>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                                   const NumericTable * const * r, const Parameter * par,
                                                                                   engines::BatchBase & engine)
{
    NumericTable * ntData     = const_cast<NumericTable *>(a[0]);
    NumericTable * ntClusters = const_cast<NumericTable *>(r[0]);

    const size_t p         = ntData->getNumberOfColumns();
    const size_t n         = ntData->getNumberOfRows();
    const size_t nClusters = par->nClusters;

    size_t clustersFound = 0;

    return init(p, n, n, nClusters, ntClusters, ntData, par->seed, engine, clustersFound);
}

template <Method method, typename algorithmFPType>
uint32_t KMeansInitDenseBatchKernelUCAPI<method, algorithmFPType>::getWorkgroupsCount(uint32_t rows)
{
    const uint32_t elementsPerGroup = _maxWorkItemsPerGroup;
    uint32_t workgroupsCount        = rows / elementsPerGroup;

    if (workgroupsCount * elementsPerGroup < rows) workgroupsCount++;

    return workgroupsCount;
}

template <Method method, typename algorithmFPType>
void KMeansInitDenseBatchKernelUCAPI<method, algorithmFPType>::gatherRandom(ExecutionContextIface & context, const KernelPtr & kernel_gather_random,
                                                                            const Buffer<algorithmFPType> & data,
                                                                            const Buffer<algorithmFPType> & clusters, UniversalBuffer & indices,
                                                                            uint32_t nRows, uint32_t nClusters, uint32_t nFeatures, Status * st)
{
    KernelArguments args(6);
    args.set(0, data, AccessModeIds::read);
    args.set(1, clusters, AccessModeIds::write);
    args.set(2, indices, AccessModeIds::read);
    args.set(3, nRows);
    args.set(4, nClusters);
    args.set(5, nFeatures);

    KernelRange local_range(1, _maxWorkItemsPerGroup);
    KernelRange global_range(nClusters, _maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        context.run(range, kernel_gather_random, args, st);
    }
}

} // namespace internal
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal

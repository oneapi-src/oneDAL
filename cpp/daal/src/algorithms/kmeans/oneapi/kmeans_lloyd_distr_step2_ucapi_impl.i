/* file: kmeans_lloyd_distr_step2_ucapi_impl.i */
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
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "services/env_detect.h"
#include "services/internal/sycl/execution_context.h"
#include "services/internal/sycl/types.h"
#include "src/services/service_data_utils.h"
#include "src/sycl/blas_gpu.h"
#include "src/algorithms/kmeans/oneapi/kmeans_lloyd_distr_step2_kernel_ucapi.h"
#include "src/algorithms/kmeans/oneapi/cl_kernels/kmeans_cl_kernels_distr_steps.cl"
#include "src/data_management/service_numeric_table.h"

#include "src/externals/service_profiler.h"
//#include <iostream>

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::services::internal::sycl;
using namespace daal::data_management;

constexpr size_t maxInt32AsSizeT     = static_cast<size_t>(daal::services::internal::MaxVal<int32_t>::get());
constexpr uint32_t maxInt32AsUint32T = static_cast<uint32_t>(daal::services::internal::MaxVal<int32_t>::get());

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
template <typename algorithmFPType>
Status KMeansDistributedStep2KernelUCAPI<algorithmFPType>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                   const NumericTable * const * r, const Parameter * par)
{
    Status st;

    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory));

    const size_t nClustersAsSizeT    = par->nClusters;
    const size_t nDataColumnsAsSizeT = r[1]->getNumberOfColumns();
    DAAL_CHECK(nClustersAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectParameter);
    DAAL_CHECK(nDataColumnsAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    DAAL_CHECK(na <= maxInt32AsSizeT, services::ErrorIncorrectParameter);
    const uint32_t nClusters = static_cast<uint32_t>(nClustersAsSizeT);
    const uint32_t nFeatures = static_cast<uint32_t>(nDataColumnsAsSizeT);

    NumericTable * ntClusterS0   = const_cast<NumericTable *>(r[0]);
    NumericTable * ntClusterS1   = const_cast<NumericTable *>(r[1]);
    NumericTable * ntObjFunction = const_cast<NumericTable *>(r[2]);
    NumericTable * ntCValues     = const_cast<NumericTable *>(r[3]);
    NumericTable * ntCCentroids  = const_cast<NumericTable *>(r[4]);
    NumericTable * ntAssignments = const_cast<NumericTable *>(r[5]);

    BlockDescriptor<int> ntClusterS0Rows;
    DAAL_CHECK_STATUS_VAR(ntClusterS0->getBlockOfRows(0, nClusters, writeOnly, ntClusterS0Rows));
    auto outCCounters = ntClusterS0Rows.getBuffer();

    BlockDescriptor<algorithmFPType> ntClusterS1Rows;
    DAAL_CHECK_STATUS_VAR(ntClusterS1->getBlockOfRows(0, nClusters, writeOnly, ntClusterS1Rows));
    auto outCentroids = ntClusterS1Rows.getBuffer();

    BlockDescriptor<algorithmFPType> ntObjFunctionRows;
    DAAL_CHECK_STATUS_VAR(ntObjFunction->getBlockOfRows(0, 1, writeOnly, ntObjFunctionRows));

    DAAL_ASSERT(ntObjFunctionRows.getBuffer().size() >= 1);
    auto outObjFunction = ntObjFunctionRows.getBuffer().toHost(data_management::writeOnly, st);
    DAAL_CHECK_STATUS_VAR(st);

    BlockDescriptor<algorithmFPType> ntCValuesRows;
    DAAL_CHECK_STATUS_VAR(ntCValues->getBlockOfRows(0, nClusters, writeOnly, ntCValuesRows));
    auto outCValues = ntCValuesRows.getBuffer();

    BlockDescriptor<int> ntCCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntCCentroids->getBlockOfRows(0, nClusters, writeOnly, ntCCentroidsRows));
    auto outCCentroids = ntCCentroidsRows.getBuffer();

    const uint32_t nBlocks = static_cast<uint32_t>(na) / 5;

    algorithmFPType tmpObjValue = 0.0;

    for (uint32_t i = 0; i < nBlocks; i++)
    {
        BlockDescriptor<int> ntParClusterS0Rows;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 0])->getBlockOfRows(0, nClusters, readOnly, ntParClusterS0Rows));
        auto inParClusterS0 = ntParClusterS0Rows.getBuffer();

        BlockDescriptor<algorithmFPType> ntParClusterS1Rows;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 1])->getBlockOfRows(0, nClusters, readOnly, ntParClusterS1Rows));
        auto inParClusterS1 = ntParClusterS1Rows.getBuffer();

        DAAL_CHECK_STATUS_VAR(updateClusters(i == 0, inParClusterS0, inParClusterS1, outCCounters, outCentroids, nClusters, nFeatures));

        BlockDescriptor<algorithmFPType> ntParObjFunctionRows;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 2])->getBlockOfRows(0, 1, readOnly, ntParObjFunctionRows));
        DAAL_ASSERT(ntParObjFunctionRows.getBuffer().size() > 0);
        auto inParObjFunction = ntParObjFunctionRows.getBuffer().toHost(data_management::readOnly, st);
        DAAL_CHECK_STATUS_VAR(st);
        tmpObjValue += *inParObjFunction.get();

        BlockDescriptor<algorithmFPType> ntParCValuesRows;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 3])->getBlockOfRows(0, nClusters, readOnly, ntParCValuesRows));
        auto inParCValues = ntParCValuesRows.getBuffer();

        BlockDescriptor<int> ntParCCandidates;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 4])->getBlockOfRows(0, nClusters, readOnly, ntParCCandidates));
        auto intParCCandidates = ntParCCandidates.getBuffer();

        DAAL_CHECK_STATUS_VAR(updateCandidates(i == 0, intParCCandidates, inParCValues, outCCentroids, outCValues, nClusters));

        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 0])->releaseBlockOfRows(ntParClusterS0Rows));
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 1])->releaseBlockOfRows(ntParClusterS1Rows));
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 2])->releaseBlockOfRows(ntParObjFunctionRows));
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 3])->releaseBlockOfRows(ntParCValuesRows));
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(a[i * 5 + 4])->releaseBlockOfRows(ntParCCandidates));
    }
    *outObjFunction.get() = tmpObjValue;
    {
        DAAL_ASSERT(outCentroids.size() >= nFeatures * nClusters);
        auto retCentroids = outCentroids.toHost(ReadWriteMode::readWrite, st);
        DAAL_CHECK_STATUS_VAR(st);
        DAAL_ASSERT(outCCounters.size() >= nClusters);
        auto retCCounters = outCCounters.toHost(ReadWriteMode::readOnly, st);
        DAAL_CHECK_STATUS_VAR(st);

        for (int j = 0; j < nClusters; j++)
        {
            int count = retCCounters.get()[j];
            if (!count) continue;
            for (int k = 0; k < nFeatures; k++) retCentroids.get()[j * nFeatures + k] *= count;
        }
    }
    DAAL_CHECK_STATUS_VAR(ntClusterS0->releaseBlockOfRows(ntClusterS0Rows));
    DAAL_CHECK_STATUS_VAR(ntClusterS1->releaseBlockOfRows(ntClusterS1Rows));
    DAAL_CHECK_STATUS_VAR(ntObjFunction->releaseBlockOfRows(ntObjFunctionRows));
    DAAL_CHECK_STATUS_VAR(ntCValues->releaseBlockOfRows(ntCValuesRows));
    DAAL_CHECK_STATUS_VAR(ntCCentroids->releaseBlockOfRows(ntCCentroidsRows));

    return st;
}

template <typename algorithmFPType>
Status KMeansDistributedStep2KernelUCAPI<algorithmFPType>::finalizeCompute(size_t na, const NumericTable * const * a, size_t nr,
                                                                           const NumericTable * const * r, const Parameter * par)
{
    Status st;
    const size_t nClustersAsSizeT    = par->nClusters;
    const size_t nDataColumnsAsSizeT = a[1]->getNumberOfColumns();
    DAAL_CHECK(nClustersAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectParameter);
    DAAL_CHECK(nDataColumnsAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    const uint32_t nClusters = static_cast<uint32_t>(nClustersAsSizeT);
    const uint32_t p         = static_cast<uint32_t>(nDataColumnsAsSizeT);
    int result               = 0;

    ReadRows<int, DAAL_BASE_CPU> mtInClusterS0(*const_cast<NumericTable *>(a[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusterS0);
    ReadRows<algorithmFPType, DAAL_BASE_CPU> mtInClusterS1(*const_cast<NumericTable *>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusterS1);
    ReadRows<algorithmFPType, DAAL_BASE_CPU> mtInTargetFunc(*const_cast<NumericTable *>(a[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtInTargetFunc);

    ReadRows<algorithmFPType, DAAL_BASE_CPU> mtCValues(*const_cast<NumericTable *>(a[3]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCValues);
    ReadRows<algorithmFPType, DAAL_BASE_CPU> mtCCentroids(*const_cast<NumericTable *>(a[4]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCCentroids);

    const int * clusterS0             = mtInClusterS0.get();
    const algorithmFPType * clusterS1 = mtInClusterS1.get();
    const algorithmFPType * inTarget  = mtInTargetFunc.get();

    const algorithmFPType * cValues    = mtCValues.get();
    const algorithmFPType * cCentroids = mtCCentroids.get();

    WriteOnlyRows<algorithmFPType, DAAL_BASE_CPU> mtClusters(*const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);
    WriteOnlyRows<algorithmFPType, DAAL_BASE_CPU> mtTargetFunct(*const_cast<NumericTable *>(r[1]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTargetFunct);

    algorithmFPType * clusters  = mtClusters.get();
    algorithmFPType * outTarget = mtTargetFunct.get();

    *outTarget = *inTarget;

    uint32_t cPos = 0;

    for (uint32_t i = 0; i < nClusters; i++)
    {
        if (clusterS0[i] > 0)
        {
            algorithmFPType coeff = 1.0 / clusterS0[i];

            for (uint32_t j = 0; j < p; j++)
            {
                clusters[i * p + j] = clusterS1[i * p + j] * coeff;
            }
        }
        else
        {
            DAAL_CHECK(!(cValues[cPos] < (algorithmFPType)0.0), services::ErrorKMeansNumberOfClustersIsTooLarge);
            outTarget[0] -= cValues[cPos];
            result |= daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), &cCentroids[cPos * p],
                                                              p * sizeof(algorithmFPType));
            cPos++;
        }
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType>
Status KMeansDistributedStep2KernelUCAPI<algorithmFPType>::updateClusters(bool init, const services::internal::Buffer<int> & partialCentroidsCounters,
                                                                          const services::internal::Buffer<algorithmFPType> & partialCentroids,
                                                                          const services::internal::Buffer<int> & centroidCounters,
                                                                          const services::internal::Buffer<algorithmFPType> & centroids,
                                                                          uint32_t nClusters, uint32_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids);
    Status st;
    auto & context            = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory      = context.getClKernelFactory();
    auto kernelUpdateClusters = init ? kernelFactory.getKernel("init_clusters", st) : kernelFactory.getKernel("update_clusters", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT(nFeatures <= maxInt32AsUint32T);
    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nFeatures, nClusters);
    DAAL_ASSERT(partialCentroidsCounters.size() >= nClusters);
    DAAL_ASSERT(partialCentroids.size() >= nClusters * nFeatures);
    DAAL_ASSERT(centroidCounters.size() >= nClusters);
    DAAL_ASSERT(centroids.size() >= nClusters * nFeatures);

    KernelArguments args(5, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, partialCentroidsCounters, AccessModeIds::read);
    args.set(1, partialCentroids, AccessModeIds::read);
    args.set(2, centroidCounters, AccessModeIds::readwrite);
    args.set(3, centroids, AccessModeIds::readwrite);
    args.set(4, static_cast<int32_t>(nFeatures));

    KernelRange local_range(1, nFeatures > _maxWGSize ? _maxWGSize : nFeatures);
    KernelRange global_range(nClusters, nFeatures);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateClusters.run);
        context.run(range, kernelUpdateClusters, args, st);
    }
    return st;
}

template <typename algorithmFPType>
Status KMeansDistributedStep2KernelUCAPI<algorithmFPType>::updateCandidates(bool init, const services::internal::Buffer<int> & partialCandidates,
                                                                            const services::internal::Buffer<algorithmFPType> & partialCValues,
                                                                            const services::internal::Buffer<int> & candidates,
                                                                            const services::internal::Buffer<algorithmFPType> & cValues,
                                                                            uint32_t nClusters)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids);
    Status st;
    auto & context              = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory        = context.getClKernelFactory();
    auto kernelUpdateCandidates = init ? kernelFactory.getKernel("init_candidates", st) : kernelFactory.getKernel("update_candidates", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);
    DAAL_ASSERT(partialCandidates.size() >= nClusters);
    DAAL_ASSERT(partialCValues.size() >= nClusters);
    DAAL_ASSERT(candidates.size() >= nClusters);
    DAAL_ASSERT(cValues.size() >= nClusters);

    KernelArguments args(5, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, partialCandidates, AccessModeIds::read);
    args.set(1, partialCValues, AccessModeIds::read);
    args.set(2, candidates, AccessModeIds::readwrite);
    args.set(3, cValues, AccessModeIds::readwrite);
    args.set(4, static_cast<int32_t>(nClusters));

    KernelRange local_range(1, 1);
    KernelRange global_range(1, 1);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_VAR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateClusters.run);
        context.run(range, kernelUpdateCandidates, args, st);
    }
    return st;
}

template <typename algorithmFPType>
Status KMeansDistributedStep2KernelUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernelFactory)
{
    Status st;
    auto fptypeName   = services::internal::sycl::getKeyFPType<algorithmFPType>();
    auto buildOptions = fptypeName;
    services::String cachekey("__daal_algorithms_kmeans_lloyd_dense_distr_step2_");
    cachekey.add(fptypeName);
    cachekey.add(buildOptions);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), kmeans_cl_kernels_distr_steps, buildOptions.c_str(), st);
    }
    return st;
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

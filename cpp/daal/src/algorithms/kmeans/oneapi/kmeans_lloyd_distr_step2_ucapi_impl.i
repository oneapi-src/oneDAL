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
#include "sycl/internal/execution_context.h"
#include "sycl/internal/types.h"
#include "src/sycl/blas_gpu.h"
#include "src/algorithms/kmeans/oneapi/kmeans_lloyd_distr_step2_kernel_ucapi.h"
#include "src/algorithms/kmeans/oneapi/cl_kernels/kmeans_cl_kernels_distr_steps.cl"
#include "src/data_management/service_numeric_table.h"

#include "src/externals/service_ittnotify.h"
//#include <iostream>

DAAL_ITTNOTIFY_DOMAIN(kmeans.dense.lloyd.distr.step2.oneapi);

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::oneapi::internal;
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
Status KMeansDistributedStep2KernelUCAPI<algorithmFPType>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                   const NumericTable * const * r, const Parameter * par)
{
    Status st;

    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    this->buildProgram(kernelFactory, &st);
    DAAL_CHECK_STATUS_VAR(st);

    const size_t nClusters = par->nClusters;
    const size_t nFeatures = r[1]->getNumberOfColumns();

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
    DAAL_CHECK_STATUS_VAR(ntObjFunction->getBlockOfRows(0, nClusters, writeOnly, ntObjFunctionRows));
    auto outObjFunction = ntObjFunctionRows.getBuffer().toHost(data_management::readOnly);

    BlockDescriptor<algorithmFPType> ntCValuesRows;
    DAAL_CHECK_STATUS_VAR(ntCValues->getBlockOfRows(0, nClusters, writeOnly, ntCValuesRows));
    auto outCValues = ntCValuesRows.getBuffer();

    BlockDescriptor<int> ntCCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntCCentroids->getBlockOfRows(0, nClusters, writeOnly, ntCCentroidsRows));
    auto outCCentroids = ntCCentroidsRows.getBuffer();

    const size_t nBlocks = na / 5;

    algorithmFPType tmpObjValue = 0.0;

    for (size_t i = 0; i < nBlocks; i++)
    {
        BlockDescriptor<int> ntParClusterS0Rows;
        const_cast<NumericTable *>(a[i * 5 + 0])->getBlockOfRows(0, nClusters, readOnly, ntParClusterS0Rows);
        auto inParClusterS0 = ntParClusterS0Rows.getBuffer();

        BlockDescriptor<algorithmFPType> ntParClusterS1Rows;
        const_cast<NumericTable *>(a[i * 5 + 1])->getBlockOfRows(0, nClusters, readOnly, ntParClusterS1Rows);
        auto inParClusterS1 = ntParClusterS1Rows.getBuffer();

        updateClusters(i == 0, inParClusterS0, inParClusterS1, outCCounters, outCentroids, nClusters, nFeatures, &st);

        BlockDescriptor<algorithmFPType> ntParObjFunctionRows;
        const_cast<NumericTable *>(a[i * 5 + 2])->getBlockOfRows(0, 1, readOnly, ntParObjFunctionRows);
        auto inParObjFunction = ntParObjFunctionRows.getBuffer().toHost(data_management::readOnly);
        tmpObjValue += inParObjFunction.get()[0];

        BlockDescriptor<algorithmFPType> ntParCValuesRows;
        const_cast<NumericTable *>(a[i * 5 + 3])->getBlockOfRows(0, nClusters, readOnly, ntParCValuesRows);
        auto inParCValues = ntParCValuesRows.getBuffer();

        BlockDescriptor<int> ntParCCandidates;
        const_cast<NumericTable *>(a[i * 5 + 4])->getBlockOfRows(0, nClusters, readOnly, ntParCCandidates);
        auto intParCCandidates = ntParCCandidates.getBuffer();

        updateCandidates(i == 0, intParCCandidates, inParCValues, outCCentroids, outCValues, nClusters, &st);

        const_cast<NumericTable *>(a[i * 5 + 0])->releaseBlockOfRows(ntParClusterS0Rows);
        const_cast<NumericTable *>(a[i * 5 + 1])->releaseBlockOfRows(ntParClusterS1Rows);
        const_cast<NumericTable *>(a[i * 5 + 2])->releaseBlockOfRows(ntParObjFunctionRows);
        const_cast<NumericTable *>(a[i * 5 + 3])->releaseBlockOfRows(ntParCValuesRows);
        const_cast<NumericTable *>(a[i * 5 + 4])->releaseBlockOfRows(ntParCCandidates);
    }
    outObjFunction.get()[0] = tmpObjValue;
    {
        auto retCentroids = outCentroids.toHost(ReadWriteMode::readWrite);
        auto retCCounters = outCCounters.toHost(ReadWriteMode::readOnly);
        for (int j = 0; j < nClusters; j++)
        {
            int count = retCCounters.get()[j];
            if (!count) continue;
            for (int k = 0; k < nFeatures; k++) retCentroids.get()[j * nFeatures + k] *= count;
        }
    }
    ntClusterS0->releaseBlockOfRows(ntClusterS0Rows);
    ntClusterS1->releaseBlockOfRows(ntClusterS1Rows);
    ntObjFunction->releaseBlockOfRows(ntObjFunctionRows);
    ntCValues->releaseBlockOfRows(ntCValuesRows);
    ntCCentroids->releaseBlockOfRows(ntCCentroidsRows);

    return st;
}

template <typename algorithmFPType>
Status KMeansDistributedStep2KernelUCAPI<algorithmFPType>::finalizeCompute(size_t na, const NumericTable * const * a, size_t nr,
                                                                           const NumericTable * const * r, const Parameter * par)
{
    const size_t p         = a[1]->getNumberOfColumns();
    const size_t nClusters = par->nClusters;
    int result             = 0;

    ReadRows<int, sse2> mtInClusterS0(*const_cast<NumericTable *>(a[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusterS0);
    ReadRows<algorithmFPType, sse2> mtInClusterS1(*const_cast<NumericTable *>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusterS1);
    ReadRows<algorithmFPType, sse2> mtInTargetFunc(*const_cast<NumericTable *>(a[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtInTargetFunc);

    ReadRows<algorithmFPType, sse2> mtCValues(*const_cast<NumericTable *>(a[3]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCValues);
    ReadRows<algorithmFPType, sse2> mtCCentroids(*const_cast<NumericTable *>(a[4]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCCentroids);

    // TODO: That should be size_t or double
    const int * clusterS0             = mtInClusterS0.get();
    const algorithmFPType * clusterS1 = mtInClusterS1.get();
    const algorithmFPType * inTarget  = mtInTargetFunc.get();

    const algorithmFPType * cValues    = mtCValues.get();
    const algorithmFPType * cCentroids = mtCCentroids.get();

    WriteOnlyRows<algorithmFPType, sse2> mtClusters(*const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);
    WriteOnlyRows<algorithmFPType, sse2> mtTargetFunct(*const_cast<NumericTable *>(r[1]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTargetFunct);

    algorithmFPType * clusters  = mtClusters.get();
    algorithmFPType * outTarget = mtTargetFunct.get();

    *outTarget = *inTarget;

    size_t cPos = 0;

    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterS0[i] > 0)
        {
            algorithmFPType coeff = 1.0 / clusterS0[i];

            for (size_t j = 0; j < p; j++)
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
void KMeansDistributedStep2KernelUCAPI<algorithmFPType>::updateClusters(bool init, const Buffer<int> & partialCentroidsCounters,
                                                                        const Buffer<algorithmFPType> & partialCentroids,
                                                                        const Buffer<int> & centroidCounters,
                                                                        const Buffer<algorithmFPType> & centroids, uint32_t nClusters,
                                                                        uint32_t nFeatures, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids);

    auto & context            = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory      = context.getClKernelFactory();
    auto kernelUpdateClusters = init ? kernelFactory.getKernel("init_clusters_distr", st) : kernelFactory.getKernel("update_clusters_distr", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(5);
    args.set(0, partialCentroidsCounters, AccessModeIds::read);
    args.set(1, partialCentroids, AccessModeIds::read);
    args.set(2, centroidCounters, AccessModeIds::readwrite);
    args.set(3, centroids, AccessModeIds::readwrite);
    args.set(4, nFeatures);

    KernelRange local_range(1, nFeatures > _maxWGSize ? _maxWGSize : nFeatures);
    KernelRange global_range(nClusters, nFeatures);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateClusters.run);
        context.run(range, kernelUpdateClusters, args, st);
    }
    DAAL_CHECK_STATUS_PTR(st);
}

template <typename algorithmFPType>
void KMeansDistributedStep2KernelUCAPI<algorithmFPType>::updateCandidates(bool init, const Buffer<int> & partialCandidates,
                                                                          const Buffer<algorithmFPType> & partialCValues,
                                                                          const Buffer<int> & candidates, const Buffer<algorithmFPType> & cValues,
                                                                          uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids);

    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    auto kernelUpdateCandidates =
        init ? kernelFactory.getKernel("init_candidates_distr", st) : kernelFactory.getKernel("update_candidates_distr", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(5);
    args.set(0, partialCandidates, AccessModeIds::read);
    args.set(1, partialCValues, AccessModeIds::read);
    args.set(2, candidates, AccessModeIds::readwrite);
    args.set(3, cValues, AccessModeIds::readwrite);
    args.set(4, nClusters);

    KernelRange local_range(1, 1);
    KernelRange global_range(1, 1);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateClusters.run);
        context.run(range, kernelUpdateCandidates, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDistributedStep2KernelUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernelFactory, Status * st)
{
    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    services::String cachekey("__daal_algorithms_kmeans_lloyd_dense_distr_step2_");
    cachekey.add(fptype_name);
    cachekey.add(build_options.c_str());
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), kmeans_cl_kernels_distr_steps, build_options.c_str(), st);
    }
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

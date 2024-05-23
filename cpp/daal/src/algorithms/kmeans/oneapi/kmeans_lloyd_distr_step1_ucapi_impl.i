/* file: kmeans_lloyd_distr_step1_ucapi_impl.i */
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
#include "src/algorithms/kmeans/oneapi/kmeans_lloyd_distr_step1_kernel_ucapi.h"

#include "src/externals/service_profiler.h"

constexpr size_t maxInt32AsSizeT = static_cast<size_t>(daal::services::internal::MaxVal<int32_t>::get());

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::services::internal::sycl;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
using namespace daal::services::internal::sycl;
template <typename algorithmFPType>
Status KMeansDistributedStep1KernelUCAPI<algorithmFPType>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                   const NumericTable * const * r, const Parameter * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    Status st;

    auto & context       = Environment::getInstance().getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();

    NumericTable * ntData        = const_cast<NumericTable *>(a[0]);
    NumericTable * ntInCentroids = const_cast<NumericTable *>(a[1]);
    NumericTable * ntClusterS0   = const_cast<NumericTable *>(r[0]);
    NumericTable * ntClusterS1   = const_cast<NumericTable *>(r[1]);
    NumericTable * ntObjFunction = const_cast<NumericTable *>(r[2]);
    NumericTable * ntCValues     = const_cast<NumericTable *>(r[3]);
    NumericTable * ntCCentroids  = const_cast<NumericTable *>(r[4]);
    NumericTable * ntAssignments = const_cast<NumericTable *>(r[5]);

    const size_t nDataRowsAsSizeT    = ntData->getNumberOfRows();
    const size_t nDataColumnsAsSizeT = ntData->getNumberOfColumns();
    DAAL_CHECK(nDataRowsAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectNumberOfRowsInInputNumericTable);
    DAAL_CHECK(nDataColumnsAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    const uint32_t nRows     = static_cast<uint32_t>(nDataRowsAsSizeT);
    const uint32_t nFeatures = static_cast<uint32_t>(nDataColumnsAsSizeT);

    const size_t nClustersAsSizeT = par->nClusters;
    DAAL_CHECK(nClustersAsSizeT <= maxInt32AsSizeT, services::ErrorIncorrectParameter);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nClustersAsSizeT, nDataColumnsAsSizeT);
    const uint32_t nClusters = static_cast<uint32_t>(nClustersAsSizeT);

    DAAL_ASSERT(ntObjFunction->getNumberOfRows() == 1 && ntObjFunction->getNumberOfColumns() == 1);

    uint32_t blockSize = 0;
    DAAL_CHECK_STATUS_VAR(this->getBlockSize(nRows, nClusters, nFeatures, blockSize));
    DAAL_CHECK_STATUS_VAR(this->fitPartialCentroidSize(nClusters, nFeatures));
    DAAL_CHECK_STATUS_VAR(this->initializeBuffers(nClusters, nFeatures, blockSize));
    DAAL_ASSERT_UNIVERSAL_BUFFER(this->_numEmptyClusters, int, 1);

    BlockDescriptor<algorithmFPType> inCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntInCentroids->getBlockOfRows(0, nClusters, readOnly, inCentroidsRows));
    auto inCentroids = inCentroidsRows.getBuffer();

    BlockDescriptor<int> ntClusterS0Rows;
    DAAL_CHECK_STATUS_VAR(ntClusterS0->getBlockOfRows(0, nClusters, writeOnly, ntClusterS0Rows));
    auto outCCounters = ntClusterS0Rows.getBuffer();

    BlockDescriptor<algorithmFPType> ntClusterS1Rows;
    DAAL_CHECK_STATUS_VAR(ntClusterS1->getBlockOfRows(0, nClusters, writeOnly, ntClusterS1Rows));
    auto outCentroids = ntClusterS1Rows.getBuffer();

    BlockDescriptor<algorithmFPType> ntObjFunctionRows;
    DAAL_CHECK_STATUS_VAR(ntObjFunction->getBlockOfRows(0, nClusters, writeOnly, ntObjFunctionRows));
    auto outObjFunction = ntObjFunctionRows.getBuffer();

    BlockDescriptor<algorithmFPType> ntCValuesRows;
    DAAL_CHECK_STATUS_VAR(ntCValues->getBlockOfRows(0, nClusters, writeOnly, ntCValuesRows));
    auto outCValues = UniversalBuffer(ntCValuesRows.getBuffer());

    BlockDescriptor<algorithmFPType> ntCCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntCCentroids->getBlockOfRows(0, nClusters, writeOnly, ntCCentroidsRows));
    auto outCCentroids = UniversalBuffer(ntCCentroidsRows.getBuffer());

    DAAL_ASSERT_UNIVERSAL_BUFFER(outCValues, algorithmFPType, nClusters);
    context.fill(outCValues, sizeof(algorithmFPType) == 4 ? FLT_MAX : DBL_MAX, st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory, nClusters));

    auto assignments = context.allocate(TypeIds::id<int>(), blockSize, st);
    DAAL_CHECK_STATUS_VAR(st);

    math::SumReducer::Result dataSums(context, blockSize, TypeIds::id<algorithmFPType>(), st);
    DAAL_CHECK_STATUS_VAR(st);
    math::SumReducer::Result centroidsSums(context, blockSize, TypeIds::id<algorithmFPType>(), st);
    DAAL_CHECK_STATUS_VAR(st);

    size_t nPartNum = this->getCandidatePartNum(nClusters);
    size_t nBlocks  = nRows / blockSize + int(nRows % blockSize != 0);

    bool needCandidates = true;
    for (size_t block = 0; block < nBlocks; block++)
    {
        auto range = Range::createFromBlock(block, blockSize, nRows);
        BlockDescriptor<algorithmFPType> dataRows;
        DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(range.startIndex, range.count, readOnly, dataRows));
        auto data = dataRows.getBuffer();
        DAAL_CHECK_STATUS_VAR(this->computeSquares(inCentroids, centroidsSums, this->_centroidsSq, nClusters, nFeatures));
        DAAL_CHECK_STATUS_VAR(this->computeDistances(data, inCentroids, range.count, nClusters, nFeatures));
        DAAL_CHECK_STATUS_VAR(this->computeAssignments(assignments, range.count, nClusters));
        DAAL_CHECK_STATUS_VAR(this->computeSquares(data, dataSums, this->_dataSq, range.count, nFeatures));
        DAAL_CHECK_STATUS_VAR(this->partialReduceCentroids(data, assignments, range.count, nClusters, nFeatures, int(block == 0)));
        if (needCandidates)
        {
            DAAL_CHECK_STATUS_VAR(this->getNumEmptyClusters(nClusters));
            DAAL_CHECK_STATUS_VAR(st);
            int numEmpty = 0;
            {
                DAAL_ASSERT_UNIVERSAL_BUFFER(this->_numEmptyClusters, int, 1);
                auto num = this->_numEmptyClusters.template get<int>().toHost(ReadWriteMode::readOnly, st);
                DAAL_CHECK_STATUS_VAR(st);
                numEmpty = num.get()[0];
            }
            bool hasEmptyClusters = numEmpty > 0;
            if (hasEmptyClusters)
            {
                DAAL_CHECK_STATUS_VAR(this->computePartialCandidates(assignments, range.count, nClusters, int(block == 0)));
                DAAL_CHECK_STATUS_VAR(this->mergePartialCandidates(nClusters));
            }
            needCandidates = hasEmptyClusters;
        }
        DAAL_CHECK_STATUS_VAR(this->updateObjectiveFunction(outObjFunction, range.count, nClusters, int(block == 0)));
        DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
        if (par->assignFlag)
        {
            BlockDescriptor<int> assignmentsRows;
            DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, nRows, writeOnly, assignmentsRows));
            auto finalAssignments = assignmentsRows.getBuffer();
            DAAL_ASSERT(finalAssignments.size() >= range.startIndex + range.count);
            DAAL_ASSERT_UNIVERSAL_BUFFER(assignments, int, range.count);
            context.copy(finalAssignments, range.startIndex, assignments, 0, range.count, st);
            DAAL_CHECK_STATUS_VAR(st);
            DAAL_CHECK_STATUS_VAR(ntAssignments->releaseBlockOfRows(assignmentsRows));
        }
    }
    DAAL_CHECK_STATUS_VAR(this->mergeReduceCentroids(outCentroids, nClusters, nFeatures));
    DAAL_ASSERT(outCCounters.size() >= nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(this->_partialCentroidsCounters, int, nClusters);
    context.copy(outCCounters, 0, this->_partialCentroidsCounters, 0, nClusters, st);
    DAAL_CHECK_STATUS_VAR(st);
    DAAL_CHECK_STATUS_VAR(ntInCentroids->releaseBlockOfRows(inCentroidsRows));
    DAAL_CHECK_STATUS_VAR(ntClusterS0->releaseBlockOfRows(ntClusterS0Rows));
    DAAL_CHECK_STATUS_VAR(ntClusterS1->releaseBlockOfRows(ntClusterS1Rows));
    DAAL_CHECK_STATUS_VAR(ntObjFunction->releaseBlockOfRows(ntObjFunctionRows));
    if (needCandidates)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(outCValues, algorithmFPType, nClusters);
        DAAL_ASSERT_UNIVERSAL_BUFFER(this->_candidateDistances, algorithmFPType, nClusters);
        context.copy(outCValues, 0, this->_candidateDistances, 0, nClusters, st);
    }
    DAAL_CHECK_STATUS_VAR(st);
    DAAL_CHECK_STATUS_VAR(ntCValues->releaseBlockOfRows(ntCValuesRows));
    if (needCandidates)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(this->_candidates, int, nClusters);
        auto hostCandidates = this->_candidates.template get<int>().toHost(ReadWriteMode::readOnly, st);
        DAAL_CHECK_STATUS_VAR(st);
        for (uint32_t cPos = 0; cPos < nClusters; cPos++)
        {
            int index = hostCandidates.get()[cPos];
            if (index < 0 || index >= nRows)
            {
                continue;
            }
            BlockDescriptor<algorithmFPType> dataRows;
            DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(index, 1, readOnly, dataRows));
            DAAL_ASSERT_UNIVERSAL_BUFFER(outCCentroids, algorithmFPType, cPos * nFeatures + nFeatures);
            DAAL_ASSERT(dataRows.getBuffer().size() >= nFeatures);
            context.copy(outCCentroids, cPos * nFeatures, dataRows.getBuffer(), 0, nFeatures, st);
            DAAL_CHECK_STATUS_VAR(st);
        }
    }
    DAAL_CHECK_STATUS_VAR(ntCCentroids->releaseBlockOfRows(ntCCentroidsRows));
    return st;
}

template <typename algorithmFPType>
Status KMeansDistributedStep1KernelUCAPI<algorithmFPType>::finalizeCompute(size_t na, const NumericTable * const * a, size_t nr,
                                                                           const NumericTable * const * r, const Parameter * par)
{
    if (!par->assignFlag) return Status();

    NumericTable * ntPartialAssignments = const_cast<NumericTable *>(a[0]);
    NumericTable * ntAssignments        = const_cast<NumericTable *>(r[0]);
    const size_t n                      = ntPartialAssignments->getNumberOfRows();

    BlockDescriptor<int> inBlock;
    DAAL_CHECK_STATUS_VAR(ntPartialAssignments->getBlockOfRows(0, n, readOnly, inBlock));

    BlockDescriptor<int> outBlock;
    DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, n, writeOnly, outBlock));

    auto & context = Environment::getInstance().getDefaultExecutionContext();
    Status status;
    DAAL_ASSERT(outBlock.getBuffer().size() >= n);
    DAAL_ASSERT(inBlock.getBuffer().size() >= n);
    context.copy(outBlock.getBuffer(), 0, inBlock.getBuffer(), 0, n, status);
    DAAL_CHECK_STATUS_VAR(ntPartialAssignments->releaseBlockOfRows(inBlock));
    DAAL_CHECK_STATUS_VAR(ntAssignments->releaseBlockOfRows(outBlock));
    return status;
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

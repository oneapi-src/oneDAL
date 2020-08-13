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
#include "sycl/internal/execution_context.h"
#include "sycl/internal/types.h"
#include "src/sycl/blas_gpu.h"
#include "src/algorithms/kmeans/oneapi/kmeans_lloyd_distr_step1_kernel_ucapi.h"

#include "src/externals/service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(kmeans.dense.lloyd.distr.step1.oneapi);

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
Status KMeansDistributedStep1KernelUCAPI<algorithmFPType>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                   const NumericTable * const * r, const Parameter * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    Status st;

    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();

    NumericTable * ntData        = const_cast<NumericTable *>(a[0]);
    NumericTable * ntInCentroids = const_cast<NumericTable *>(a[1]);
    NumericTable * ntClusterS0   = const_cast<NumericTable *>(r[0]);
    NumericTable * ntClusterS1   = const_cast<NumericTable *>(r[1]);
    NumericTable * ntObjFunction = const_cast<NumericTable *>(r[2]);
    NumericTable * ntCValues     = const_cast<NumericTable *>(r[3]);
    NumericTable * ntCCentroids  = const_cast<NumericTable *>(r[4]);
    NumericTable * ntAssignments = const_cast<NumericTable *>(r[5]);

    const size_t nIter     = par->maxIterations;
    const size_t nRows     = ntData->getNumberOfRows();
    const size_t nFeatures = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;

    uint32_t blockSize = 0;
    DAAL_CHECK_STATUS_VAR(this->getBlockSize(nRows, nClusters, nFeatures, blockSize));
    DAAL_CHECK_STATUS_VAR(this->initializeBuffers(nClusters, nFeatures, blockSize));

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

    if (!inCentroids || !outCCounters || !outCentroids || !outObjFunction)
    {
        return Status(ErrorNullPtr);
    }

    BlockDescriptor<algorithmFPType> ntCValuesRows;
    DAAL_CHECK_STATUS_VAR(ntCValues->getBlockOfRows(0, nClusters, writeOnly, ntCValuesRows));
    auto outCValues = UniversalBuffer(ntCValuesRows.getBuffer());

    BlockDescriptor<algorithmFPType> ntCCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntCCentroids->getBlockOfRows(0, nClusters, writeOnly, ntCCentroidsRows));
    auto outCCentroids = UniversalBuffer(ntCCentroidsRows.getBuffer());

    context.fill(outCValues, sizeof(algorithmFPType) == 4 ? FLT_MAX : DBL_MAX, &st);
    DAAL_CHECK_STATUS_VAR(st);

    this->buildProgram(kernelFactory, nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);

    auto assignments = context.allocate(TypeIds::id<int>(), blockSize, &st);
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
        if (!data)
        {
            return Status(ErrorNullPtr);
        }
        this->computeSquares(inCentroids, this->_centroidsSq, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeDistances(data, inCentroids, range.count, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeAssignments(assignments, range.count, nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeSquares(data, this->_dataSq, range.count, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->partialReduceCentroids(data, assignments, range.count, nClusters, nFeatures, int(block == 0), &st);
        DAAL_CHECK_STATUS_VAR(st);
        if (needCandidates)
        {
            this->getNumEmptyClusters(nClusters, &st);
            DAAL_CHECK_STATUS_VAR(st);
            int numEmpty = 0;
            {
                auto num = this->_numEmptyClusters.template get<int>().toHost(ReadWriteMode::readOnly);
                if (!num.get())
                {
                    return Status(ErrorNullPtr);
                }

                numEmpty = num.get()[0];
            }
            bool hasEmptyClusters = numEmpty > 0;
            if (hasEmptyClusters)
            {
                this->computePartialCandidates(assignments, range.count, nClusters, int(block == 0), &st);
                DAAL_CHECK_STATUS_VAR(st);
                this->mergePartialCandidates(nClusters, &st);
                DAAL_CHECK_STATUS_VAR(st);
            }
            needCandidates = hasEmptyClusters;
        }
        this->updateObjectiveFunction(outObjFunction, range.count, nClusters, int(block == 0), &st);
        DAAL_CHECK_STATUS_VAR(st);
        ntData->releaseBlockOfRows(dataRows);
        if (par->assignFlag)
        {
            BlockDescriptor<int> assignmentsRows;
            DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, nRows, writeOnly, assignmentsRows));
            auto finalAssignments = assignmentsRows.getBuffer();
            context.copy(finalAssignments, range.startIndex, assignments, 0, range.count, &st);
            DAAL_CHECK_STATUS_VAR(st);
            DAAL_CHECK_STATUS_VAR(ntAssignments->releaseBlockOfRows(assignmentsRows));
        }
    }
    this->mergeReduceCentroids(outCentroids, nClusters, nFeatures, &st);
    DAAL_CHECK_STATUS_VAR(st);
    context.copy(outCCounters, 0, this->_partialCentroidsCounters, 0, nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);
    DAAL_CHECK_STATUS_VAR(ntInCentroids->releaseBlockOfRows(inCentroidsRows));
    DAAL_CHECK_STATUS_VAR(ntClusterS0->releaseBlockOfRows(ntClusterS0Rows));
    DAAL_CHECK_STATUS_VAR(ntClusterS1->releaseBlockOfRows(ntClusterS1Rows));
    DAAL_CHECK_STATUS_VAR(ntObjFunction->releaseBlockOfRows(ntObjFunctionRows));
    if (needCandidates)
    {
        context.copy(outCValues, 0, this->_candidateDistances, 0, nClusters, &st);
    }
    DAAL_CHECK_STATUS_VAR(st);
    DAAL_CHECK_STATUS_VAR(ntCValues->releaseBlockOfRows(ntCValuesRows));
    if (needCandidates)
    {
        auto hostCandidates = this->_candidates.template get<int>().toHost(ReadWriteMode::readOnly);
        if (!hostCandidates)
        {
            return Status(ErrorNullPtr);
        }
        for (uint32_t cPos = 0; cPos < nClusters; cPos++)
        {
            int index = hostCandidates.get()[cPos];
            if (index < 0 || index >= nRows)
            {
                continue;
            }
            BlockDescriptor<algorithmFPType> dataRows;
            DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(index, 1, readOnly, dataRows));
            context.copy(outCCentroids, cPos * nFeatures, dataRows.getBuffer(), 0, nFeatures, &st);
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

    auto & context = Environment::getInstance()->getDefaultExecutionContext();
    Status status;
    context.copy(outBlock.getBuffer(), 0, inBlock.getBuffer(), 0, n, &status);
    DAAL_CHECK_STATUS_VAR(ntPartialAssignments->releaseBlockOfRows(inBlock));
    DAAL_CHECK_STATUS_VAR(ntAssignments->releaseBlockOfRows(outBlock));
    return status;
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

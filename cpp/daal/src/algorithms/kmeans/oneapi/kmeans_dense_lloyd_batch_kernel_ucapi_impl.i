/* file: kmeans_dense_lloyd_batch_kernel_ucapi_impl.i */
/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_IMPL__
#define __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_IMPL__

#include "services/env_detect.h"
#include "src/algorithms/kmeans/oneapi/cl_kernels/kmeans_cl_kernels.cl"
#include "sycl/internal/execution_context.h"
#include "sycl/internal/types.h"
#include "src/sycl/blas_gpu.h"
#include "src/algorithms/kmeans/oneapi/kmeans_dense_lloyd_kernel_base_ucapi_impl.i"

#include "src/externals/service_ittnotify.h"

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
Status KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::compute(const NumericTable * const * a, const NumericTable * const * r,
                                                                  const Parameter * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    Status st;

    NumericTable * ntData         = const_cast<NumericTable *>(a[0]);
    NumericTable * ntInCentroids  = const_cast<NumericTable *>(a[1]);
    NumericTable * ntOutCentroids = const_cast<NumericTable *>(r[0]);
    NumericTable * ntAssignments  = const_cast<NumericTable *>(r[1]);
    NumericTable * ntObjFunction  = const_cast<NumericTable *>(r[2]);
    NumericTable * ntNIterations  = const_cast<NumericTable *>(r[3]);

    if (ntData->getNumberOfRows() > static_cast<size_t>(UINT_MAX) || ntData->getNumberOfColumns() > static_cast<size_t>(UINT_MAX)
        || par->maxIterations > static_cast<size_t>(UINT_MAX) || par->nClusters > static_cast<size_t>(UINT_MAX))
    {
        return Status(ErrorBufferSizeIntegerOverflow);
    }

    const uint32_t nIter     = static_cast<uint32_t>(par->maxIterations);
    const uint32_t nRows     = static_cast<uint32_t>(ntData->getNumberOfRows());
    const uint32_t nFeatures = static_cast<uint32_t>(ntData->getNumberOfColumns());
    const uint32_t nClusters = static_cast<uint32_t>(par->nClusters);

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    this->buildProgram(kernel_factory, nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);

    uint32_t blockSize = 0;
    DAAL_CHECK_STATUS_VAR(this->getBlockSize(nRows, nClusters, nFeatures, blockSize));
    DAAL_CHECK_STATUS_VAR(this->initializeBuffers(nClusters, nFeatures, blockSize));

    BlockDescriptor<algorithmFPType> inCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntInCentroids->getBlockOfRows(0, nClusters, readOnly, inCentroidsRows));
    auto inCentroids = inCentroidsRows.getBuffer();

    BlockDescriptor<algorithmFPType> outCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntOutCentroids->getBlockOfRows(0, nClusters, readWrite, outCentroidsRows));
    auto outCentroids = outCentroidsRows.getBuffer();

    BlockDescriptor<algorithmFPType> objFunctionRows;
    DAAL_CHECK_STATUS_VAR(ntObjFunction->getBlockOfRows(0, nClusters, readWrite, objFunctionRows));
    auto objFunction = objFunctionRows.getBuffer();
    if (!inCentroids || !outCentroids || !objFunction)
    {
        return Status(ErrorNullPtr);
    }

    algorithmFPType prevObjFunction = (algorithmFPType)0.0;

    uint32_t iter    = 0;
    uint32_t nBlocks = nRows / blockSize + int32_t(nRows % blockSize != 0);

    for (; iter < nIter; iter++)
    {
        bool needCandidates = true;
        for (uint32_t block = 0; block < nBlocks; block++)
        {
            auto range = Range::createFromBlock(block, blockSize, nRows);

            BlockDescriptor<algorithmFPType> dataRows;
            DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(range.startIndex, range.count, readOnly, dataRows));
            auto data = dataRows.getBuffer();
            BlockDescriptor<int> assignmentsRows;
            DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(range.startIndex, range.count, writeOnly, assignmentsRows));
            auto assignments = assignmentsRows.getBuffer();
            if (!data || !assignments)
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
            this->updateObjectiveFunction(objFunction, range.count, nClusters, int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);
            DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
            DAAL_CHECK_STATUS_VAR(ntAssignments->releaseBlockOfRows(assignmentsRows));
        }

        this->mergeReduceCentroids(outCentroids, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        algorithmFPType objFuncCorrection = 0.0;
        if (needCandidates)
        {
            DAAL_CHECK_STATUS_VAR(this->setEmptyClusters(ntData, nRows, nClusters, nFeatures, outCentroids, objFuncCorrection));
        }
        algorithmFPType curObjFunction = (algorithmFPType)0.0;
        {
            auto hostPtr = objFunction.toHost(data_management::readOnly);
            if (!hostPtr)
            {
                return Status(ErrorNullPtr);
            }

            curObjFunction = *hostPtr;
            curObjFunction -= objFuncCorrection;
        }

        if (par->accuracyThreshold > (algorithmFPType)0.0)
        {
            algorithmFPType objFuncDiff =
                curObjFunction - prevObjFunction > 0 ? curObjFunction - prevObjFunction : -(curObjFunction - prevObjFunction);
            if (objFuncDiff < par->accuracyThreshold)
            {
                iter++;
                break;
            }
        }
        prevObjFunction = curObjFunction;

        inCentroids = outCentroids;
    }
    for (uint32_t block = 0; block < nBlocks; block++)
    {
        auto range = Range::createFromBlock(block, blockSize, nRows);

        BlockDescriptor<algorithmFPType> dataRows;
        DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(range.startIndex, range.count, readOnly, dataRows));
        auto data = dataRows.getBuffer();

        BlockDescriptor<int> assignmentsRows;
        DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(range.startIndex, range.count, writeOnly, assignmentsRows));
        auto assignments = assignmentsRows.getBuffer();
        if (!data || !assignments)
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
        this->updateObjectiveFunction(objFunction, range.count, nClusters, int(block == 0), &st);
        DAAL_CHECK_STATUS_VAR(st);
        DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
        DAAL_CHECK_STATUS_VAR(ntAssignments->releaseBlockOfRows(assignmentsRows));
    }

    DAAL_CHECK_STATUS_VAR(ntInCentroids->releaseBlockOfRows(inCentroidsRows));
    DAAL_CHECK_STATUS_VAR(ntOutCentroids->releaseBlockOfRows(outCentroidsRows));
    DAAL_CHECK_STATUS_VAR(ntObjFunction->releaseBlockOfRows(objFunctionRows));
    {
        BlockDescriptor<int> nIterationsRows;
        DAAL_CHECK_STATUS_VAR(ntNIterations->getBlockOfRows(0, 1, writeOnly, nIterationsRows));
        auto nIterationsHostPtr = nIterationsRows.getBlockSharedPtr();
        int * nIterations       = nIterationsHostPtr.get();
        if (!nIterations)
        {
            return Status(ErrorNullPtr);
        }
        nIterations[0] = iter;
        DAAL_CHECK_STATUS_VAR(ntNIterations->releaseBlockOfRows(nIterationsRows));
    }

    return st;
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif

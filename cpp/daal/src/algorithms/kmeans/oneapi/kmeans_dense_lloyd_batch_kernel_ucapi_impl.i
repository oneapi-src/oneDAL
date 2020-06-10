/* file: kmeans_dense_lloyd_batch_kernel_ucapi_impl.i */
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

#ifndef __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_IMPL__
#define __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_IMPL__

#include "services/env_detect.h"
#include "src/algorithms/kmeans/oneapi/cl_kernels/kmeans_cl_kernels.cl"
#include "sycl/internal/execution_context.h"
#include "sycl/internal/types.h"
#include "src/sycl/blas_gpu.h"

#include "src/externals/service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(kmeans.dense.lloyd.batch.oneapi);

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
class Range
{
public:
    static Range createFromBlock(uint32_t blockIndex, uint32_t maxBlockSize, uint32_t sumOfBlocksSize)
    {
        const uint32_t startIndex = blockIndex * maxBlockSize;
        const uint32_t endIndex   = startIndex + maxBlockSize;
        return Range { startIndex, endIndex > sumOfBlocksSize ? sumOfBlocksSize : endIndex };
    }

    uint32_t startIndex;
    uint32_t endIndex;
    uint32_t count;

private:
    Range(uint32_t startIndex, uint32_t endIndex) : startIndex(startIndex), endIndex(endIndex), count(endIndex - startIndex) {}
};

template <typename algorithmFPType>
Status KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::initializeBuffers(uint32_t nClusters, uint32_t nFeatures, uint32_t blockSize)
{
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, blockSize, nClusters);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _nPartialCentroids, nClusters);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _nPartialCentroids * nClusters, nFeatures);
    uint32_t nCandidateParts = getCandidatePartNum(nClusters);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nCandidateParts, nClusters);
    Status st;
    auto & context = Environment::getInstance()->getDefaultExecutionContext();
    _dataSq        = context.allocate(TypeIds::id<algorithmFPType>(), blockSize, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _centroidsSq = context.allocate(TypeIds::id<algorithmFPType>(), nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _distances = context.allocate(TypeIds::id<algorithmFPType>(), blockSize * nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _mindistances = context.allocate(TypeIds::id<algorithmFPType>(), blockSize, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _candidates = context.allocate(TypeIds::id<int>(), nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _candidateDistances = context.allocate(TypeIds::id<algorithmFPType>(), nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _partialCandidates = context.allocate(TypeIds::id<int>(), nClusters * nCandidateParts, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _partialCandidateDistances = context.allocate(TypeIds::id<algorithmFPType>(), nClusters * nCandidateParts, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _partialCentroids = context.allocate(TypeIds::id<algorithmFPType>(), _nPartialCentroids * nClusters * nFeatures, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _partialCentroidsCounters = context.allocate(TypeIds::id<int>(), _nPartialCentroids * nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);
    _numEmptyClusters = context.allocate(TypeIds::id<int>(), 1, &st);
    DAAL_CHECK_STATUS_VAR(st);
    return Status();
}

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
    buildProgram(kernel_factory, nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);

    uint32_t blockSize = 0;
    DAAL_CHECK_STATUS_VAR(getBlockSize(nRows, nClusters, nFeatures, blockSize));
    DAAL_CHECK_STATUS_VAR(initializeBuffers(nClusters, nFeatures, blockSize));

    BlockDescriptor<algorithmFPType> inCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntInCentroids->getBlockOfRows(0, nClusters, readOnly, inCentroidsRows));
    auto inCentroids = inCentroidsRows.getBuffer();

    BlockDescriptor<algorithmFPType> outCentroidsRows;
    DAAL_CHECK_STATUS_VAR(ntOutCentroids->getBlockOfRows(0, nClusters, readWrite, outCentroidsRows));
    auto outCentroids = outCentroidsRows.getBuffer();

    BlockDescriptor<algorithmFPType> objFunctionRows;
    DAAL_CHECK_STATUS_VAR(ntObjFunction->getBlockOfRows(0, nClusters, readWrite, objFunctionRows));
    auto objFunction = objFunctionRows.getBuffer();
    if (inCentroids == NULL || outCentroids == NULL || objFunction == NULL)
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
            if (data == NULL || assignments == NULL)
            {
                return Status(ErrorNullPtr);
            }

            computeSquares(inCentroids, _centroidsSq, nClusters, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            computeDistances(data, inCentroids, range.count, nClusters, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            computeAssignments(assignments, range.count, nClusters, &st);
            DAAL_CHECK_STATUS_VAR(st);
            computeSquares(data, _dataSq, range.count, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            partialReduceCentroids(data, assignments, range.count, nClusters, nFeatures, int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);
            if (needCandidates)
            {
                getNumEmptyClusters(nClusters, &st);
                DAAL_CHECK_STATUS_VAR(st);
                int numEmpty = 0;
                {
                    auto num = _numEmptyClusters.get<int>().toHost(ReadWriteMode::readOnly);
                    if (num.get() == NULL)
                    {
                        return Status(ErrorNullPtr);
                    }

                    numEmpty = num.get()[0];
                }
                bool hasEmptyClusters = numEmpty > 0;
                if (hasEmptyClusters)
                {
                    computePartialCandidates(assignments, range.count, nClusters, int(block == 0), &st);
                    DAAL_CHECK_STATUS_VAR(st);
                    mergePartialCandidates(nClusters, &st);
                    DAAL_CHECK_STATUS_VAR(st);
                }
                needCandidates = hasEmptyClusters;
            }
            updateObjectiveFunction(assignments, objFunction, range.count, nClusters, int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);
            DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
            DAAL_CHECK_STATUS_VAR(ntAssignments->releaseBlockOfRows(assignmentsRows));
        }

        mergeReduceCentroids(outCentroids, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        algorithmFPType objFuncCorrection = 0.0;
        if (needCandidates)
        {
            DAAL_CHECK_STATUS_VAR(setEmptyClusters(ntData, nRows, nClusters, nFeatures, outCentroids, objFuncCorrection));
        }
        algorithmFPType curObjFunction = (algorithmFPType)0.0;
        {
            auto hostPtr = objFunction.toHost(data_management::readOnly);
            if (hostPtr == NULL)
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
        if (data == NULL || assignments == NULL)
        {
            return Status(ErrorNullPtr);
        }

        computeSquares(inCentroids, _centroidsSq, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        computeDistances(data, inCentroids, range.count, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        computeAssignments(assignments, range.count, nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        computeSquares(data, _dataSq, range.count, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        updateObjectiveFunction(assignments, objFunction, range.count, nClusters, int(block == 0), &st);
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
        if (nIterations == NULL)
        {
            return Status(ErrorNullPtr);
        }
        nIterations[0] = iter;
        DAAL_CHECK_STATUS_VAR(ntNIterations->releaseBlockOfRows(nIterationsRows));
    }

    return st;
}

template <typename algorithmFPType>
uint32_t KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getCandidatePartNum(uint32_t nClusters)
{
    return _maxLocalBuffer / nClusters / sizeof(algorithmFPType);
}
template <typename algorithmFPType>
services::String KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getBuildOptions(uint32_t nClusters)
{
    uint32_t numParts = getCandidatePartNum(nClusters);
    if (numParts > _preferableSubGroup) numParts = _preferableSubGroup;
    char buffer[DAAL_MAX_STRING_SIZE];
    services::String build_options;
    build_options.add("-cl-std=CL1.2 -D LOCAL_SUM_SIZE=");
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, _maxWorkItemsPerGroup);
    build_options.add(buffer);
    build_options.add(" -D CND_PART_SIZE=");
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, nClusters);
    build_options.add(buffer);
    build_options.add(" -D CND_PART_SIZE=");
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, nClusters);
    build_options.add(buffer);
    build_options.add(" -D NUM_PARTS_CND=");
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, numParts);
    build_options.add(buffer);
    return build_options;
}

template <typename algorithmFPType>
uint32_t KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getWorkgroupsCount(uint32_t rows)
{
    const uint32_t elementsPerGroup = _maxWorkItemsPerGroup;
    uint32_t workgroupsCount        = rows / elementsPerGroup;

    if (workgroupsCount * elementsPerGroup < rows) workgroupsCount++;

    return workgroupsCount;
}

template <typename algorithmFPType>
uint32_t KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getComputeSquaresWorkgroupsCount(uint32_t nFeatures)
{
    uint32_t workItemsPerGroup = nFeatures < _maxWorkItemsPerGroup ? nFeatures : _maxWorkItemsPerGroup;
    while (workItemsPerGroup & (workItemsPerGroup - 1))
    {
        workItemsPerGroup++;
    }
    if (nFeatures <= 32)
    {
        workItemsPerGroup = nFeatures;
    }
    else if (nFeatures <= 64)
    {
        workItemsPerGroup = nFeatures / 2;
        if (nFeatures % 2 > 0) workItemsPerGroup++;
    }
    else if (nFeatures <= 128)
    {
        workItemsPerGroup = nFeatures / 4;
        if (nFeatures % 4 > 0) workItemsPerGroup++;
    }
    return workItemsPerGroup;
}

template <typename algorithmFPType>
const char * KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getComputeSquaresKernelName(uint32_t nFeatures)
{
    if (nFeatures <= 32)
    {
        return "compute_squares_32";
    }
    else if (nFeatures <= 64)
    {
        return "compute_squares_64";
    }
    else if (nFeatures <= 128)
    {
        return "compute_squares_128";
    }
    return "compute_squares";
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::computeSquares(const Buffer<algorithmFPType> & data, UniversalBuffer & dataSq, uint32_t nRows,
                                                                       uint32_t nFeatures, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeSquares);

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    auto kernel           = kernel_factory.getKernel(getComputeSquaresKernelName(nFeatures), st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(4);
    args.set(0, data, AccessModeIds::read);
    args.set(1, dataSq, AccessModeIds::write);
    args.set(2, nRows);
    args.set(3, nFeatures);

    uint32_t workItemsPerGroup = getComputeSquaresWorkgroupsCount(nFeatures);

    KernelRange local_range(1, workItemsPerGroup);
    KernelRange global_range(nRows, workItemsPerGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel, args, st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getNumEmptyClusters(uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countEmptyClusters);
    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    auto kernel           = kernel_factory.getKernel("count_empty_clusters", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(4);
    args.set(0, _partialCentroidsCounters, AccessModeIds::read);
    args.set(1, nClusters);
    args.set(2, nClusters);
    args.set(3, _numEmptyClusters, AccessModeIds::write);

    KernelRange local_range(1, _maxWorkItemsPerGroup);
    KernelRange global_range(1, _maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel, args, st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::computeDistances(const Buffer<algorithmFPType> & data,
                                                                         const Buffer<algorithmFPType> & centroids, uint32_t blockSize,
                                                                         uint32_t nClusters, uint32_t nFeatures, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeDistances);

    auto gemmStatus = BlasGpu<algorithmFPType>::xgemm(math::Layout::ColMajor, math::Transpose::Trans, math::Transpose::NoTrans, blockSize, nClusters,
                                                      nFeatures, algorithmFPType(-1.0), data, nFeatures, 0, centroids, nFeatures, 0,
                                                      algorithmFPType(0.0), _distances.get<algorithmFPType>(), blockSize, 0);

    if (st != nullptr)
    {
        *st = gemmStatus;
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::computeAssignments(const Buffer<int> & assignments, uint32_t blockSize, uint32_t nClusters,
                                                                           Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeAssignments);

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    auto kernel           = kernel_factory.getKernel("reduce_assignments", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(6);
    args.set(0, _centroidsSq, AccessModeIds::read);
    args.set(1, _distances, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::write);
    args.set(3, _mindistances, AccessModeIds::write);
    args.set(4, blockSize);
    args.set(5, nClusters);

    KernelRange local_range(1, _preferableSubGroup);
    KernelRange global_range(blockSize, _preferableSubGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel, args, st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::computePartialCandidates(const Buffer<int> & assignments, uint32_t blockSize,
                                                                                 uint32_t nClusters, uint32_t reset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialCandidates);

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    auto kernel           = kernel_factory.getKernel("partial_candidates", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(10);
    args.set(0, assignments, AccessModeIds::read);
    args.set(1, _mindistances, AccessModeIds::read);
    args.set(2, _dataSq, AccessModeIds::read);
    args.set(3, _candidates, AccessModeIds::read);
    args.set(4, _candidateDistances, AccessModeIds::read);
    args.set(5, _partialCandidates, AccessModeIds::write);
    args.set(6, _partialCandidateDistances, AccessModeIds::write);
    args.set(7, blockSize);
    args.set(8, nClusters);
    args.set(9, reset);

    int num_parts = getCandidatePartNum(nClusters);
    if (num_parts > _preferableSubGroup) num_parts = _preferableSubGroup;
    KernelRange local_range(1, _preferableSubGroup);
    KernelRange global_range(num_parts, _preferableSubGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel, args, st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::mergePartialCandidates(uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergePartialCandidates);

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    auto kernel           = kernel_factory.getKernel("merge_candidates", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(5);
    args.set(0, _candidates, AccessModeIds::write);
    args.set(1, _candidateDistances, AccessModeIds::write);
    args.set(2, _partialCandidates, AccessModeIds::read);
    args.set(3, _partialCandidateDistances, AccessModeIds::read);
    args.set(4, (int)nClusters);

    int num_parts = getCandidatePartNum(nClusters);
    if (num_parts > _preferableSubGroup) num_parts = _preferableSubGroup;
    KernelRange local_range(1, num_parts);
    KernelRange global_range(1, num_parts);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel, args, st);
    DAAL_CHECK_STATUS_PTR(st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::partialReduceCentroids(const Buffer<algorithmFPType> & data, const Buffer<int> & assignments,
                                                                               uint32_t blockSize, uint32_t nClusters, uint32_t nFeatures,
                                                                               uint32_t doReset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partialReduceCentroids);

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    auto kernel           = kernel_factory.getKernel("partial_reduce_centroids", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(9);
    args.set(0, data, AccessModeIds::read);
    args.set(1, _distances, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::read);
    args.set(3, _partialCentroids, AccessModeIds::write);
    args.set(4, _partialCentroidsCounters, AccessModeIds::write);
    args.set(5, blockSize);
    args.set(6, nClusters);
    args.set(7, nFeatures);
    args.set(8, doReset);

    KernelRange global_range(_nPartialCentroids * nFeatures);
    context.run(global_range, kernel, args, st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::mergeReduceCentroids(const Buffer<algorithmFPType> & centroids, uint32_t nClusters,
                                                                             uint32_t nFeatures, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids);

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    auto kernel           = kernel_factory.getKernel("merge_reduce_centroids", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(6);
    args.set(0, _partialCentroids, AccessModeIds::readwrite);
    args.set(1, _partialCentroidsCounters, AccessModeIds::readwrite);
    args.set(2, centroids, AccessModeIds::write);
    args.set(3, nClusters);
    args.set(4, nFeatures);
    args.set(5, _nPartialCentroids);

    KernelRange local_range(_nPartialCentroids);
    KernelRange global_range(_nPartialCentroids * nClusters);

    KernelNDRange range(1);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel, args, st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::updateObjectiveFunction(const Buffer<int> & assignments,
                                                                                const Buffer<algorithmFPType> & objFunction, uint32_t blockSize,
                                                                                uint32_t nClusters, uint32_t doReset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateObjectiveFunction);

    if (doReset)
    {
        auto hostPtr = objFunction.toHost(data_management::writeOnly);
        if (hostPtr.get() == NULL)
        {
            if (st)
            {
                st->add(Status(ErrorNullPtr));
            }
            return;
        }
        *hostPtr = 0.0f;
    }

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    auto kernel           = kernel_factory.getKernel("update_objective_function", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(6);
    args.set(0, _dataSq, AccessModeIds::read);
    args.set(1, _mindistances, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::read);
    args.set(3, objFunction, AccessModeIds::write);
    args.set(4, (int)blockSize);
    args.set(5, (int)nClusters);

    KernelRange local_range(_maxWorkItemsPerGroup);
    KernelRange global_range(_maxWorkItemsPerGroup);

    KernelNDRange range(1);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel, args, st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernelFactory, uint32_t nClusters, Status * st)
{
    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add(getBuildOptions(nClusters));
    services::String cachekey("__daal_algorithms_kmeans_lloyd_dense_batch_");
    cachekey.add(build_options.c_str());
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), kmeans_cl_kernels, build_options.c_str(), st);
    }
}

template <typename algorithmFPType>
Status KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getBlockSize(uint32_t nRows, uint32_t nClusters, uint32_t nFeatures, uint32_t & blockSize)
{
    uint32_t gemmBlockSize = _nValuesInBlock;
    while (gemmBlockSize > _nValuesInBlock / nClusters)
    {
        gemmBlockSize >>= 1;
    }
    if (gemmBlockSize < _nMinRows)
    {
        return Status(ErrorKMeansNumberOfClustersIsTooLarge);
    }
    uint32_t datasetBlockSize = _nValuesInBlock;
    while (datasetBlockSize > _nValuesInBlock / nFeatures)
    {
        datasetBlockSize >>= 1;
    }
    if (datasetBlockSize < _nMinRows)
    {
        return Status(ErrorIncorrectNumberOfFeatures);
    }

    blockSize = datasetBlockSize > gemmBlockSize ? gemmBlockSize : datasetBlockSize;
    if (blockSize > nRows)
    {
        blockSize = nRows;
    }
    return Status();
}

template <typename algorithmFPType>
Status KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::setEmptyClusters(NumericTable * const ntData, uint32_t nRows, uint32_t nClusters,
                                                                           uint32_t nFeatures, Buffer<algorithmFPType> & outCentroids,
                                                                           algorithmFPType & objFuncCorrection)
{
    auto counters        = _partialCentroidsCounters.template get<int>().toHost(ReadWriteMode::readOnly);
    auto candidatesIds   = _candidates.template get<int>().toHost(ReadWriteMode::readOnly);
    auto candidatesDists = _candidateDistances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
    auto clusterFeatures = outCentroids.toHost(ReadWriteMode::readWrite);
    if (counters.get() == NULL || candidatesIds.get() == NULL || candidatesDists.get() == NULL || clusterFeatures.get() == NULL)
    {
        return Status(ErrorNullPtr);
    }

    int cPos = 0;
    for (int iCl = 0; iCl < nClusters; iCl++)
        if (counters.get()[iCl] == 0)
        {
            if (cPos >= nClusters) continue;
            int id = candidatesIds.get()[cPos];
            if (id < 0 || id >= nRows)
            {
                continue;
            }
            objFuncCorrection += candidatesDists.get()[cPos];
            BlockDescriptor<algorithmFPType> singleRow;
            DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(0, nRows, readOnly, singleRow));
            auto rowData = singleRow.getBlockPtr();
            if (rowData == NULL)
            {
                return Status(ErrorNullPtr);
            }
            for (int iFeature = 0; iFeature < nFeatures; iFeature++)
                clusterFeatures.get()[iCl * nFeatures + iFeature] = rowData[id * nFeatures + iFeature];
            cPos++;
            DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(singleRow));
        }
    return Status();
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif

/* file: kmeans_dense_lloyd_batch_kernel_ucapi_impl.i */
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
//  Implementation of K-means Batch Kernel for GPU.
//--
*/

#ifndef __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_IMPL__
#define __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_IMPL__

#include "env_detect.h"
#include "cl_kernels/kmeans_cl_kernels.cl"
#include "execution_context.h"
#include "oneapi/service_defines_oneapi.h"
#include "oneapi/internal/types.h"
#include "oneapi/blas_gpu.h"

#include "service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(kmeans.dense.lloyd.batch.oneapi);

using namespace daal::services;
using namespace daal::oneapi::internal;
using namespace daal::data_management;

inline char * utoa(uint32_t value, char * buffer, uint32_t buffer_size)
{
    uint32_t i = 0;
    while (value && i < buffer_size - 1)
    {
        size_t rem  = value % 10;
        buffer[i++] = 48 + rem;
        value /= 10;
    }

    for (uint32_t j = 0; j < i - j - 1; j++)
    {
        char tmp          = buffer[j];
        buffer[j]         = buffer[i - j - 1];
        buffer[i - j - 1] = tmp;
    }
    buffer[i] = 0;
    return buffer;
}

inline char * append(char * buffer, uint32_t & pos, uint32_t buffer_size, const char * append, uint32_t append_size)
{
    uint32_t i = 0;
    while (pos + i < buffer_size && i < append_size)
    {
        if (append[i] == 0) break;
        buffer[pos + i] = append[i];
        i++;
    }
    pos += i;
    return buffer;
}

inline uint32_t constStrLen(const char * s)
{
    uint32_t len = 0;
    while (s[len] != 0) len++;
    return len;
}

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

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    NumericTable * ntData         = const_cast<NumericTable *>(a[0]);
    NumericTable * ntInCentroids  = const_cast<NumericTable *>(a[1]);
    NumericTable * ntOutCentroids = const_cast<NumericTable *>(r[0]);
    NumericTable * ntAssignments  = const_cast<NumericTable *>(r[1]);
    NumericTable * ntObjFunction  = const_cast<NumericTable *>(r[2]);
    NumericTable * ntNIterations  = const_cast<NumericTable *>(r[3]);

    const size_t nIter     = par->maxIterations;
    const size_t nRows     = ntData->getNumberOfRows();
    const size_t nFeatures = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;

    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;

    build_options.add(getBuildOptions(nClusters));

    services::String cachekey("__daal_algorithms_kmeans_lloyd_dense_batch_");
    cachekey.add(fptype_name);
    cachekey.add(build_options.c_str());

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), kmeans_cl_kernels, build_options.c_str());
    }

    const size_t nPartialCentroids = 128;
    const size_t nValuesInBlock    = 1024 * 1024 * 1024 / sizeof(algorithmFPType);
    const size_t nMinRows          = 1;
    size_t gemmBlockSize           = nValuesInBlock;

    while (gemmBlockSize > nValuesInBlock / nClusters)
    {
        gemmBlockSize >>= 1;
    }

    if (gemmBlockSize < nMinRows)
    {
        return Status(ErrorKMeansNumberOfClustersIsTooLarge);
    }

    size_t datasetBlockSize = nValuesInBlock;
    while (datasetBlockSize > nValuesInBlock / nFeatures)
    {
        datasetBlockSize >>= 1;
    }

    if (datasetBlockSize < nMinRows)
    {
        return Status(ErrorIncorrectNumberOfFeatures);
    }

    size_t blockSize = datasetBlockSize > gemmBlockSize ? gemmBlockSize : datasetBlockSize;
    if (blockSize > nRows)
    {
        blockSize = nRows;
    }

    auto dataSq                    = context.allocate(TypeIds::id<algorithmFPType>(), blockSize, &st);
    auto centroidsSq               = context.allocate(TypeIds::id<algorithmFPType>(), nClusters, &st);
    auto distances                 = context.allocate(TypeIds::id<algorithmFPType>(), blockSize * nClusters, &st);
    auto mindistances              = context.allocate(TypeIds::id<algorithmFPType>(), blockSize, &st);
    auto candidates                = context.allocate(TypeIds::id<int>(), nClusters, &st);
    auto candidateDistances        = context.allocate(TypeIds::id<algorithmFPType>(), nClusters, &st);
    auto partialCandidates         = context.allocate(TypeIds::id<int>(), nClusters * getCandidatePartNum(nClusters), &st);
    auto partialCandidateDistances = context.allocate(TypeIds::id<algorithmFPType>(), nClusters * getCandidatePartNum(nClusters), &st);
    auto partialCentroids          = context.allocate(TypeIds::id<algorithmFPType>(), nPartialCentroids * nClusters * nFeatures, &st);
    auto partialCentroidsCounters  = context.allocate(TypeIds::id<int>(), nPartialCentroids * nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);

    auto compute_squares = kernel_factory.getKernel(getComputeSquaresKernelName(nFeatures), &st);
    DAAL_CHECK_STATUS_VAR(st);

    auto init_distances             = kernel_factory.getKernel("init_distances", &st);
    auto compute_assignments        = kernel_factory.getKernel("reduce_assignments", &st);
    auto partial_reduce_centroids   = kernel_factory.getKernel("partial_reduce_centroids", &st);
    auto merge_reduce_centroids     = kernel_factory.getKernel("merge_reduce_centroids", &st);
    auto update_objective_function  = kernel_factory.getKernel("update_objective_function", &st);
    auto compute_partial_candidates = kernel_factory.getKernel("partial_candidates", &st);
    auto merge_partial_candidates   = kernel_factory.getKernel("merge_candidates", &st);

    BlockDescriptor<algorithmFPType> inCentroidsRows;
    ntInCentroids->getBlockOfRows(0, nClusters, readOnly, inCentroidsRows);
    auto inCentroids = inCentroidsRows.getBuffer();

    BlockDescriptor<algorithmFPType> outCentroidsRows;
    ntOutCentroids->getBlockOfRows(0, nClusters, readWrite, outCentroidsRows);
    auto outCentroids = outCentroidsRows.getBuffer();

    BlockDescriptor<algorithmFPType> objFunctionRows;
    ntObjFunction->getBlockOfRows(0, nClusters, readWrite, objFunctionRows);
    auto objFunction = objFunctionRows.getBuffer();

    algorithmFPType prevObjFunction = (algorithmFPType)0.0;

    size_t iter    = 0;
    size_t nBlocks = nRows / blockSize + int(nRows % blockSize != 0);

    for (; iter < nIter; iter++)
    {
        for (size_t block = 0; block < nBlocks; block++)
        {
            size_t first = block * blockSize;
            size_t last  = first + blockSize;

            if (last > nRows)
            {
                last = nRows;
            }

            size_t curBlockSize = last - first;

            BlockDescriptor<algorithmFPType> dataRows;
            ntData->getBlockOfRows(first, curBlockSize, readOnly, dataRows);
            auto data = dataRows.getBuffer();

            BlockDescriptor<int> assignmentsRows;
            ntAssignments->getBlockOfRows(first, curBlockSize, writeOnly, assignmentsRows);
            auto assignments = assignmentsRows.getBuffer();

            computeSquares(context, compute_squares, inCentroids, centroidsSq, nClusters, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            initDistances(context, init_distances, centroidsSq, distances, curBlockSize, nClusters, &st);
            DAAL_CHECK_STATUS_VAR(st);
            computeDistances(context, data, inCentroids, distances, blockSize, nClusters, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            computeAssignments(context, compute_assignments, distances, assignments, mindistances, curBlockSize, nClusters, &st);
            DAAL_CHECK_STATUS_VAR(st);
            computeSquares(context, compute_squares, data, dataSq, curBlockSize, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            computePartialCandidates(context, compute_partial_candidates, assignments, mindistances, dataSq, candidates, candidateDistances,
                                     partialCandidates, partialCandidateDistances, curBlockSize, nClusters, int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);
            mergePartialCandidates(context, merge_partial_candidates, candidates, candidateDistances, partialCandidates, partialCandidateDistances,
                                   nClusters, &st);
            DAAL_CHECK_STATUS_VAR(st);
            partialReduceCentroids(context, partial_reduce_centroids, data, distances, assignments, partialCentroids, partialCentroidsCounters,
                                   curBlockSize, nClusters, nFeatures, nPartialCentroids, int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);
            updateObjectiveFunction(context, update_objective_function, dataSq, distances, assignments, objFunction, curBlockSize, nClusters,
                                    int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);

            ntData->releaseBlockOfRows(dataRows);
            ntAssignments->releaseBlockOfRows(assignmentsRows);
        }

        mergeReduceCentroids(context, merge_reduce_centroids, partialCentroids, partialCentroidsCounters, outCentroids, nClusters, nFeatures,
                             nPartialCentroids, &st);
        DAAL_CHECK_STATUS_VAR(st);
        auto counters                     = partialCentroidsCounters.template get<int>().toHost(ReadWriteMode::readOnly);
        auto candidatesIds                = candidates.get<int>().toHost(ReadWriteMode::readOnly);
        auto candidatesDists              = candidateDistances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        auto clusterFeatures              = outCentroids.toHost(ReadWriteMode::readWrite);
        algorithmFPType objFuncCorrection = 0.0;
        int cPos                          = 0;
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
                ntData->getBlockOfRows(0, blockSize, readOnly, singleRow);
                auto row_data = singleRow.getBlockPtr();
                for (int iFeature = 0; iFeature < nFeatures; iFeature++)
                    clusterFeatures.get()[iCl * nFeatures + iFeature] = row_data[id * nFeatures + iFeature];
                cPos++;
                ntData->releaseBlockOfRows(singleRow);
            }
        algorithmFPType curObjFunction = (algorithmFPType)0.0;
        {
            auto hostPtr   = objFunction.toHost(data_management::readOnly);
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
    for (size_t block = 0; block < nBlocks; block++)
    {
        size_t first = block * blockSize;
        size_t last  = first + blockSize;

        if (last > nRows)
        {
            last = nRows;
        }

        size_t curBlockSize = last - first;

        BlockDescriptor<algorithmFPType> dataRows;
        ntData->getBlockOfRows(first, curBlockSize, readOnly, dataRows);
        auto data = dataRows.getBuffer();

        BlockDescriptor<int> assignmentsRows;
        ntAssignments->getBlockOfRows(first, curBlockSize, writeOnly, assignmentsRows);
        auto assignments = assignmentsRows.getBuffer();

        computeSquares(context, compute_squares, inCentroids, centroidsSq, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        initDistances(context, init_distances, centroidsSq, distances, curBlockSize, nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        computeDistances(context, data, inCentroids, distances, blockSize, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        computeAssignments(context, compute_assignments, distances, assignments, mindistances, curBlockSize, nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        computeSquares(context, compute_squares, data, dataSq, curBlockSize, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        updateObjectiveFunction(context, update_objective_function, dataSq, distances, assignments, objFunction, curBlockSize, nClusters,
                                int(block == 0), &st);
        DAAL_CHECK_STATUS_VAR(st);
        DAAL_CHECK_STATUS_VAR(st);
    }

    ntInCentroids->releaseBlockOfRows(inCentroidsRows);
    ntOutCentroids->releaseBlockOfRows(outCentroidsRows);
    ntObjFunction->releaseBlockOfRows(objFunctionRows);

    {
        BlockDescriptor<int> nIterationsRows;
        ntNIterations->getBlockOfRows(0, 1, writeOnly, nIterationsRows);
        auto nIterationsHostPtr = nIterationsRows.getBlockSharedPtr();
        int * nIterations       = nIterationsHostPtr.get();
        nIterations[0]          = iter;
        ntNIterations->releaseBlockOfRows(nIterationsRows);
    }

    return st;
}

template <typename algorithmFPType>
uint32_t KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getCandidatePartNum(uint32_t nClusters)
{
    return _maxLocalBuffer / nClusters / sizeof(algorithmFPType);
}
template <typename algorithmFPType>
const char * KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::getBuildOptions(uint32_t nClusters)
{
    const uint32_t bufSize    = 1024;
    const uint32_t valBufSize = 16;
    static char buffer[bufSize];
    static char valBuffer[valBufSize];
    uint32_t numParts = getCandidatePartNum(nClusters);
    if (numParts > _preferableSubGroup) numParts = _preferableSubGroup;
    const char * s1 = "-cl-std=CL1.2 -D LOCAL_SUM_SIZE=";
    const char * s2 = " -D CND_PART_SIZE=";
    const char * s3 = " -D NUM_PARTS_CND=";
    uint32_t pos    = 0;
    append(buffer, pos, bufSize, s1, constStrLen(s1));
    append(buffer, pos, bufSize, utoa(_maxWorkItemsPerGroup, valBuffer, valBufSize), valBufSize);
    append(buffer, pos, bufSize, s2, constStrLen(s2));
    append(buffer, pos, bufSize, utoa(nClusters, valBuffer, valBufSize), valBufSize);
    append(buffer, pos, bufSize, s3, constStrLen(s3));
    append(buffer, pos, bufSize, utoa(numParts, valBuffer, valBufSize), valBufSize);
    buffer[pos] = 0;
    return buffer;
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
    size_t workItemsPerGroup = nFeatures < _maxWorkItemsPerGroup ? nFeatures : _maxWorkItemsPerGroup;
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
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::computeSquares(ExecutionContextIface & context, const KernelPtr & kernel_compute_squares,
                                                                       const Buffer<algorithmFPType> & data, UniversalBuffer & dataSq, uint32_t nRows,
                                                                       uint32_t nFeatures, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeSquares);

    KernelArguments args(4);
    args.set(0, data, AccessModeIds::read);
    args.set(1, dataSq, AccessModeIds::write);
    args.set(2, nRows);
    args.set(3, nFeatures);

    size_t workItemsPerGroup = getComputeSquaresWorkgroupsCount(nFeatures);

    KernelRange local_range(1, workItemsPerGroup);
    KernelRange global_range(nRows, workItemsPerGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeSquares.run);
        context.run(range, kernel_compute_squares, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::initDistances(ExecutionContextIface & context, const KernelPtr & kernel_init_distances,
                                                                      UniversalBuffer & centroidsSq, UniversalBuffer & distances, uint32_t blockSize,
                                                                      uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initDistances);

    KernelArguments args(4);
    args.set(0, centroidsSq, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::write);
    args.set(2, blockSize);
    args.set(3, nClusters);

    size_t workgroupsCount = getWorkgroupsCount(blockSize);

    KernelRange local_range(_maxWorkItemsPerGroup, 1);
    KernelRange global_range(workgroupsCount * _maxWorkItemsPerGroup, nClusters);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.initDistances.run);
        context.run(range, kernel_init_distances, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::computeDistances(ExecutionContextIface & context, const Buffer<algorithmFPType> & data,
                                                                         const Buffer<algorithmFPType> & centroids, UniversalBuffer & distances,
                                                                         uint32_t blockSize, uint32_t nClusters, uint32_t nFeatures, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeDistances);

    auto gemmStatus = BlasGpu<algorithmFPType>::xgemm(math::Layout::ColMajor, math::Transpose::Trans, math::Transpose::NoTrans, blockSize, nClusters,
                                                      nFeatures, algorithmFPType(-1.0), data, nFeatures, 0, centroids, nFeatures, 0,
                                                      algorithmFPType(1.0), distances.get<algorithmFPType>(), blockSize, 0);

    if (st != nullptr)
    {
        *st = gemmStatus;
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::computeAssignments(ExecutionContextIface & context,
                                                                           const KernelPtr & kernel_compute_assignments, UniversalBuffer & distances,
                                                                           const Buffer<int> & assignments, UniversalBuffer & mindistances,
                                                                           uint32_t blockSize, uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeAssignments);

    KernelArguments args(5);
    args.set(0, distances, AccessModeIds::read);
    args.set(1, assignments, AccessModeIds::write);
    args.set(2, mindistances, AccessModeIds::write);
    args.set(3, blockSize);
    args.set(4, nClusters);

    KernelRange local_range(1, _preferableSubGroup);
    KernelRange global_range(blockSize, _preferableSubGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeAssignments.run);
        context.run(range, kernel_compute_assignments, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::computePartialCandidates(
    ExecutionContextIface & context, const KernelPtr & kernel_partial_candidates, const Buffer<int> & assignments, UniversalBuffer & mindistances,
    UniversalBuffer & dataSq, UniversalBuffer & candidates, UniversalBuffer & candidateDistances, UniversalBuffer & partialCandidates,
    UniversalBuffer & partialCandidateDistances, uint32_t blockSize, uint32_t nClusters, uint32_t reset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialCandidates);

    KernelArguments args(10);
    args.set(0, assignments, AccessModeIds::read);
    args.set(1, mindistances, AccessModeIds::read);
    args.set(2, dataSq, AccessModeIds::read);
    args.set(3, candidates, AccessModeIds::read);
    args.set(4, candidateDistances, AccessModeIds::read);
    args.set(5, partialCandidates, AccessModeIds::write);
    args.set(6, partialCandidateDistances, AccessModeIds::write);
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

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialCandidates.run);
        context.run(range, kernel_partial_candidates, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::mergePartialCandidates(
    ExecutionContextIface & context, const KernelPtr & kernel_merge_candidates, UniversalBuffer & candidates, UniversalBuffer & candidateDistances,
    UniversalBuffer & partialCandidates, UniversalBuffer & partialCandidateDistances, uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergePartialCandidates);

    KernelArguments args(5);
    args.set(0, candidates, AccessModeIds::write);
    args.set(1, candidateDistances, AccessModeIds::write);
    args.set(2, partialCandidates, AccessModeIds::read);
    args.set(3, partialCandidateDistances, AccessModeIds::read);
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

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergePartialCandidates.run);
        context.run(range, kernel_merge_candidates, args, st);
    }
    DAAL_CHECK_STATUS_PTR(st);
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::partialReduceCentroids(
    ExecutionContextIface & context, const KernelPtr & kernel_partial_reduce_centroids, const Buffer<algorithmFPType> & data,
    UniversalBuffer & distances, const Buffer<int> & assignments, UniversalBuffer & partialCentroids, UniversalBuffer & partialCentroidsCounters,
    uint32_t blockSize, uint32_t nClusters, uint32_t nFeatures, uint32_t nPartialCentroids, uint32_t doReset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partialReduceCentroids);

    KernelArguments args(9);
    args.set(0, data, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::read);
    args.set(3, partialCentroids, AccessModeIds::write);
    args.set(4, partialCentroidsCounters, AccessModeIds::write);
    args.set(5, blockSize);
    args.set(6, nClusters);
    args.set(7, nFeatures);
    args.set(8, doReset);

    KernelRange global_range(nPartialCentroids * nFeatures);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.partialReduce.run);
        context.run(global_range, kernel_partial_reduce_centroids, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::mergeReduceCentroids(ExecutionContextIface & context,
                                                                             const KernelPtr & kernel_merge_reduce_centroids,
                                                                             UniversalBuffer & partialCentroids,
                                                                             UniversalBuffer & partialCentroidsCounters,
                                                                             const Buffer<algorithmFPType> & centroids, uint32_t nClusters,
                                                                             uint32_t nFeatures, uint32_t nPartialCentroids, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids);

    KernelArguments args(6);
    args.set(0, partialCentroids, AccessModeIds::readwrite);
    args.set(1, partialCentroidsCounters, AccessModeIds::readwrite);
    args.set(2, centroids, AccessModeIds::write);
    args.set(3, nClusters);
    args.set(4, nFeatures);
    args.set(5, nPartialCentroids);

    KernelRange local_range(nPartialCentroids);
    KernelRange global_range(nPartialCentroids * nClusters);

    KernelNDRange range(1);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids.run);
        context.run(range, kernel_merge_reduce_centroids, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::updateObjectiveFunction(ExecutionContextIface & context,
                                                                                const KernelPtr & kernel_update_objective_function,
                                                                                UniversalBuffer & dataSq, UniversalBuffer & distances,
                                                                                const Buffer<int> & assignments,
                                                                                const Buffer<algorithmFPType> & objFunction, uint32_t blockSize,
                                                                                uint32_t nClusters, uint32_t doReset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateObjectiveFunction);

    if (doReset)
    {
        auto hostPtr = objFunction.toHost(data_management::writeOnly);
        *hostPtr     = 0.0f;
    }

    KernelArguments args(6);
    args.set(0, dataSq, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::read);
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

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateObjectiveFunction.run);
        context.run(range, kernel_update_objective_function, args, st);
    }
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif

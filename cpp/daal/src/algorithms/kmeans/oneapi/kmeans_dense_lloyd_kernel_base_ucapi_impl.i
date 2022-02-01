/* file: kmeans_dense_lloyd_kernel_base_ucapi_impl.i */
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
//  Implementation of K-means Base Kernel for GPU.
//--
*/

#ifndef __KMEANS_DENSE_LLOYD_KERNEL_BASE_UCAPI_IMPL__
#define __KMEANS_DENSE_LLOYD_KERNEL_BASE_UCAPI_IMPL__

#include "services/env_detect.h"
#include "src/algorithms/kmeans/oneapi/cl_kernels/kmeans_cl_kernels.cl"
#include "services/internal/sycl/execution_context.h"
#include "services/internal/sycl/types.h"
#include "src/services/service_data_utils.h"
#include "src/sycl/blas_gpu.h"

#include "src/externals/service_profiler.h"

using namespace daal::services;
using namespace daal::services::internal::sycl;
using namespace daal::data_management;

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
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::initializeBuffers(uint32_t nClusters, uint32_t nFeatures, uint32_t blockSize)
{
    DAAL_ASSERT(_nPartialCentroids <= maxInt32AsUint32T);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, blockSize, nClusters);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _nPartialCentroids, nClusters);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _nPartialCentroids * nClusters, nFeatures);
    uint32_t nCandidateParts = getCandidatePartNum(nClusters);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nCandidateParts, nClusters);
    Status st;
    auto & context = Environment::getInstance()->getDefaultExecutionContext();
    _distances     = context.allocate(TypeIds::id<algorithmFPType>(), blockSize * nClusters, st);
    DAAL_CHECK_STATUS_VAR(st);
    _mindistances = context.allocate(TypeIds::id<algorithmFPType>(), blockSize, st);
    DAAL_CHECK_STATUS_VAR(st);
    _candidates = context.allocate(TypeIds::id<int>(), nClusters, st);
    DAAL_CHECK_STATUS_VAR(st);
    _candidateDistances = context.allocate(TypeIds::id<algorithmFPType>(), nClusters, st);
    DAAL_CHECK_STATUS_VAR(st);
    _partialCandidates = context.allocate(TypeIds::id<int>(), nClusters * nCandidateParts, st);
    DAAL_CHECK_STATUS_VAR(st);
    _partialCandidateDistances = context.allocate(TypeIds::id<algorithmFPType>(), nClusters * nCandidateParts, st);
    DAAL_CHECK_STATUS_VAR(st);
    _partialCentroids = context.allocate(TypeIds::id<algorithmFPType>(), _nPartialCentroids * nClusters * nFeatures, st);
    DAAL_CHECK_STATUS_VAR(st);
    _partialCentroidsCounters = context.allocate(TypeIds::id<int>(), _nPartialCentroids * nClusters, st);
    DAAL_CHECK_STATUS_VAR(st);
    _numEmptyClusters = context.allocate(TypeIds::id<int>(), 1, st);
    DAAL_CHECK_STATUS_VAR(st);
    return Status();
}

template <typename algorithmFPType>
uint32_t KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getCandidatePartNum(uint32_t nClusters)
{
    DAAL_ASSERT(_maxLocalBuffer / nClusters / sizeof(algorithmFPType) > 0);
    return _maxLocalBuffer / nClusters / sizeof(algorithmFPType);
}
template <typename algorithmFPType>
services::String KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getBuildOptions(uint32_t nClusters)
{
    uint32_t numParts = getCandidatePartNum(nClusters);
    if (numParts > _preferableSubGroup) numParts = _preferableSubGroup;
    char buffer[DAAL_MAX_STRING_SIZE];
    services::String buildOptions;
    buildOptions.add("-cl-std=CL1.2 -D LOCAL_SUM_SIZE=");
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, _maxWorkItemsPerGroup);
    buildOptions.add(buffer);
    buildOptions.add(" -D CND_PART_SIZE=");
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, nClusters);
    buildOptions.add(buffer);
    buildOptions.add(" -D NUM_PARTS_CND=");
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, numParts);
    buildOptions.add(buffer);
    return buildOptions;
}

template <typename algorithmFPType>
uint32_t KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getWorkgroupsCount(uint32_t rows)
{
    const uint32_t elementsPerGroup = _maxWorkItemsPerGroup;
    uint32_t workgroupsCount        = rows / elementsPerGroup;

    if (workgroupsCount * elementsPerGroup < rows) workgroupsCount++;

    return workgroupsCount;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::computeSquares(const services::internal::Buffer<algorithmFPType> & data,
                                                                        math::SumReducer::Result & result, UniversalBuffer & dataSq, uint32_t nRows,
                                                                        uint32_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeSquares);
    DAAL_ASSERT(data.size() >= nRows * nFeatures);
    DAAL_ASSERT(nRows <= maxInt32AsUint32T);
    DAAL_ASSERT(nFeatures <= maxInt32AsUint32T);
    Status st;
    dataSq = math::SumReducer::sum(math::Layout::RowMajor, data, nRows, nFeatures, result, st).sumOfSquares;
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getNumEmptyClusters(uint32_t nClusters)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countEmptyClusters);
    Status st;
    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory, nClusters));
    auto kernel = kernelFactory.getKernel("count_empty_clusters", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCentroidsCounters, int, _nPartialCentroids * nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_numEmptyClusters, int, 1);
    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);

    KernelArguments args(4, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, _partialCentroidsCounters, AccessModeIds::read);
    args.set(1, static_cast<int32_t>(nClusters));
    args.set(2, static_cast<int32_t>(_nPartialCentroids));
    args.set(3, _numEmptyClusters, AccessModeIds::write);

    KernelRange local_range(1, _maxWorkItemsPerGroup);
    KernelRange global_range(1, _maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::computeDistances(const services::internal::Buffer<algorithmFPType> & data,
                                                                          const services::internal::Buffer<algorithmFPType> & centroids,
                                                                          uint32_t blockSize, uint32_t nClusters, uint32_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeDistances);
    DAAL_ASSERT(data.size() >= blockSize * nFeatures);
    DAAL_ASSERT(centroids.size() >= nClusters * nFeatures);
    Status st = BlasGpu<algorithmFPType>::xgemm(math::Layout::ColMajor, math::Transpose::Trans, math::Transpose::NoTrans, blockSize, nClusters,
                                                nFeatures, algorithmFPType(-1.0), data, nFeatures, 0, centroids, nFeatures, 0, algorithmFPType(0.0),
                                                _distances.get<algorithmFPType>(), blockSize, 0);
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::computeAssignments(const UniversalBuffer & assignments, uint32_t blockSize,
                                                                            uint32_t nClusters)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeAssignments);
    Status st;
    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory, nClusters));
    auto kernel = kernelFactory.getKernel("reduce_assignments", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT_UNIVERSAL_BUFFER(_centroidsSq, algorithmFPType, nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_distances, algorithmFPType, blockSize * nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(assignments, int, blockSize);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_mindistances, algorithmFPType, blockSize);

    DAAL_ASSERT(blockSize <= maxInt32AsUint32T);
    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);

    KernelArguments args(7, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, _centroidsSq, AccessModeIds::read);
    args.set(1, _distances, AccessModeIds::read);
    args.set(2, static_cast<int32_t>(blockSize));
    args.set(3, static_cast<int32_t>(nClusters));
    if (TypeIds::id<algorithmFPType>() == TypeIds::float32)
    {
        args.set(4, FLT_MAX);
    }
    else
    {
        args.set(4, DBL_MAX);
    }
    args.set(5, assignments, AccessModeIds::write);
    args.set(6, _mindistances, AccessModeIds::write);

    KernelRange local_range(1, _preferableSubGroup);
    KernelRange global_range(blockSize, _preferableSubGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::computePartialCandidates(const UniversalBuffer & assignments, uint32_t blockSize,
                                                                                  uint32_t nClusters, uint32_t reset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialCandidates);
    Status st;
    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory, nClusters));
    auto kernel = kernelFactory.getKernel("partial_candidates", st);
    DAAL_CHECK_STATUS_VAR(st);

    int numParts = getCandidatePartNum(nClusters);
    if (numParts > _preferableSubGroup) numParts = _preferableSubGroup;
    DAAL_ASSERT_UNIVERSAL_BUFFER(assignments, int, blockSize);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_mindistances, algorithmFPType, blockSize);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_dataSq, algorithmFPType, blockSize);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_candidates, int, nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_candidateDistances, algorithmFPType, nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCandidates, int, nClusters * numParts);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCandidateDistances, algorithmFPType, nClusters * numParts);

    DAAL_ASSERT(blockSize <= maxInt32AsUint32T);
    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);
    DAAL_ASSERT(reset <= maxInt32AsUint32T);

    KernelArguments args(10, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, assignments, AccessModeIds::read);
    args.set(1, _mindistances, AccessModeIds::read);
    args.set(2, _dataSq, AccessModeIds::read);
    args.set(3, _candidates, AccessModeIds::read);
    args.set(4, _candidateDistances, AccessModeIds::read);
    args.set(5, _partialCandidates, AccessModeIds::write);
    args.set(6, _partialCandidateDistances, AccessModeIds::write);
    args.set(7, static_cast<int32_t>(blockSize));
    args.set(8, static_cast<int32_t>(nClusters));
    args.set(9, static_cast<int32_t>(reset));

    KernelRange local_range(1, _preferableSubGroup);
    KernelRange global_range(numParts, _preferableSubGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::mergePartialCandidates(uint32_t nClusters)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergePartialCandidates);
    Status st;
    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory, nClusters));
    auto kernel = kernelFactory.getKernel("merge_candidates", st);
    DAAL_CHECK_STATUS_VAR(st);

    int numParts = getCandidatePartNum(nClusters);
    if (numParts > _preferableSubGroup) numParts = _preferableSubGroup;
    DAAL_ASSERT_UNIVERSAL_BUFFER(_candidates, int, nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_candidateDistances, algorithmFPType, nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCandidates, int, numParts * nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCandidateDistances, algorithmFPType, numParts * nClusters);

    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);

    KernelArguments args(5, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, _candidates, AccessModeIds::write);
    args.set(1, _candidateDistances, AccessModeIds::write);
    args.set(2, _partialCandidates, AccessModeIds::read);
    args.set(3, _partialCandidateDistances, AccessModeIds::read);
    args.set(4, static_cast<int32_t>(nClusters));

    KernelRange local_range(1, numParts);
    KernelRange global_range(1, numParts);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    context.run(range, kernel, args, st);
    DAAL_CHECK_STATUS_VAR(st);
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::partialReduceCentroids(const services::internal::Buffer<algorithmFPType> & data,
                                                                                const UniversalBuffer & assignments, uint32_t blockSize,
                                                                                uint32_t nClusters, uint32_t nFeatures, uint32_t doReset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partialReduceCentroids);
    Status st;
    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory, nClusters));
    auto kernel = kernelFactory.getKernel("partial_reduce_centroids", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT(data.size() >= blockSize * nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_distances, algorithmFPType, blockSize * nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(assignments, int, blockSize);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCentroids, algorithmFPType, _nPartialCentroids * nClusters * nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCentroidsCounters, int, _nPartialCentroids * nClusters);

    DAAL_ASSERT(blockSize <= maxInt32AsUint32T);
    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);
    DAAL_ASSERT(nFeatures <= maxInt32AsUint32T);
    DAAL_ASSERT(doReset <= maxInt32AsUint32T);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _nPartialCentroids * nClusters, nFeatures);

    KernelArguments args(9, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, data, AccessModeIds::read);
    args.set(1, _distances, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::read);
    args.set(3, _partialCentroids, AccessModeIds::write);
    args.set(4, _partialCentroidsCounters, AccessModeIds::write);
    args.set(5, static_cast<int32_t>(blockSize));
    args.set(6, static_cast<int32_t>(nClusters));
    args.set(7, static_cast<int32_t>(nFeatures));
    args.set(8, static_cast<int32_t>(doReset));

    KernelRange global_range(_nPartialCentroids * nFeatures);
    context.run(global_range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::mergeReduceCentroids(const services::internal::Buffer<algorithmFPType> & centroids,
                                                                              uint32_t nClusters, uint32_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids);
    Status st;
    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory, nClusters));
    auto kernel = kernelFactory.getKernel("merge_reduce_centroids", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT(centroids.size() >= nClusters * nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCentroids, algorithmFPType, _nPartialCentroids * nClusters * nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCentroidsCounters, int, _nPartialCentroids * nClusters);

    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);
    DAAL_ASSERT(nFeatures <= maxInt32AsUint32T);

    KernelArguments args(6, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, _partialCentroids, AccessModeIds::readwrite);
    args.set(1, _partialCentroidsCounters, AccessModeIds::readwrite);
    args.set(2, centroids, AccessModeIds::write);
    args.set(3, static_cast<int32_t>(nClusters));
    args.set(4, static_cast<int32_t>(nFeatures));
    args.set(5, static_cast<int32_t>(_nPartialCentroids));

    KernelRange local_range(_nPartialCentroids);
    KernelRange global_range(_nPartialCentroids * nClusters);

    KernelNDRange range(1);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::updateObjectiveFunction(const services::internal::Buffer<algorithmFPType> & objFunction,
                                                                                 uint32_t blockSize, uint32_t nClusters, uint32_t doReset)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateObjectiveFunction);
    Status st;
    if (doReset)
    {
        auto hostPtr = objFunction.toHost(data_management::writeOnly, st);
        DAAL_CHECK_STATUS_VAR(st);
        *hostPtr = 0.0f;
    }

    auto & context       = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(this->buildProgram(kernelFactory, nClusters));
    auto kernel = kernelFactory.getKernel("update_objective_function", st);
    DAAL_CHECK_STATUS_VAR(st);

    DAAL_ASSERT(objFunction.size() >= 1);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_dataSq, algorithmFPType, blockSize);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_mindistances, algorithmFPType, blockSize);

    DAAL_ASSERT(blockSize <= maxInt32AsUint32T);
    DAAL_ASSERT(nClusters <= maxInt32AsUint32T);

    KernelArguments args(5, st);
    DAAL_CHECK_STATUS_VAR(st);
    args.set(0, _dataSq, AccessModeIds::read);
    args.set(1, _mindistances, AccessModeIds::read);
    args.set(2, blockSize);
    args.set(3, nClusters);
    args.set(4, objFunction, AccessModeIds::readwrite);

    KernelRange local_range(_maxWorkItemsPerGroup);
    KernelRange global_range(_maxWorkItemsPerGroup);

    KernelNDRange range(1);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_VAR(st);
    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernelFactory, uint32_t nClusters)
{
    auto fptypeName   = services::internal::sycl::getKeyFPType<algorithmFPType>();
    auto buildOptions = fptypeName;
    buildOptions.add(getBuildOptions(nClusters));
    services::String cachekey("__daal_algorithms_kmeans_lloyd_dense_batch_");
    cachekey.add(buildOptions.c_str());

    Status st;
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), kmeans_cl_kernels, buildOptions.c_str(), st);
    }
    return st;
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getBlockSize(uint32_t nRows, uint32_t nClusters, uint32_t nFeatures, uint32_t & blockSize)
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
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::fitPartialCentroidSize(uint32_t nClusters, uint32_t nFeatures)
{
    while (_nPartialCentroids * nClusters * nFeatures > _nValuesInBlock)
    {
        _nPartialCentroids >>= 1;
    }
    if (_nPartialCentroids < _nMinRows)
    {
        return Status(ErrorKMeansNumberOfClustersIsTooLarge);
    }
    return Status();
}

template <typename algorithmFPType>
Status KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::setEmptyClusters(NumericTable * const ntData, uint32_t nRows, uint32_t nClusters,
                                                                          uint32_t nFeatures,
                                                                          services::internal::Buffer<algorithmFPType> & outCentroids,
                                                                          algorithmFPType & objFuncCorrection)
{
    services::Status status;
    DAAL_ASSERT(outCentroids.size() >= nClusters * nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_partialCentroidsCounters, int, nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_candidates, int, nClusters);
    DAAL_ASSERT_UNIVERSAL_BUFFER(_candidateDistances, algorithmFPType, nClusters);

    auto counters        = _partialCentroidsCounters.template get<int>().toHost(ReadWriteMode::readOnly, status);
    auto candidatesIds   = _candidates.template get<int>().toHost(ReadWriteMode::readOnly, status);
    auto candidatesDists = _candidateDistances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly, status);
    auto clusterFeatures = outCentroids.toHost(ReadWriteMode::readWrite, status);
    DAAL_CHECK_STATUS_VAR(status);

    uint32_t cPos = 0;
    for (uint32_t iCl = 0; iCl < nClusters; iCl++)
        if (counters.get()[iCl] == 0)
        {
            if (cPos >= nClusters)
            {
                continue;
            }
            int id = candidatesIds.get()[cPos];
            if (id < 0 || id >= nRows)
            {
                continue;
            }
            objFuncCorrection += candidatesDists.get()[cPos];
            BlockDescriptor<algorithmFPType> singleRow;
            DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(0, nRows, readOnly, singleRow));
            auto rowData = singleRow.getBlockPtr();
            if (!rowData)
            {
                return Status(ErrorNullPtr);
            }
            for (uint32_t iFeature = 0; iFeature < nFeatures; iFeature++)
                clusterFeatures.get()[iCl * nFeatures + iFeature] = rowData[id * nFeatures + iFeature];
            cPos++;
            DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(singleRow));
        }
    return status;
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif

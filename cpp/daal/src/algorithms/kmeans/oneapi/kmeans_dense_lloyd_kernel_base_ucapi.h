/* file: kmeans_dense_lloyd_kernel_base_ucapi.h */
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
//  Implementation of K-means BASE Batch Kernel for GPU.
//--
*/

#ifndef __KMEANS_DENSE_LLOYD_KERNEL_BASE_UCAPI_H__
#define __KMEANS_DENSE_LLOYD_KERNEL_BASE_UCAPI_H__

#include "sycl/internal/types.h"
#include "sycl/internal/execution_context.h"
#include "algorithms/kmeans/kmeans_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

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
class KMeansDenseLloydKernelBaseUCAPI : public Kernel
{
protected:
    void computeSquares(const services::Buffer<algorithmFPType> & data, oneapi::internal::UniversalBuffer & dataSq, uint32_t nRows,
                        uint32_t nFeatures, services::Status * st);

    void computeDistances(const services::Buffer<algorithmFPType> & data, const services::Buffer<algorithmFPType> & centroids, uint32_t blockSize,
                          uint32_t nClusters, uint32_t nFeatures, services::Status * st);

    void computeAssignments(const oneapi::internal::UniversalBuffer & assignments, uint32_t blockSize, uint32_t nClusters, services::Status * st);

    void computePartialCandidates(const oneapi::internal::UniversalBuffer & assignments, uint32_t blockSize, uint32_t nClusters, uint32_t reset,
                                  services::Status * st);

    void mergePartialCandidates(uint32_t nClusters, services::Status * st);

    void partialReduceCentroids(const services::Buffer<algorithmFPType> & data, const oneapi::internal::UniversalBuffer & assignments,
                                uint32_t blockSize, uint32_t nClusters, uint32_t nFeatures, uint32_t doReset, services::Status * st);

    void mergeReduceCentroids(const services::Buffer<algorithmFPType> & centroids, uint32_t nClusters, uint32_t nFeatures, services::Status * st);

    void updateObjectiveFunction(const services::Buffer<algorithmFPType> & objFunction, uint32_t blockSize, uint32_t nClusters, uint32_t doReset,
                                 services::Status * st);
    void getNumEmptyClusters(uint32_t nClusters, services::Status * st);
    void buildProgram(oneapi::internal::ClKernelFactoryIface & kernelFactory, uint32_t nClusters, daal::services::Status * st);
    services::Status setEmptyClusters(NumericTable * const ntData, uint32_t nRows, uint32_t nClusters, uint32_t nFeatures,
                                      services::Buffer<algorithmFPType> & outCentroids, algorithmFPType & objFuncCorrection);
    services::Status initializeBuffers(uint32_t nClusters, uint32_t nFeatures, uint32_t blockSize);
    services::Status getBlockSize(uint32_t nRows, uint32_t nClusters, uint32_t nFeatures, uint32_t & blockSize);
    uint32_t getCandidatePartNum(uint32_t nClusters);
    uint32_t getWorkgroupsCount(uint32_t rows);
    uint32_t getComputeSquaresWorkgroupsCount(uint32_t nFeatures);
    const char * getComputeSquaresKernelName(uint32_t nFeatures);
    services::String getBuildOptions(uint32_t nClusters);

    oneapi::internal::UniversalBuffer _dataSq;
    oneapi::internal::UniversalBuffer _centroidsSq;
    oneapi::internal::UniversalBuffer _distances;
    oneapi::internal::UniversalBuffer _mindistances;
    oneapi::internal::UniversalBuffer _candidates;
    oneapi::internal::UniversalBuffer _candidateDistances;
    oneapi::internal::UniversalBuffer _partialCandidates;
    oneapi::internal::UniversalBuffer _partialCandidateDistances;
    oneapi::internal::UniversalBuffer _partialCentroids;
    oneapi::internal::UniversalBuffer _partialCentroidsCounters;
    oneapi::internal::UniversalBuffer _numEmptyClusters;

    const uint32_t _maxWorkItemsPerGroup = 128;                                          // should be a power of two for interal needs
    const uint32_t _maxLocalBuffer       = 30000;                                        // should be less than a half of local memory (two buffers)
    const uint32_t _preferableSubGroup   = 16;                                           // preferable maximal sub-group size
    const uint32_t _nPartialCentroids    = 128;                                          // Recommended number of partial centroids
    const uint32_t _nValuesInBlock       = 1024 * 1024 * 1024 / sizeof(algorithmFPType); // Max block size is 1GB
    const uint32_t _nMinRows             = 1;                                            // At least a single row should fit into block
};

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif

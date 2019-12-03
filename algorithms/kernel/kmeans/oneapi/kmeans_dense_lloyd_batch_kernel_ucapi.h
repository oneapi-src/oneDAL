/* file: kmeans_dense_lloyd_batch_kernel_ucapi.h */
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

#ifndef __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_H__
#define __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_H__

#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "kmeans_types.h"
#include "kernel.h"
#include "numeric_table.h"

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
class KMeansDenseLloydBatchKernelUCAPI : public Kernel
{
public:
    services::Status compute(const NumericTable * const * a, const NumericTable * const * r, const Parameter * par);

private:
    void computeSquares(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::KernelPtr & kernel_compute_squares,
                        const services::Buffer<algorithmFPType> & data, oneapi::internal::UniversalBuffer & dataSq, uint32_t nRows,
                        uint32_t nFeatures, services::Status * st);

    void initDistances(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::KernelPtr & kernel_distances_init,
                       oneapi::internal::UniversalBuffer & centroidsSq, oneapi::internal::UniversalBuffer & distances, uint32_t blockSize,
                       uint32_t nClusters, services::Status * st);

    void computeDistances(oneapi::internal::ExecutionContextIface & context, const services::Buffer<algorithmFPType> & data,
                          const services::Buffer<algorithmFPType> & centroids, oneapi::internal::UniversalBuffer & distances, uint32_t blockSize,
                          uint32_t nClusters, uint32_t nFeatures, services::Status * st);

    void computeAssignments(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::KernelPtr & kernel_compute_assignments,
                            oneapi::internal::UniversalBuffer & distances, const services::Buffer<int> & assignments,
                            oneapi::internal::UniversalBuffer & mindistances, uint32_t blockSize, uint32_t nClusters, services::Status * st);

    void computePartialCandidates(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::KernelPtr & kernel_partial_candidates,
                                  const services::Buffer<int> & assignments, oneapi::internal::UniversalBuffer & mindistances,
                                  oneapi::internal::UniversalBuffer & dataSq, oneapi::internal::UniversalBuffer & candidates,
                                  oneapi::internal::UniversalBuffer & candidateDistances, oneapi::internal::UniversalBuffer & partialCandidates,
                                  oneapi::internal::UniversalBuffer & partialCandidateDistances, uint32_t blockSize, uint32_t nClusters,
                                  uint32_t reset, services::Status * st);

    void mergePartialCandidates(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::KernelPtr & kernel_merge_candidates,
                                oneapi::internal::UniversalBuffer & candidates, oneapi::internal::UniversalBuffer & candidateDistances,
                                oneapi::internal::UniversalBuffer & partialCandidates, oneapi::internal::UniversalBuffer & partialCandidateDistances,
                                uint32_t nClusters, services::Status * st);

    void partialReduceCentroids(oneapi::internal::ExecutionContextIface & context,
                                const oneapi::internal::KernelPtr & kernel_partial_reduce_centroids, const services::Buffer<algorithmFPType> & data,
                                oneapi::internal::UniversalBuffer & distances, const services::Buffer<int> & assignments,
                                oneapi::internal::UniversalBuffer & partialCentroids, oneapi::internal::UniversalBuffer & partialCentroidsCounters,
                                uint32_t blockSize, uint32_t nClusters, uint32_t nFeatures, uint32_t nPartialCentroids, uint32_t doReset,
                                services::Status * st);

    void mergeReduceCentroids(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::KernelPtr & kernel_merge_reduce_centroids,
                              oneapi::internal::UniversalBuffer & partialCentroids, oneapi::internal::UniversalBuffer & partialCentroidsCounters,
                              const services::Buffer<algorithmFPType> & centroids, uint32_t nClusters, uint32_t nFeatures, uint32_t nPartialCentroids,
                              services::Status * st);

    void updateObjectiveFunction(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::KernelPtr & kernel_update_goal_function,
                                 oneapi::internal::UniversalBuffer & dataSq, oneapi::internal::UniversalBuffer & distances,
                                 const services::Buffer<int> & assignments, const services::Buffer<algorithmFPType> & objFunction, uint32_t blockSize,
                                 uint32_t nClusters, uint32_t doReset, services::Status * st);
    uint32_t getCandidatePartNum(uint32_t nClusters);
    uint32_t getWorkgroupsCount(uint32_t rows);
    uint32_t getComputeSquaresWorkgroupsCount(uint32_t nFeatures);
    const char * getComputeSquaresKernelName(uint32_t nFeatures);
    const char * getBuildOptions(uint32_t nClusters);

    const uint32_t _maxWorkItemsPerGroup = 128;   // should be a power of two for interal needs
    const uint32_t _maxLocalBuffer       = 30000; // should be less than a half of local memory (two buffers)
    const uint32_t _preferableSubGroup   = 16;    // preferable maximal sub-group size
};

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif

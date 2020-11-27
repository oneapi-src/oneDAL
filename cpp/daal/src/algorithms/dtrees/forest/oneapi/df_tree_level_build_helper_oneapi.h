/* file: df_tree_level_build_helper_oneapi.h */
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
//  Implementation of a service class that provides
//  common kernels required for building tree levels
//--
*/

#ifndef __DF_TREE_LEVEL_BUILD_HELPER_ONEAPI_H__
#define __DF_TREE_LEVEL_BUILD_HELPER_ONEAPI_H__

#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/dtrees/service_array.h"
#include "src/services/service_arrays.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"

#include "services/internal/sycl/execution_context.h"
#include "services/internal/sycl/types.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// TreeLevelBuildHelperOneAPI - contains common kernels (for classification and regression)
// required for building tree level
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType>
class TreeLevelBuildHelperOneAPI
{
public:
    TreeLevelBuildHelperOneAPI() : _nNodeProps(0) {}
    ~TreeLevelBuildHelperOneAPI() {}

    services::Status init(const char * buildOptions, size_t nNodeProps);

    services::Status initializeTreeOrder(size_t nRows, services::internal::sycl::UniversalBuffer & treeOrder);

    services::Status convertSplitToLeaf(services::internal::sycl::UniversalBuffer & nodeList, size_t nNodes);

    services::Status markPresentRows(const services::internal::sycl::UniversalBuffer & rowsList,
                                     services::internal::sycl::UniversalBuffer & rowsBuffer, size_t nRows, size_t localSize, size_t nSubgroupSums);
    services::Status countAbsentRowsForBlocks(const services::internal::sycl::UniversalBuffer & rowsBuffer, size_t nRows,
                                              services::internal::sycl::UniversalBuffer & partialSums, size_t localSize, size_t nSubgroupSums);
    services::Status countAbsentRowsTotal(const services::internal::sycl::UniversalBuffer & partialSums,
                                          services::internal::sycl::UniversalBuffer & partialPrefixSums,
                                          services::internal::sycl::UniversalBuffer & totalSum, size_t localSize, size_t nSubgroupSums);
    services::Status fillOOBRowsListByBlocks(const services::internal::sycl::UniversalBuffer & rowsBuffer, size_t nRows,
                                             const services::internal::sycl::UniversalBuffer & partialPrefixSums,
                                             services::internal::sycl::UniversalBuffer & oobRowsList, size_t localSize, size_t nSubgroupSums,
                                             size_t nOOBRows);

    services::Status getOOBRows(const services::internal::sycl::UniversalBuffer & rowsList, size_t nRows, size_t & nOOBRows,
                                services::internal::sycl::UniversalBuffer & oobRowsList);

    services::Status getNumOfSplitNodes(const services::internal::sycl::UniversalBuffer & nodeList, size_t nNodes, size_t & nSplitNodes);

    services::Status doNodesSplit(const services::internal::sycl::UniversalBuffer & nodeList, size_t nNodes,
                                  services::internal::sycl::UniversalBuffer & nodeListNew, size_t nNodesNew);

    services::Status splitNodeListOnGroupsBySize(const services::internal::sycl::UniversalBuffer & nodeList, size_t nNodes,
                                                 services::internal::sycl::UniversalBuffer & bigNodesGroups, const size_t nGroups,
                                                 const size_t nGroupProps, services::internal::sycl::UniversalBuffer & nodeIndices);

    services::Status doLevelPartition(const services::internal::sycl::UniversalBuffer & data, services::internal::sycl::UniversalBuffer & nodeList,
                                      size_t nNodes, services::internal::sycl::UniversalBuffer & treeOrder,
                                      services::internal::sycl::UniversalBuffer & treeOrderBuf, size_t nRows, size_t nFeatures);

    services::Status partitionCopy(services::internal::sycl::UniversalBuffer & treeOrderBuf, services::internal::sycl::UniversalBuffer & treeOrder,
                                   size_t iStart, size_t nRows);

    services::Status updateMDIVarImportance(const services::internal::sycl::UniversalBuffer & nodeList,
                                            const services::internal::sycl::UniversalBuffer & nodeImpDecreaseList, size_t nNodes,
                                            services::internal::Buffer<algorithmFPType> & varImp, size_t nFeatures);

private:
    services::Status buildProgram(services::internal::sycl::ClKernelFactoryIface & factory, const char * buildOptions = nullptr);

    services::internal::sycl::KernelPtr kernelInitializeTreeOrder;
    services::internal::sycl::KernelPtr kernelConvertSplitToLeaf;
    services::internal::sycl::KernelPtr kernelGetNumOfSplitNodes;
    services::internal::sycl::KernelPtr kernelDoNodesSplit;
    services::internal::sycl::KernelPtr kernelDoLevelPartitionByGroups;
    services::internal::sycl::KernelPtr kernelSplitNodeListOnGroupsBySize;

    services::internal::sycl::KernelPtr kernelMarkPresentRows;
    services::internal::sycl::KernelPtr kernelCountAbsentRowsForBlocks;
    services::internal::sycl::KernelPtr kernelCountAbsentRowsTotal;
    services::internal::sycl::KernelPtr kernelFillOOBRowsListByBlocks;

    services::internal::sycl::KernelPtr kernelUpdateMDIVarImportance;
    services::internal::sycl::KernelPtr kernelPartitionCopy;

    const size_t _maxLocalSums = 256;
    const size_t _minRowsBlock = 256;

    const size_t _preferableGroupSize          = 256;
    const size_t _preferablePartitionGroupSize = 128; // it showed best perf
    const size_t _preferablePartitionGroupsNum = 8192;
    const size_t _maxWorkItemsPerGroup         = 256; // should be a power of two for interal needs
    const size_t _preferableSubGroup           = 16;  // preferable maximal sub-group size
    const size_t _auxNodeBufferProps           = 2;   // auxilliary buffer for nodes partitioning
    const size_t _partitionMaxBlocksNum        = 256; // max blocks number for one node

    const size_t _int32max = static_cast<size_t>(services::internal::MaxVal<int32_t>::get());
    size_t _nNodeProps;
};

} /* namespace internal */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

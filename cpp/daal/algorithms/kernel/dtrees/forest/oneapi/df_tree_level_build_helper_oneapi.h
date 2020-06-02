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

#include "algorithms/kernel/service_error_handling.h"
#include "algorithms/kernel/dtrees/service_array.h"
#include "service/kernel/service_arrays.h"
#include "externals/service_memory.h"
#include "service/kernel/service_data_utils.h"

#include "oneapi/internal/execution_context.h"
#include "oneapi/internal/types.h"

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
    TreeLevelBuildHelperOneAPI() {}
    ~TreeLevelBuildHelperOneAPI() {}

    services::Status init(const char * buildOptions);

    services::Status initializeTreeOrder(size_t nRows, oneapi::internal::UniversalBuffer & treeOrder);

    services::Status convertSplitToLeaf(oneapi::internal::UniversalBuffer & nodeList, size_t nNodes);

    services::Status markPresentRows(const oneapi::internal::UniversalBuffer & rowsList, oneapi::internal::UniversalBuffer & rowsBuffer, size_t nRows,
                                     size_t localSize, size_t nSubgroupSums);
    services::Status countAbsentRowsForBlocks(const oneapi::internal::UniversalBuffer & rowsBuffer, size_t nRows,
                                              oneapi::internal::UniversalBuffer & partialSums, size_t localSize, size_t nSubgroupSums);
    services::Status countAbsentRowsTotal(const oneapi::internal::UniversalBuffer & partialSums,
                                          oneapi::internal::UniversalBuffer & partialPrefixSums, oneapi::internal::UniversalBuffer & totalSum,
                                          size_t localSize, size_t nSubgroupSums);
    services::Status fillOOBRowsListByBlocks(const oneapi::internal::UniversalBuffer & rowsBuffer, size_t nRows,
                                             const oneapi::internal::UniversalBuffer & partialPrefixSums,
                                             oneapi::internal::UniversalBuffer & oobRowsList, size_t localSize, size_t nSubgroupSums);

    services::Status getOOBRows(const oneapi::internal::UniversalBuffer & rowsList, size_t nRows, size_t & nOOBRows,
                                oneapi::internal::UniversalBuffer & oobRowsList);

    services::Status getNumOfSplitNodes(const oneapi::internal::UniversalBuffer & nodeList, size_t nNodes, size_t & nSplitNodes);

    services::Status doNodesSplit(const oneapi::internal::UniversalBuffer & nodeList, size_t nNodes, oneapi::internal::UniversalBuffer & nodeListNew);

    services::Status splitNodeListOnGroupsBySize(const oneapi::internal::UniversalBuffer & nodeList, size_t nNodes,
                                                 oneapi::internal::UniversalBuffer & bigNodesGroups, oneapi::internal::UniversalBuffer & nodeIndeces);

    services::Status doLevelPartition(const oneapi::internal::UniversalBuffer & data, oneapi::internal::UniversalBuffer & nodeList, size_t nNodes,
                                      oneapi::internal::UniversalBuffer & treeOrder, oneapi::internal::UniversalBuffer & treeOrderBuf, size_t nRows,
                                      size_t nFeatures);

    services::Status partitionCopy(oneapi::internal::UniversalBuffer & treeOrderBuf, oneapi::internal::UniversalBuffer & treeOrder, size_t iStart,
                                   size_t nRows);

    services::Status updateMDIVarImportance(const oneapi::internal::UniversalBuffer & nodeList,
                                            const oneapi::internal::UniversalBuffer & nodeImpDecreaseList, size_t nNodes,
                                            services::Buffer<algorithmFPType> & varImp, size_t nFeatures);

private:
    services::Status buildProgram(oneapi::internal::ClKernelFactoryIface & factory, const char * buildOptions = nullptr);

    oneapi::internal::KernelPtr kernelInitializeTreeOrder;
    oneapi::internal::KernelPtr kernelConvertSplitToLeaf;
    oneapi::internal::KernelPtr kernelGetNumOfSplitNodes;
    oneapi::internal::KernelPtr kernelDoNodesSplit;
    oneapi::internal::KernelPtr kernelDoLevelPartition;
    oneapi::internal::KernelPtr kernelSplitNodeListOnGroupsBySize;

    oneapi::internal::KernelPtr kernelMarkPresentRows;
    oneapi::internal::KernelPtr kernelCountAbsentRowsForBlocks;
    oneapi::internal::KernelPtr kernelCountAbsentRowsTotal;
    oneapi::internal::KernelPtr kernelFillOOBRowsListByBlocks;

    oneapi::internal::KernelPtr kernelUpdateMDIVarImportance;
    oneapi::internal::KernelPtr kernelPartitionCopy;

    const uint32_t _maxLocalSums = 256;
    const uint32_t _minRowsBlock = 256;

    const uint32_t _preferableGroupSize  = 256;
    const uint32_t _maxWorkItemsPerGroup = 256; // should be a power of two for interal needs
    const uint32_t _preferableSubGroup   = 16;  // preferable maximal sub-group size
};

} /* namespace internal */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

/* file: df_regression_predict_dense_oneapi_impl.i */
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
//  Implementation of auxiliary functions for decision forest regression
//  hist method.
//--
*/

#ifndef __DF_REGRESSION_PREDICT_DENSE_ONEAPI_IMPL_I__
#define __DF_REGRESSION_PREDICT_DENSE_ONEAPI_IMPL_I__

#include "src/algorithms/dtrees/forest/regression/oneapi/df_regression_predict_dense_kernel_oneapi.h"
#include "src/algorithms/dtrees/forest/regression/oneapi/cl_kernels/df_batch_predict_regression_kernels.cl"

#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"

#include "src/externals/service_profiler.h"
#include "services/internal/buffer.h"
#include "data_management/data/numeric_table.h"
#include "src/data_management/service_numeric_table.h"
#include "services/env_detect.h"
#include "services/error_indexes.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_algo_utils.h"
#include "src/services/service_arrays.h"
#include "src/services/service_utils.h"
#include "services/internal/sycl/types.h"

using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::internal;
using namespace daal::services::internal::sycl;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace prediction
{
namespace internal
{
template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::buildProgram(ClKernelFactoryIface & factory, const char * programName,
                                                                            const char * programSrc)
{
    services::Status status;

    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
    {
        auto fptype_name   = getKeyFPType<algorithmFPType>();
        auto build_options = fptype_name;
        build_options.add(" -cl-std=CL1.2 ");

        services::String cachekey("__daal_algorithms_df_batch_regression_");
        cachekey.add(build_options);
        cachekey.add(programName);

        factory.build(ExecutionTargetIds::device, cachekey.c_str(), programSrc, build_options.c_str(), status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

///////////////////////////////////////////////////////////////////////////////////////////
/* compute method for PredictKernelOneAPI */
///////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::compute(services::HostAppIface * const pHostApp, const NumericTable * const x,
                                                                       const decision_forest::regression::Model * const m, NumericTable * const res)
{
    services::Status status;

    const size_t nRows = x->getNumberOfRows();
    const size_t nCols = x->getNumberOfColumns();

    const daal::algorithms::decision_forest::regression::internal::ModelImpl * const pModel =
        static_cast<const daal::algorithms::decision_forest::regression::internal::ModelImpl * const>(m);
    const auto nTrees = pModel->size();

    auto & context        = Environment::getInstance().getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    if (nRows > _int32max)
    {
        return services::Status(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
    }
    if (nCols > _int32max)
    {
        return services::Status(services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    }
    if (nTrees > _int32max)
    {
        return services::Status(services::ErrorIncorrectSizeOfModel);
    }

    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory, "predict_reg_kernels", df_batch_predict_regression_kernels));

    kernelPredictByTreesGroup = kernel_factory.getKernel("predictByTreesGroup", status);
    kernelReduceResponse      = kernel_factory.getKernel("reduceResponse", status);
    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> dataBlock;
    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->getBlockOfRows(0, nRows, readOnly, dataBlock));

    BlockDescriptor<algorithmFPType> resBlock;
    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(res)->getBlockOfRows(0, nRows, writeOnly, resBlock));

    auto dataBuffer = dataBlock.getBuffer();
    auto resBuffer  = resBlock.getBuffer();

    DAAL_CHECK_STATUS_VAR(predictByAllTrees(dataBuffer, m, resBuffer, nRows, nCols));

    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->releaseBlockOfRows(dataBlock));
    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(res)->releaseBlockOfRows(resBlock));

    return status;
}

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::predictByAllTrees(const services::internal::Buffer<algorithmFPType> & srcBuffer,
                                                                                 const decision_forest::regression::Model * const m,
                                                                                 services::internal::Buffer<algorithmFPType> & resObsResponse,
                                                                                 size_t nRows, size_t nCols)
{
    services::Status status;
    const daal::algorithms::decision_forest::regression::internal::ModelImpl * const pModel =
        static_cast<const daal::algorithms::decision_forest::regression::internal::ModelImpl * const>(m);

    auto & context = services::internal::getDefaultContext();

    const auto nTrees = pModel->size();

    TArray<const dtrees::internal::DecisionTreeTable *, DAAL_BASE_CPU> _aTree;

    _aTree.reset(nTrees);
    DAAL_CHECK_MALLOC(_aTree.get());

    _nTreeGroups = _nTreeGroupsMin;

    if (nTrees > _nTreesLarge)
    {
        _nTreeGroups = _nTreeGroupsForLarge;
    }
    else if (nTrees > _nTreesMedium)
    {
        _nTreeGroups = _nTreeGroupsForMedium;
    }
    else if (nTrees > _nTreesSmall)
    {
        _nTreeGroups = _nTreeGroupsForSmall;
    }

    size_t maxTreeSize = 0;
    for (size_t i = 0; i < nTrees; ++i)
    {
        _aTree[i]   = pModel->at(i);
        maxTreeSize = maxTreeSize < _aTree[i]->getNumberOfRows() ? _aTree[i]->getNumberOfRows() : maxTreeSize;
    }
    if (maxTreeSize > _int32max)
    {
        return services::Status(services::ErrorIncorrectSizeOfModel);
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, maxTreeSize, nTrees);
    const size_t treeBlockSize = maxTreeSize * nTrees;

    TArray<int32_t, DAAL_BASE_CPU> tFI(treeBlockSize);
    TArray<int32_t, DAAL_BASE_CPU> tLC(treeBlockSize);
    TArray<algorithmFPType, DAAL_BASE_CPU> tFV(treeBlockSize);

    auto ftrIdxArr = context.allocate(TypeIds::id<int>(), treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto leftNodeIdxOrClassIdArr = context.allocate(TypeIds::id<int>(), treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto ftrValueOrResponseArr = context.allocate(TypeIds::id<algorithmFPType>(), treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, _nTreeGroups);
    auto obsResponses = context.allocate(TypeIds::id<algorithmFPType>(), nRows * _nTreeGroups, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.fill(obsResponses, (algorithmFPType)0, status);
    DAAL_CHECK_STATUS_VAR(status);

    for (size_t iTree = 0; iTree < nTrees; iTree++)
    {
        const size_t treeSize                = _aTree[iTree]->getNumberOfRows();
        const DecisionTreeNode * const aNode = (const DecisionTreeNode *)(*_aTree[iTree]).getArray();

        int32_t * const fi         = tFI.get() + iTree * maxTreeSize;
        int32_t * const lc         = tLC.get() + iTree * maxTreeSize;
        algorithmFPType * const fv = tFV.get() + iTree * maxTreeSize;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < treeSize; i++)
        {
            fi[i] = aNode[i].featureIndex;
            lc[i] = aNode[i].leftIndexOrClass;
            fv[i] = (algorithmFPType)aNode[i].featureValueOrResponse;
        }
    }

    algorithmFPType probasScale = (algorithmFPType)1 / nTrees;

    context.copy(ftrIdxArr, 0, (void *)tFI.get(), treeBlockSize, 0, treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.copy(leftNodeIdxOrClassIdArr, 0, (void *)tLC.get(), treeBlockSize, 0, treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.copy(ftrValueOrResponseArr, 0, (void *)tFV.get(), treeBlockSize, 0, treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(
        predictByTreesGroup(srcBuffer, ftrIdxArr, leftNodeIdxOrClassIdArr, ftrValueOrResponseArr, obsResponses, nRows, nCols, nTrees, maxTreeSize));
    DAAL_CHECK_STATUS_VAR(reduceResponse(obsResponses, resObsResponse, nRows, _nTreeGroups, probasScale));

    return status;
}

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::predictByTreesGroup(const services::internal::Buffer<algorithmFPType> & srcBuffer,
                                                                                   const UniversalBuffer & featureIndexList,
                                                                                   const UniversalBuffer & leftOrClassTypeList,
                                                                                   const UniversalBuffer & featureValueList,
                                                                                   UniversalBuffer & obsResponses, size_t nRows, size_t nCols,
                                                                                   size_t nTrees, size_t maxTreeSize)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.predictByTreesGroup);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelPredictByTreesGroup;

    DAAL_CHECK_STATUS_VAR(status);

    size_t localSize   = _maxLocalSize;
    size_t nRowsBlocks = 1;
    if (nRows > _nRowsLarge)
    {
        nRowsBlocks = _nRowsBlocksForLarge;
    }
    else if (nRows > _nRowsMedium)
    {
        nRowsBlocks = _nRowsBlocksForMedium;
    }
    {
        KernelRange local_range(localSize, 1);
        KernelRange global_range(nRowsBlocks * localSize, _nTreeGroups);

        KernelNDRange range(2);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(nRows <= _int32max);
        DAAL_ASSERT(nCols <= _int32max);
        DAAL_ASSERT(nTrees <= _int32max);
        DAAL_ASSERT(maxTreeSize <= _int32max);

        DAAL_ASSERT(srcBuffer.size() == nRows * nCols);

        DAAL_ASSERT_UNIVERSAL_BUFFER(featureIndexList, int32_t, maxTreeSize * nTrees);
        DAAL_ASSERT_UNIVERSAL_BUFFER(leftOrClassTypeList, int32_t, maxTreeSize * nTrees);
        DAAL_ASSERT_UNIVERSAL_BUFFER(featureValueList, algorithmFPType, maxTreeSize * nTrees);
        DAAL_ASSERT_UNIVERSAL_BUFFER(obsResponses, algorithmFPType, nRows * _nTreeGroups);

        for (size_t procTrees = 0; procTrees < nTrees; procTrees += _nTreeGroups)
        {
            KernelArguments args(10, status);
            DAAL_CHECK_STATUS_VAR(status);
            args.set(0, srcBuffer, AccessModeIds::read);
            args.set(1, featureIndexList, AccessModeIds::read);
            args.set(2, leftOrClassTypeList, AccessModeIds::read);
            args.set(3, featureValueList, AccessModeIds::read);
            args.set(4, obsResponses, AccessModeIds::readwrite);
            args.set(5, static_cast<int32_t>(nRows));
            args.set(6, static_cast<int32_t>(nCols));
            args.set(7, static_cast<int32_t>(nTrees));
            args.set(8, static_cast<int32_t>(maxTreeSize));
            args.set(9, static_cast<int32_t>(procTrees));

            context.run(range, kernel, args, status);

            DAAL_CHECK_STATUS_VAR(status);
        }
    }

    return status;
}

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::reduceResponse(const UniversalBuffer & obsResponses,
                                                                              services::internal::Buffer<algorithmFPType> & resObsResponse,
                                                                              size_t nRows, size_t nTrees, algorithmFPType scale)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.reduceResponse);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & kernel  = kernelReduceResponse;

    size_t localSize = _preferableSubGroup;
    size_t nGroups   = _maxGroupsNum;
    {
        DAAL_ASSERT(nRows <= _int32max);
        DAAL_ASSERT(nTrees <= _int32max);
        DAAL_ASSERT(resObsResponse.size() == nRows * 1);
        DAAL_ASSERT_UNIVERSAL_BUFFER(obsResponses, algorithmFPType, nRows * _nTreeGroups);

        KernelArguments args(5, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, obsResponses, AccessModeIds::read);
        args.set(1, resObsResponse, AccessModeIds::readwrite);
        args.set(2, static_cast<int32_t>(nRows));
        args.set(3, static_cast<int32_t>(nTrees));
        args.set(4, scale);

        KernelRange local_range(localSize);
        KernelRange global_range(nGroups * localSize);

        KernelNDRange range(1);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

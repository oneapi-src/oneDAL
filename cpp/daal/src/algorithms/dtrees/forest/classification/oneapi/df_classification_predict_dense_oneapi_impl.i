/* file: df_classification_predict_dense_oneapi_impl.i */
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
//  Implementation of auxiliary functions for decision forest classification
//  hist method.
//--
*/

#ifndef __DF_CLASSIFICATION_PREDICT_DENSE_ONEAPI_IMPL_I__
#define __DF_CLASSIFICATION_PREDICT_DENSE_ONEAPI_IMPL_I__

#include "src/algorithms/dtrees/forest/classification/oneapi/df_classification_predict_dense_kernel_oneapi.h"
#include "src/algorithms/dtrees/forest/classification/oneapi/cl_kernels/df_batch_predict_classification_kernels.cl"

#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"

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
#include "src/services/daal_strings.h"
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
namespace classification
{
namespace prediction
{
namespace internal
{
static services::String getBuildOptions(size_t nClasses)
{
    DAAL_ASSERT(nClasses <= static_cast<size_t>(services::internal::MaxVal<int32_t>::get()));
    char buffer[DAAL_MAX_STRING_SIZE] = { 0 };
    const auto written                = daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, static_cast<int32_t>(nClasses));
    services::String nClassesStr(buffer, written);
    services::String buildOptions = " -D NUM_OF_CLASSES=";
    buildOptions.add(nClassesStr);

    return buildOptions;
}

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::buildProgram(ClKernelFactoryIface & factory, const char * programName,
                                                                            const char * programSrc, const char * buildOptions)
{
    services::Status status;

    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
    {
        auto fptype_name   = getKeyFPType<algorithmFPType>();
        auto build_options = fptype_name;
        build_options.add(" -cl-std=CL1.2 ");

        if (buildOptions)
        {
            build_options.add(buildOptions);
        }

        services::String cachekey("__daal_algorithms_df_batch_classification_");
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
                                                                       const decision_forest::classification::Model * const m,
                                                                       NumericTable * const res, NumericTable * const prob, const size_t nClasses,
                                                                       const VotingMethod votingMethod)
{
    services::Status status;

    _nClasses     = nClasses;
    _votingMethod = votingMethod;

    const size_t nRows = x->getNumberOfRows();
    const size_t nCols = x->getNumberOfColumns();

    const daal::algorithms::decision_forest::classification::internal::ModelImpl * const pModel =
        static_cast<const daal::algorithms::decision_forest::classification::internal::ModelImpl * const>(m);
    const auto nTrees = pModel->size();

    DAAL_CHECK_EX((nClasses <= _int32max), ErrorIncorrectParameter, ParameterName, nClassesStr());

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

    services::String buildOptions = getBuildOptions(_nClasses);

    auto & context        = Environment::getInstance().getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory, "predict_cls_kernels", df_batch_predict_classification_kernels, buildOptions.c_str()));

    kernelPredictByTreesWeighted   = kernel_factory.getKernel("predictByTreesWeighted", status);
    kernelPredictByTreesUnweighted = kernel_factory.getKernel("predictByTreesUnweighted", status);
    kernelReduceClassHist          = kernel_factory.getKernel("reduceClassHist", status);
    kernelDetermineWinners         = kernel_factory.getKernel("determineWinners", status);
    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> dataBlock;
    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->getBlockOfRows(0, nRows, readOnly, dataBlock));

    BlockDescriptor<algorithmFPType> probBlock;

    auto dataBuffer = dataBlock.getBuffer();

    UniversalBuffer classHist;
    if (prob)
    {
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(prob)->getBlockOfRows(0, nRows, readWrite, probBlock));
        classHist = probBlock.getBuffer();
    }
    else
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nClasses, nRows);
        classHist = context.allocate(TypeIds::id<algorithmFPType>(), _nClasses * nRows, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    DAAL_CHECK_STATUS_VAR(predictByAllTrees(dataBuffer, m, classHist, nRows, nCols));

    if (res)
    {
        BlockDescriptor<algorithmFPType> resBlock;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(res)->getBlockOfRows(0, nRows, writeOnly, resBlock));

        auto resBuffer = resBlock.getBuffer();
        DAAL_CHECK_STATUS_VAR(determineWinners(classHist, resBuffer, nRows));
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(res)->releaseBlockOfRows(resBlock));
    }

    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->releaseBlockOfRows(dataBlock));
    if (prob)
    {
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(prob)->releaseBlockOfRows(probBlock));
    }

    return status;
}

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::predictByAllTrees(const services::internal::Buffer<algorithmFPType> & srcBuffer,
                                                                                 const decision_forest::classification::Model * const m,
                                                                                 UniversalBuffer & classHist, size_t nRows, size_t nCols)
{
    services::Status status;
    const daal::algorithms::decision_forest::classification::internal::ModelImpl * const pModel =
        static_cast<const daal::algorithms::decision_forest::classification::internal::ModelImpl * const>(m);

    auto & context = services::internal::getDefaultContext();

    TArray<const dtrees::internal::DecisionTreeTable *, DAAL_BASE_CPU> _aTree;

    const auto nTrees = pModel->size();

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

    bool weighted  = false;
    auto ftrIdxArr = context.allocate(TypeIds::id<int>(), treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto leftNodeIdxOrClassIdArr = context.allocate(TypeIds::id<int>(), treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto ftrValueArr = context.allocate(TypeIds::id<algorithmFPType>(), treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);

    size_t mulClassesTreeGroups = _nClasses * _nTreeGroups;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, mulClassesTreeGroups);
    auto obsClassHist = context.allocate(TypeIds::id<algorithmFPType>(), nRows * mulClassesTreeGroups, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.fill(obsClassHist, (algorithmFPType)0, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.fill(classHist, (algorithmFPType)0, status);
    DAAL_CHECK_STATUS_VAR(status);

    UniversalBuffer probasArr;
    TArray<algorithmFPType, DAAL_BASE_CPU> probasArrHost(treeBlockSize * _nClasses);
    DAAL_CHECK_MALLOC(probasArrHost.get());

    if (_votingMethod == VotingMethod::weighted && pModel->getProbas(0))
    {
        probasArr = context.allocate(TypeIds::id<algorithmFPType>(), treeBlockSize * _nClasses, status);
        DAAL_CHECK_STATUS_VAR(status);
        weighted = true;
    }

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

        if (weighted)
        {
            const double * probas = pModel->getProbas(iTree);

            algorithmFPType * dst_ptr = probasArrHost.get() + iTree * maxTreeSize * _nClasses;

            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < treeSize * _nClasses; i++)
            {
                dst_ptr[i] = static_cast<algorithmFPType>(probas[i]);
            }
        }
    }

    if (weighted)
    {
        context.copy(probasArr, 0, (void *)probasArrHost.get(), treeBlockSize * _nClasses, 0, treeBlockSize * _nClasses, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    algorithmFPType probasScale = (algorithmFPType)1 / nTrees;

    context.copy(ftrIdxArr, 0, (void *)tFI.get(), treeBlockSize, 0, treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.copy(leftNodeIdxOrClassIdArr, 0, (void *)tLC.get(), treeBlockSize, 0, treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.copy(ftrValueArr, 0, (void *)tFV.get(), treeBlockSize, 0, treeBlockSize, status);
    DAAL_CHECK_STATUS_VAR(status);

    if (weighted)
    {
        DAAL_CHECK_STATUS_VAR(predictByTreesWeighted(srcBuffer, ftrIdxArr, leftNodeIdxOrClassIdArr, ftrValueArr, probasArr, obsClassHist, probasScale,
                                                     nRows, nCols, nTrees, maxTreeSize));
    }
    else
    {
        DAAL_CHECK_STATUS_VAR(predictByTreesUnweighted(srcBuffer, ftrIdxArr, leftNodeIdxOrClassIdArr, ftrValueArr, obsClassHist, probasScale, nRows,
                                                       nCols, nTrees, maxTreeSize));
    }
    DAAL_CHECK_STATUS_VAR(reduceClassHist(obsClassHist, classHist, nRows, _nTreeGroups));

    return status;
}

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::predictByTreesWeighted(
    const services::internal::Buffer<algorithmFPType> & srcBuffer, const UniversalBuffer & featureIndexList,
    const UniversalBuffer & leftOrClassTypeList, const UniversalBuffer & featureValueList, const UniversalBuffer & classProba,
    UniversalBuffer & obsClassHist, algorithmFPType scale, size_t nRows, size_t nCols, size_t nTrees, size_t maxTreeSize)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.predictByTreesWeighted);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelPredictByTreesWeighted;

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
        DAAL_ASSERT_UNIVERSAL_BUFFER(classProba, algorithmFPType, maxTreeSize * nTrees * _nClasses);
        DAAL_ASSERT_UNIVERSAL_BUFFER(obsClassHist, algorithmFPType, nRows * _nClasses * _nTreeGroups);

        for (size_t procTrees = 0; procTrees < nTrees; procTrees += _nTreeGroups)
        {
            KernelArguments args(12, status);
            DAAL_CHECK_STATUS_VAR(status);
            args.set(0, srcBuffer, AccessModeIds::read);
            args.set(1, featureIndexList, AccessModeIds::read);
            args.set(2, leftOrClassTypeList, AccessModeIds::read);
            args.set(3, featureValueList, AccessModeIds::read);
            args.set(4, classProba, AccessModeIds::read);
            args.set(5, obsClassHist, AccessModeIds::readwrite);
            args.set(6, scale);
            args.set(7, static_cast<int32_t>(nRows));
            args.set(8, static_cast<int32_t>(nCols));
            args.set(9, static_cast<int32_t>(nTrees));
            args.set(10, static_cast<int32_t>(maxTreeSize));
            args.set(11, static_cast<int32_t>(procTrees));

            context.run(range, kernel, args, status);

            DAAL_CHECK_STATUS_VAR(status);
        }
    }

    return status;
}

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::predictByTreesUnweighted(const services::internal::Buffer<algorithmFPType> & srcBuffer,
                                                                                        const UniversalBuffer & featureIndexList,
                                                                                        const UniversalBuffer & leftOrClassTypeList,
                                                                                        const UniversalBuffer & featureValueList,
                                                                                        UniversalBuffer & obsClassHist, algorithmFPType scale,
                                                                                        size_t nRows, size_t nCols, size_t nTrees, size_t maxTreeSize)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.predictByTreesUnweighted);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelPredictByTreesUnweighted;

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
        DAAL_ASSERT_UNIVERSAL_BUFFER(obsClassHist, algorithmFPType, nRows * _nClasses * _nTreeGroups);

        for (size_t procTrees = 0; procTrees < nTrees; procTrees += _nTreeGroups)
        {
            KernelArguments args(11, status);
            DAAL_CHECK_STATUS_VAR(status);
            args.set(0, srcBuffer, AccessModeIds::read);
            args.set(1, featureIndexList, AccessModeIds::read);
            args.set(2, leftOrClassTypeList, AccessModeIds::read);
            args.set(3, featureValueList, AccessModeIds::read);
            args.set(4, obsClassHist, AccessModeIds::readwrite);
            args.set(5, scale);
            args.set(6, static_cast<int32_t>(nRows));
            args.set(7, static_cast<int32_t>(nCols));
            args.set(8, static_cast<int32_t>(nTrees));
            args.set(9, static_cast<int32_t>(maxTreeSize));
            args.set(10, static_cast<int32_t>(procTrees));

            context.run(range, kernel, args, status);

            DAAL_CHECK_STATUS_VAR(status);
        }
    }

    return status;
}

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::reduceClassHist(const UniversalBuffer & obsClassHist, UniversalBuffer & classHist,
                                                                               size_t nRows, size_t nTrees)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.reduceClassHist);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & kernel  = kernelReduceClassHist;

    size_t localSize = _preferableSubGroup;
    size_t nGroups   = _maxGroupsNum;
    {
        DAAL_ASSERT(nRows <= _int32max);
        DAAL_ASSERT(nTrees <= _int32max);

        DAAL_ASSERT_UNIVERSAL_BUFFER(obsClassHist, algorithmFPType, nRows * _nClasses * _nTreeGroups);
        DAAL_ASSERT_UNIVERSAL_BUFFER(classHist, algorithmFPType, nRows * _nClasses);

        KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, obsClassHist, AccessModeIds::read);
        args.set(1, classHist, AccessModeIds::readwrite);
        args.set(2, static_cast<int32_t>(nRows));
        args.set(3, static_cast<int32_t>(nTrees));

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

template <typename algorithmFPType, prediction::Method method>
services::Status PredictKernelOneAPI<algorithmFPType, method>::determineWinners(const UniversalBuffer & classHist,
                                                                                services::internal::Buffer<algorithmFPType> & resBuffer, size_t nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.determineWinners);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelDetermineWinners;

    size_t localSize = _maxLocalSize;
    size_t nGroups   = _maxGroupsNum;

    {
        DAAL_ASSERT(nRows <= _int32max);

        DAAL_ASSERT(resBuffer.size() == nRows * 1);
        DAAL_ASSERT_UNIVERSAL_BUFFER(classHist, algorithmFPType, nRows * _nClasses);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, classHist, AccessModeIds::read);
        args.set(1, resBuffer, AccessModeIds::write);
        args.set(2, static_cast<int32_t>(nRows));

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
} /* namespace classification */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

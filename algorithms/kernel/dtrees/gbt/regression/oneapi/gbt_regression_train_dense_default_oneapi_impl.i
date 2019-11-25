/* file: gbt_regression_train_dense_default_oneapi_impl.i */
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
//  Implementation of auxiliary functions for gradient boosted trees regression
//  (defaultDense) method.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_DENSE_DEFAULT_ONEAPI_IMPL_I__
#define __GBT_REGRESSION_TRAIN_DENSE_DEFAULT_ONEAPI_IMPL_I__

#include "gbt_regression_train_kernel.h"
#include "gbt_regression_model_impl.h"
#include "gbt_train_dense_default_impl.i"
#include "gbt_train_tree_builder.i"
#include "oneapi/gbt_feature_type_helper_oneapi.i"

#include "services/buffer.h"
#include "numeric_table.h"
#include "env_detect.h"
#include "error_indexes.h"
#include "cl_kernels/gbt_batch_regression_kernels.cl"
#include "service_ittnotify.h"
#include "service_data_utils.h"
#include "service_algo_utils.h"
#include "oneapi/service_defines_oneapi.h"
#include "oneapi/internal/types.h"
#include "../gbt_regression_tree_impl.h"
#include "gbt_model_impl.h"

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::internal;
using namespace daal::algorithms::gbt::training::internal;
using namespace daal::algorithms::gbt::regression::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
namespace internal
{

template <typename algorithmFPType>
static void __buildProgram(ClKernelFactoryIface& factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
    {
        auto fptype_name = getKeyFPType<algorithmFPType>();
        auto build_options = fptype_name;
        build_options.add("-cl-std=CL1.2");

        services::String cachekey("__daal_algorithms_gbt_batch_regression_");
        cachekey.add(fptype_name);
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), gbt_batch_regression_kernels, build_options.c_str());
//        printf ("kernels:\n%s\n", gbt_batch_regression_kernels);
    }
}

/*
//////////////////////////////////////////////////////////////////////////////////////////
// Squared loss function, L(y,f)=1/2([y-f(x)]^2)
//////////////////////////////////////////////////////////////////////////////////////////
class SquaredLoss
{
public:
    virtual void getGradients(size_t n, size_t nRows, const services::Buffer<algorithmFPType>& y, const const services::Buffer<algorithmFPType>& f,
                                                      const services::Buffer<algorithmFPType>& gh) DAAL_C11_OVERRIDE
    {
        auto yHost = y.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        auto fHost = f.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        auto ghHost = gh.template get<algorithmFPType>().toHost(ReadWriteMode::writeOnly);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < n; ++i)
        {
            gh.get()[2 * i] = f.get()[i] - y.get()[i]; //gradient
            gh.get()[2 * i + 1] = 1; //hessian
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchTask for regression
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename BinIndexType, gbt::regression::training::Method method>
class TrainBatchTask : public TrainBatchTaskBaseXBoost<algorithmFPType, BinIndexType, cpu>
{
    typedef TrainBatchTaskBaseXBoost<algorithmFPType, BinIndexType, cpu> super;
public:
    TrainBatchTask(HostAppIface* pHostApp, const NumericTable *x, const NumericTable *y,
        const gbt::training::Parameter& par,
        const dtrees::internal::FeatureTypes& featTypes,
        const dtrees::internal::IndexedFeatures* indexedFeatures,
        engines::internal::BatchBaseImpl& engine, size_t dummy) :
        super(pHostApp, x, y, par, featTypes, indexedFeatures, engine, 1),
        _builder(nullptr)
    {
        _builder = TreeBuilder<algorithmFPType, int, BinIndexType, cpu>::create(*this); // TODO: replace int
    }
    ~TrainBatchTask() { delete _builder; }
    bool done() { return false; }
    virtual services::Status init() DAAL_C11_OVERRIDE
    {
        auto s = super::init();
        if(s)
            s = _builder->init();
        return s;
    }

protected:
    virtual void initLossFunc() DAAL_C11_OVERRIDE
    {
        switch(static_cast<const gbt::regression::training::Parameter&>(this->_par).loss)
        {
        case squared:
            this->_loss = new SquaredLoss<algorithmFPType, cpu>(); break;
        default:
            DAAL_ASSERT(false);
        }
    }

    virtual bool getInitialF(algorithmFPType& val) DAAL_C11_OVERRIDE
    {
        const auto py = this->_dataHelper.y();
        const size_t n = this->_dataHelper.data()->getNumberOfRows();
        const algorithmFPType div = algorithmFPType(1.) / algorithmFPType(n);
        val = algorithmFPType(0);
        for(size_t i = 0; i < n; ++i)
            val += div*py[i];
        return true;
    }

    virtual services::Status buildTrees(gbt::internal::GbtDecisionTree** aTbl,
        HomogenNumericTable<double>** aTblImp, HomogenNumericTable<int>** aTblSmplCnt, GlobalStorages<algorithmFPType, BinIndexType, cpu>& GH_SUMS_BUF) DAAL_C11_OVERRIDE
    {
        this->_nParallelNodes.inc();
        return _builder->run(aTbl[0], aTblImp[0], aTblSmplCnt[0], 0, GH_SUMS_BUF);
    }

protected:
    TreeBuilder<algorithmFPType, int, BinIndexType, cpu>* _builder; // TODO: replace int
};
*/

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::scan(const services::Buffer<algorithmFPType>& values,
                                                                                 UniversalBuffer& partialSums,
                                                                                 int nRows,
                                                                                 int localSize,
                                                                                 int nLocalSums)
{
     // printf ("opencl scan\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.scan);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelScan;

    {
        KernelArguments args(3);
        args.set(0, values, AccessModeIds::read);
        args.set(1, partialSums, AccessModeIds::write);
        args.set(2, nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalSums);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

     // printf ("opencl scan finished\n"); fflush(stdout);

/*    {
        auto col = partialHists.template get<int>().toHost(ReadWriteMode::readOnly);
        for (size_t j = 0; j < nLocalSums; j++)
        {
            for (size_t k = 0; k < (1 << 4); k++)
            {
                printf ("%d ", col.get()[(j << 4) + k]);
            }
            printf ("\n");
        }
        printf ("\n"); fflush (stdout);
    }*/

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::reduce(UniversalBuffer& partialSums,
                                                                                   UniversalBuffer& totalSum,
                                                                                   int localSize,
                                                                                   int nSubgroupSums)
{
     // printf ("opencl reduce\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.reduce);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelReduce;

    {
        KernelArguments args(3);
        args.set(0, partialSums, AccessModeIds::read);
        args.set(1, totalSum, AccessModeIds::write);
        args.set(2, nSubgroupSums);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

     // printf ("opencl reduce finished\n"); fflush(stdout);
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::getInitialResponse(NumericTable& y, algorithmFPType *response)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getInitialResponse);

    services::Status status;

    const size_t nRows = y.getNumberOfRows();

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    const int subSize = _preferableSubGroup;
    const int localSize = _preferableSubGroup;
    const int nLocalSums = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);
    const int nSubgroupSums = nLocalSums * (localSize / subSize);

    auto partialSums = context.allocate(TypeIds::id<algorithmFPType>(), nSubgroupSums, &status);
    auto totalSum = context.allocate(TypeIds::id<algorithmFPType>(), 1, &status);

    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> yBlock;
    y.getBlockOfRows(0, nRows, readOnly, yBlock);
    auto yBuffer = yBlock.getBuffer();

    DAAL_CHECK_STATUS_VAR(scan(yBuffer, partialSums, nRows, localSize, nLocalSums));
    DAAL_CHECK_STATUS_VAR(reduce(partialSums, totalSum, localSize, nLocalSums));

    {
        auto totalSumHost = totalSum.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        *response = totalSumHost.get()[0] / nRows;
    }

    y.releaseBlockOfRows(yBlock);

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeOptCoeffs(NumericTable& y,
                                                                                             UniversalBuffer& response,
                                                                                             UniversalBuffer& optCoeffs)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeOptCoeffs);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelComputeOptCoeffs;

    const size_t nRows = y.getNumberOfRows();

    BlockDescriptor<algorithmFPType> yBlock;
    y.getBlockOfRows(0, nRows, readOnly, yBlock);
    auto yBuffer = yBlock.getBuffer();

    {
        KernelArguments args(3);
        args.set(0, yBuffer, AccessModeIds::read);
        args.set(1, response, AccessModeIds::read);
        args.set(2, optCoeffs, AccessModeIds::write);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    y.releaseBlockOfRows(yBlock);

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::initializeTreeOrder(size_t nRows,
                                                                                                UniversalBuffer& treeOrder)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initializeTreeOrder);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelInitializeTreeOrder;

    {
        KernelArguments args(1);
        args.set(0, treeOrder, AccessModeIds::write);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }
/*
                    {
                        printf ("treeOrder\n");
                        auto col = treeOrder.template get<int>().toHost(ReadWriteMode::readOnly);
                        for (size_t i = 0; i < 100 && i < nRows; i++)
                        {
                            printf ("%d ", col.get()[i]);
                        }
                        printf ("\n"); fflush (stdout);
                    }
*/
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computePartialHistograms(const UniversalBuffer& data,
                                                                                                     UniversalBuffer& treeOrder,
                                                                                                     UniversalBuffer& optCoeffs,
                                                                                                     UniversalBuffer& partialHistograms,
                                                                                                     size_t iStart,
                                                                                                     size_t nRows,
                                                                                                     UniversalBuffer& binOffsets,
                                                                                                     size_t nTotalBins,
                                                                                                     size_t nFeatures,
                                                                                                     size_t localSize,
                                                                                                     size_t nPartialHistograms)
{
    // printf ("opencl computePartialHistograms\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialHistograms);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelComputePartialHistograms;
/*
    {
        auto col = data.template get<int>().toHost(ReadWriteMode::readOnly);
        for (size_t j = 0; j < nRows; j++)
        {
            for (size_t k = 0; k < nFeatures; k++)
            {
                printf ("%d ", col.get()[j * nFeatures + k]);
            }
            printf ("\n");
        }
        printf ("\n"); fflush (stdout);
    }

                    {
                        printf ("partialHistograms\n");
                        auto col = partialHistograms.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
                        for (size_t i = 0; i < nPartialHistograms; i++)
                        {
                            for (size_t j = 0; j < nTotalBins; j++)
                            {
                                for (size_t k = 0; k < 2; k++)
                                {
                                    printf ("%.3f ", col.get()[i * nTotalBins + j * 2 + k]);
                                }
                            }
                            printf ("\n");
                        }
                        printf ("\n"); fflush (stdout);
                    }
*/
/*                    {
                        printf ("binOffsets\n");
                        auto col = binOffsets.template get<int>().toHost(ReadWriteMode::readOnly);
                        for (size_t i = 0; i < nFeatures + 1; i++)
                        {
                            printf ("%d ", col.get()[i]);
                        }
                        printf ("\n"); fflush (stdout);
                    }
*/
    {
        KernelArguments args(9);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, optCoeffs, AccessModeIds::read);
        args.set(3, partialHistograms, AccessModeIds::write);
        args.set(4, (int)iStart);
        args.set(5, (int)nRows);
        args.set(6, binOffsets, AccessModeIds::read);
        args.set(7, (int)nTotalBins);
        args.set(8, (int)nFeatures);

        size_t localSize = nFeatures < _maxLocalSize ? nFeatures : _maxLocalSize;

        KernelRange local_range(1, localSize);
        KernelRange global_range(nPartialHistograms, localSize);

        KernelNDRange range(2);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }
/*
                    {
                        printf ("partialHistograms\n");
                        auto col = partialHistograms.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
                        for (size_t i = 0; i < nPartialHistograms; i++)
                        {
                            for (size_t j = 0; j < nTotalBins; j++)
                            {
                                for (size_t k = 0; k < 2; k++)
                                {
                                    printf ("%.3f ", col.get()[i * nTotalBins + j * 2 + k]);
                                }
                            }
                            printf ("\n");
                        }
                        printf ("\n"); fflush (stdout);
                    }
*/
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::reducePartialHistograms(UniversalBuffer& partialHistograms,
                                                                                                    UniversalBuffer& histograms,
                                                                                                    size_t nTotalBins,
                                                                                                    size_t reduceLocalSize,
                                                                                                    size_t nPartialHistograms)
{
    // printf ("opencl reducePartialHistograms\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.reducePartialHistograms);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelReducePartialHistograms;

    {
        KernelArguments args(4);
        args.set(0, partialHistograms, AccessModeIds::read);
        args.set(1, histograms, AccessModeIds::write);
        args.set(2, (int)nPartialHistograms);
        args.set(3, (int)nTotalBins);

        KernelRange local_range(1, reduceLocalSize);
        KernelRange global_range(nTotalBins, reduceLocalSize);

        KernelNDRange range(2);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }
/*
                    {
                        printf ("histograms\n");
                        auto col = histograms.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
                        for (size_t j = 0; j < nTotalBins; j++)
                        {
                            for (size_t k = 0; k < 2; k++)
                            {
                                printf ("%.3f ", col.get()[j * 2 + k]);
                            }
                        }
                        printf ("\n"); fflush (stdout);
                    }
*/
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeHistogram(const UniversalBuffer& data,
                                                                                             UniversalBuffer& treeOrder,
                                                                                             UniversalBuffer& optCoeffs,
                                                                                             UniversalBuffer& partialHistograms,
                                                                                             UniversalBuffer& histograms,
                                                                                             size_t iStart,
                                                                                             size_t nRows,
                                                                                             UniversalBuffer& binOffsets,
                                                                                             size_t nTotalBins,
                                                                                             size_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeHistogram);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    const int localSize = _preferableSubGroup;
    const int nPartialHistograms = (nRows < _preferableGroupSize * _maxLocalHistograms) ? nRows / _preferableGroupSize + !!(nRows % _preferableGroupSize) : _maxLocalHistograms;

    int reduceLocalSize = 1;
    while (reduceLocalSize * 2 <= nPartialHistograms)
    {
        reduceLocalSize *= 2;
    }

    DAAL_CHECK_STATUS_VAR(computePartialHistograms(data, treeOrder, optCoeffs, partialHistograms, iStart, nRows, binOffsets, nTotalBins, nFeatures, localSize, nPartialHistograms));
    DAAL_CHECK_STATUS_VAR(reducePartialHistograms(partialHistograms, histograms, nTotalBins, reduceLocalSize, nPartialHistograms));

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeHistogramDiff(UniversalBuffer& histogramSrc,
                                                                                                 UniversalBuffer& histogramTotal,
                                                                                                 UniversalBuffer& histogramDst,
                                                                                                 size_t nTotalBins)
{
     // printf ("opencl computeHistogramDiff\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeHistogramDiff);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelComputeHistogramDiff;

    {
        KernelArguments args(3);
        args.set(0, histogramSrc, AccessModeIds::read);
        args.set(1, histogramTotal, AccessModeIds::read);
        args.set(2, histogramDst, AccessModeIds::write);

        KernelRange global_range(nTotalBins);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeTotalOptCoeffs(UniversalBuffer& histogram,
                                                                                                  UniversalBuffer& totalOptCoeffs,
                                                                                                  UniversalBuffer& binOffsets,
                                                                                                  size_t nTotalBins,
                                                                                                  size_t nFeatures,
                                                                                                  size_t localSize)
{
    // printf ("opencl computeTotalOptCoeffs\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeTotalOptCoeffs);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelComputeTotalOptCoeffs;

    {
        KernelArguments args(4);
        args.set(0, histogram, AccessModeIds::read);
        args.set(1, totalOptCoeffs, AccessModeIds::write);
        args.set(2, binOffsets, AccessModeIds::read);
        args.set(3, (int)nTotalBins);

        KernelRange global_range(localSize, nFeatures);

        KernelNDRange range(2);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeBestSplitForFeatures(UniversalBuffer& histogram,
                                                                                                        UniversalBuffer& totalOptCoeffs,
                                                                                                        UniversalBuffer& splitInfo,
                                                                                                        UniversalBuffer& splitValue,
                                                                                                        UniversalBuffer& binOffsets,
                                                                                                        size_t nTotalBins,
                                                                                                        size_t nFeatures,
                                                                                                        algorithmFPType lambda,
                                                                                                        size_t localSize)
{
    // printf ("opencl computeBestSplitForFeatures\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeBestSplitForFeatures);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelComputeBestSplitForFeatures;

    {
        KernelArguments args(7);
        args.set(0, histogram, AccessModeIds::read);
        args.set(1, totalOptCoeffs, AccessModeIds::read);
        args.set(2, splitInfo, AccessModeIds::write);
        args.set(3, splitValue, AccessModeIds::write);
        args.set(4, binOffsets, AccessModeIds::read);
        args.set(5, (int)nTotalBins);
        args.set(6, lambda);

        KernelRange local_range(localSize, 1);
        KernelRange global_range(localSize, nFeatures);

        KernelNDRange range(2);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    // printf ("opencl computeBestSplitForFeatures finished\n"); fflush(stdout);
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeBestSplit(UniversalBuffer& histograms,
                                                                                             UniversalBuffer& binOffsets,
                                                                                             size_t nTotalBins,
                                                                                             size_t nFeatures,
                                                                                             algorithmFPType lambda,
                                                                                             BestSplitOneAPI<algorithmFPType>& bestSplit,
                                                                                             algorithmFPType* gTotal,
                                                                                             algorithmFPType* hTotal)
{
     // printf ("opencl computeBestSplit\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeBestSplit);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto totalOptCoeffs = context.allocate(TypeIds::id<algorithmFPType>(), 2, &status);
    auto splitInfo = context.allocate(TypeIds::id<algorithmFPType>(), nFeatures * 5, &status);
    auto splitValue = context.allocate(TypeIds::id<int>(), nFeatures * 1, &status);

    const int localSize = _preferableSubGroup;

    DAAL_CHECK_STATUS_VAR(computeTotalOptCoeffs(histograms, totalOptCoeffs, binOffsets, nTotalBins, nFeatures, localSize));
    DAAL_CHECK_STATUS_VAR(computeBestSplitForFeatures(histograms, totalOptCoeffs, splitInfo, splitValue, binOffsets, nTotalBins, nFeatures, lambda, localSize));

    if (gTotal && hTotal)
    {
        auto totalOptCoeffsHost = totalOptCoeffs.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        *gTotal = totalOptCoeffsHost.get()[0];
        *hTotal = totalOptCoeffsHost.get()[1];
         // printf ("total opt coeffs = %.10f %.10f\n", totalOptCoeffsHost.get()[0], totalOptCoeffsHost.get()[1]);
    }
    {   
        auto splitInfoHost = splitInfo.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        auto splitValueHost = splitValue.template get<int>().toHost(ReadWriteMode::readOnly);
        for (size_t featId = 0; featId < nFeatures; featId++)
        {
            algorithmFPType impurityDecrease = splitInfoHost.get()[featId * 5 + 0];
            int featureValue = splitValueHost.get()[featId];
//             printf ("found best split for feature = %d, value = %d, impDec = %.5f, %.5f %.5f | %.5f %.5f\n", (int)featId, featureValue, impurityDecrease, splitInfoHost.get()[1], splitInfoHost.get()[2], splitInfoHost.get()[3], splitInfoHost.get()[4]);
            if (featureValue != -1)
            {
                if (impurityDecrease > bestSplit._impurityDecrease || (impurityDecrease == bestSplit._impurityDecrease && featId < bestSplit._featureIndex))
                {
                    bestSplit._impurityDecrease = impurityDecrease;
                    bestSplit._featureIndex = featId;
                    bestSplit._featureValue = featureValue;
                    bestSplit._leftGTotal = splitInfoHost.get()[featId * 5 + 1];
                    bestSplit._leftHTotal = splitInfoHost.get()[featId * 5 + 2];
                    bestSplit._rightGTotal = splitInfoHost.get()[featId * 5 + 3];
                    bestSplit._rightHTotal = splitInfoHost.get()[featId * 5 + 4];
                }
            }
        }
    }

     // printf ("opencl computeBestSplit finished\n"); fflush(stdout);
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::partitionScan(const UniversalBuffer& data,
                                                                                          UniversalBuffer& treeOrder,
                                                                                          UniversalBuffer& partialSums,
                                                                                          int splitValue,
                                                                                          size_t iStart,
                                                                                          size_t nRows,
                                                                                          size_t localSize,
                                                                                          size_t nLocalSums)
{
     // printf ("opencl partitionScan\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionScan);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelPartitionScan;

    {
        KernelArguments args(6);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, partialSums, AccessModeIds::write);
        args.set(3, splitValue);
        args.set(4, (int)iStart);
        args.set(5, (int)nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalSums);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

/*    {
        auto partialSumsHost = partialSums.template get<int>().toHost(ReadWriteMode::readOnly);
        for (int i = 0; i <= nLocalSums; i++)
        {
             printf ("%d ", partialSumsHost.get()[i]);
        }
         // printf ("\n");
    }*/
     // printf ("opencl partitionScan finished\n"); fflush(stdout);
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::partitionSumScan(UniversalBuffer& partialSums,
                                                                                             UniversalBuffer& partialPrefixSums,
                                                                                             UniversalBuffer& totalSum,
                                                                                             size_t localSize,
                                                                                             size_t nSubgroupSums)
{
     // printf ("opencl partitionSumScan\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionSumScan);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelPartitionSumScan;

    {
        KernelArguments args(4);
        args.set(0, partialSums, AccessModeIds::read);
        args.set(1, partialPrefixSums, AccessModeIds::write);
        args.set(2, totalSum, AccessModeIds::write);
        args.set(3, (int)nSubgroupSums);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

/*    {
        auto partialPrefixSumsHost = partialPrefixSums.template get<int>().toHost(ReadWriteMode::readOnly);
        auto totalSumHost = totalSum.template get<int>().toHost(ReadWriteMode::readOnly);
        for (int i = 0; i <= nSubgroupSums; i++)
        {
             printf ("%d ", partialPrefixSumsHost.get()[i]);
        }
        printf ("\n");
        printf ("%d\n", totalSumHost.get()[0]);
    }*/
     // printf ("opencl partitionSumScan finished\n"); fflush(stdout);
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::partitionReorder(const UniversalBuffer& data,
                                                                                             UniversalBuffer& treeOrder,
                                                                                             UniversalBuffer& treeOrderBuf,
                                                                                             UniversalBuffer& partialPrefixSums,
                                                                                             int splitValue,
                                                                                             size_t iStart,
                                                                                             size_t nRows,
                                                                                             size_t localSize,
                                                                                             size_t nLocalSums)
{
     // printf ("opencl partitionReorder\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionReorder);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelPartitionReorder;

    {
        KernelArguments args(7);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, treeOrderBuf, AccessModeIds::write);
        args.set(3, partialPrefixSums, AccessModeIds::read);
        args.set(4, splitValue);
        args.set(5, (int)iStart);
        args.set(6, (int)nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalSums);

        KernelNDRange range(1);
        range.global(global_range, &status); DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status); DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

     // printf ("opencl partitionReorder finished\n"); fflush(stdout);
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::partitionCopy(UniversalBuffer& treeOrderBuf,
                                                                                          UniversalBuffer& treeOrder,
                                                                                          size_t iStart,
                                                                                          size_t nRows)
{
     // printf ("opencl partitionCopy\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionCopy);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelPartitionCopy;

    {
        KernelArguments args(3);
        args.set(0, treeOrderBuf, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::write);
        args.set(2, (int)iStart);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

     // printf ("opencl partitionCopy finished\n"); fflush(stdout);
    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::doPartition(const UniversalBuffer& data,
                                                                                        UniversalBuffer& treeOrder,
                                                                                        UniversalBuffer& treeOrderBuf,
                                                                                        int splitValue,
                                                                                        size_t iStart,
                                                                                        size_t nRows,
                                                                                        size_t& nLeft,
                                                                                        size_t& nRight)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.doPartition);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    const int subSize = _preferableSubGroup;
    const int localSize = _preferableSubGroup;
    const int nLocalSums = _maxLocalSums * localSize < nRows ? _maxLocalSums : (nRows / localSize) + !!(nRows % localSize);
    const int nSubgroupSums = nLocalSums * (localSize / subSize);

    auto partialSums = context.allocate(TypeIds::id<int>(), nSubgroupSums + 1, &status);
    auto partialPrefixSums = context.allocate(TypeIds::id<int>(), nSubgroupSums + 1, &status);
    auto totalSum = context.allocate(TypeIds::id<int>(), 1, &status);

    DAAL_CHECK_STATUS_VAR(status);

    size_t rev = 0;
    
    DAAL_CHECK_STATUS_VAR(partitionScan(data, treeOrder, partialSums, splitValue, iStart, nRows, localSize, nLocalSums));
    DAAL_CHECK_STATUS_VAR(partitionSumScan(partialSums, partialPrefixSums, totalSum, localSize, nSubgroupSums));
    DAAL_CHECK_STATUS_VAR(partitionReorder(data, treeOrder, treeOrderBuf, partialPrefixSums, splitValue, iStart, nRows, localSize, nLocalSums));
    DAAL_CHECK_STATUS_VAR(partitionCopy(treeOrderBuf, treeOrder, iStart, nRows));

    {
        auto totalSumHost = totalSum.template get<int>().toHost(ReadWriteMode::readOnly);
        nRight = totalSumHost.get()[0];
        nLeft = nRows - totalSumHost.get()[0];
        if (nLeft == 0 || nRight == 0)
        {
            return status;
        }
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::updateResponse(UniversalBuffer& treeOrder,
                                                                                           UniversalBuffer& response,
                                                                                           size_t iStart,
                                                                                           size_t nRows,
                                                                                           algorithmFPType inc)
{
     // printf ("opencl update response\n"); fflush(stdout);
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateResponse);

    services::Status status;

    auto& context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto& kernel = kernelUpdateResponse;

    {
        KernelArguments args(5);
        args.set(0, treeOrder, AccessModeIds::read);
        args.set(1, response, AccessModeIds::write);
        args.set(2, (int)iStart);
        args.set(3, (int)nRows);
        args.set(4, inc);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

     // printf ("opencl update response finished\n"); fflush(stdout);

    return status;
}

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::compute(
    HostAppIface* pHostApp, const NumericTable *x, const NumericTable *y, gbt::regression::Model& m, Result& res, const Parameter& par,
    engines::internal::BatchBaseImpl& engine)
{
    typedef TreeTableConnector<algorithmFPType> ConnectorType;

    const size_t nRows = x->getNumberOfRows();
    const size_t nFeatures = x->getNumberOfColumns();
    const size_t nFeaturesPerNode = par.featuresPerNode ? par.featuresPerNode : nFeatures;
    const bool inexactWithHistMethod = !par.memorySavingMode && par.splitMethod == gbt::training::inexact && x->getNumberOfColumns() == nFeaturesPerNode;

    DAAL_ASSERT(inexactWithHistMethod);

    gbt::internal::ModelImpl &modelImpl = *static_cast<daal::algorithms::gbt::regression::internal::ModelImpl*>(&m);
    DAAL_CHECK_MALLOC(modelImpl.reserve(par.maxIterations));

    services::Status status;

    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto& kernel_factory = context.getClKernelFactory();

    __buildProgram<algorithmFPType>(kernel_factory);

    kernelScan = kernel_factory.getKernel("scan");
    kernelReduce = kernel_factory.getKernel("reduce");
    kernelInitializeTreeOrder = kernel_factory.getKernel("initializeTreeOrder");
    kernelComputePartialHistograms = kernel_factory.getKernel("computePartialHistograms");
    kernelReducePartialHistograms = kernel_factory.getKernel("reducePartialHistograms");
    kernelComputeHistogramDiff = kernel_factory.getKernel("computeHistogramDiff");
    kernelComputeOptCoeffs = kernel_factory.getKernel("computeOptCoeffs");
    kernelComputeTotalOptCoeffs = kernel_factory.getKernel("computeTotalOptCoeffs");
    kernelComputeBestSplitForFeatures = kernel_factory.getKernel("computeBestSplitForFeatures");
    kernelPartitionScan = kernel_factory.getKernel("partitionScan");
    kernelPartitionSumScan = kernel_factory.getKernel("partitionSumScan");
    kernelPartitionReorder = kernel_factory.getKernel("partitionReorder");
    kernelPartitionCopy = kernel_factory.getKernel("partitionCopy");
    kernelUpdateResponse = kernel_factory.getKernel("updateResponse");

    gbt::internal::IndexedFeaturesOneAPI<algorithmFPType> indexedFeatures;
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK_MALLOC(featTypes.init(*x));

    BinParams prm(par.maxBins, par.minBinSize);
    DAAL_CHECK_STATUS(status, (indexedFeatures.init(*const_cast<NumericTable*>(x), &featTypes, &prm)));

    auto response = context.allocate(TypeIds::id<algorithmFPType>(), nRows, &status);
    auto optCoeffs = context.allocate(TypeIds::id<algorithmFPType>(), nRows * 2, &status);
    auto treeOrder = context.allocate(TypeIds::id<int>(), nRows, &status);
    auto treeOrderBuf = context.allocate(TypeIds::id<int>(), nRows, &status);
    auto partialHistograms = context.allocate(TypeIds::id<algorithmFPType>(), _maxLocalHistograms * indexedFeatures.totalBins() * 2, &status);

    DAAL_CHECK_STATUS_VAR(status);

    algorithmFPType initResp = 0.0;
    DAAL_CHECK_STATUS(status, getInitialResponse(*const_cast<NumericTable*>(y), &initResp));

     // printf ("initial response = %.10f\n", initResp); fflush(stdout);

    context.fill(response, initResp, &status);
    DAAL_CHECK_STATUS_VAR(status);

    //printf ("filled response\n"); fflush(stdout);
    AOSNumericTablePtr treeStructure = ConnectorType::createGBTree(par.maxTreeDepth, &status);
    DAAL_CHECK_STATUS_VAR(status);
    ConnectorType connector(treeStructure.get());

     // printf ("created tree structure\n"); fflush(stdout);
    const size_t maxNodes = treeStructure->getNumberOfRows();

    Collection<TreeNodeStorage> treeNodeStorages(maxNodes);
    DAAL_CHECK_MALLOC(treeNodeStorages.data());
     // printf ("created tree storages\n"); fflush(stdout);

    for(size_t i = 0; (i < par.maxIterations) && !algorithms::internal::isCancelled(status, pHostApp); ++i)
    {
        printf ("computeOptCoeffs, iter = %d\n", (int)i); fflush(stdout);
        DAAL_CHECK_STATUS_VAR(computeOptCoeffs(*const_cast<NumericTable*>(y), response, optCoeffs));
//        printf ("initialiazeTreeOrder\n"); fflush(stdout);
        DAAL_CHECK_STATUS_VAR(initializeTreeOrder(nRows, treeOrder));
//        printf ("start squanching\n"); fflush(stdout);

        TableRecord<algorithmFPType>  *record = connector.get(0);

        record->level = 0;
        record->nid = 0;
        record->iStart = 0;
        record->n = nRows;
        record->nodeState = ConnectorType::split;
        record->isFinalized = false;

        Collection<SplitRecord<algorithmFPType> > splits;
        Collection<SplitRecord<algorithmFPType> > leafs;

        SplitRecord<algorithmFPType> splitRecord(record);
        splits.push_back(splitRecord);

        for (size_t splitId = 0; splitId < splits.size(); splitId++)
        {
            SplitRecord<algorithmFPType>& split = splits[splitId];
             // printf ("process split %p %p\n", split.first, split.second);
            if (split.first && split.second)
            {
                TableRecord<algorithmFPType>* leftRecord = split.first;
                TableRecord<algorithmFPType>* rightRecord = split.second;
                 // printf ("split %d %d | %d %d | %d %d\n", (int)leftRecord->nid, (int)rightRecord->nid, (int)leftRecord->n, (int)rightRecord->n, (int)leftRecord->iStart, (int)rightRecord->iStart);
                const size_t parentId = (leftRecord->nid - 1) / 2;
                DAAL_CHECK_STATUS_VAR(treeNodeStorages[leftRecord->nid].allocate(indexedFeatures));
                DAAL_CHECK_STATUS_VAR(treeNodeStorages[rightRecord->nid].allocate(indexedFeatures));
                BestSplitOneAPI<algorithmFPType> bestSplitLeft;
                BestSplitOneAPI<algorithmFPType> bestSplitRight;
                if (leftRecord->n < rightRecord->n)
                {
                    DAAL_CHECK_STATUS_VAR(computeHistogram(indexedFeatures.getFullData(), treeOrder, optCoeffs, partialHistograms,
                                                           treeNodeStorages[leftRecord->nid].getHistograms(), leftRecord->iStart,
                                                           leftRecord->n, indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nFeatures));
                    DAAL_CHECK_STATUS_VAR(computeHistogramDiff(treeNodeStorages[leftRecord->nid].getHistograms(),
                                                               treeNodeStorages[parentId].getHistograms(),
                                                               treeNodeStorages[rightRecord->nid].getHistograms(),
                                                               indexedFeatures.totalBins()));
                }
                else
                {
                    DAAL_CHECK_STATUS_VAR(computeHistogram(indexedFeatures.getFullData(), treeOrder, optCoeffs, partialHistograms,
                                                           treeNodeStorages[rightRecord->nid].getHistograms(), rightRecord->iStart,
                                                           rightRecord->n, indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nFeatures));
                    DAAL_CHECK_STATUS_VAR(computeHistogramDiff(treeNodeStorages[rightRecord->nid].getHistograms(),
                                                               treeNodeStorages[parentId].getHistograms(),
                                                               treeNodeStorages[leftRecord->nid].getHistograms(),
                                                               indexedFeatures.totalBins()));
                }
/*                    {
                        printf ("featureId = %d, nBins = %d, histogram:\n", (int)featId, (int)(indexedFeatures.numIndices(featId)));
                        {
                            auto col = treeNodeStorages[leftRecord->nid].getHistogram(featId).template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
                            for (size_t j = 0; j < indexedFeatures.numIndices(featId); j++)
                            {
                                for (size_t k = 0; k < 2; k++)
                                {
                                    printf ("%.3f ", col.get()[j * 2 + k]);
                                }
                            }
                            printf ("\n"); fflush (stdout);
                        }
                        {
                            auto col = treeNodeStorages[rightRecord->nid].getHistogram(featId).template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
                            for (size_t j = 0; j < indexedFeatures.numIndices(featId); j++)
                            {
                                for (size_t k = 0; k < 2; k++)
                                {
                                    printf ("%.3f ", col.get()[j * 2 + k]);
                                }
                            }
                            printf ("\n"); fflush (stdout);
                        }
                    }*/
                DAAL_CHECK_STATUS_VAR(computeBestSplit(treeNodeStorages[leftRecord->nid].getHistograms(), indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nFeatures, par.lambda, bestSplitLeft));
                DAAL_CHECK_STATUS_VAR(computeBestSplit(treeNodeStorages[rightRecord->nid].getHistograms(), indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nFeatures, par.lambda, bestSplitRight));

                bestSplitLeft._impurityDecrease -= (leftRecord->gTotal / (leftRecord->hTotal + par.lambda)) * leftRecord->gTotal;
                // printf ("left node, found overall best split, feature = %d, value = %d, impDec = %.5f, %.5f %.5f | %.5f %.5f\n", bestSplitLeft._featureIndex, bestSplitLeft._featureValue, bestSplitLeft._impurityDecrease, bestSplitLeft._leftGTotal, bestSplitLeft._leftHTotal, bestSplitLeft._rightGTotal, bestSplitLeft._rightHTotal);
                if (bestSplitLeft._impurityDecrease < par.minSplitLoss || bestSplitLeft._featureIndex < 0 || bestSplitLeft._featureValue < 0)
                {
                    leftRecord->isFinalized = true;
                    leftRecord->nodeState = ConnectorType::badSplit;
                }
                else
                {
                    size_t nLeft = 0;
                    size_t nRight = 0;
                    DAAL_CHECK_STATUS_VAR(doPartition(indexedFeatures.getFeature(bestSplitLeft._featureIndex), treeOrder, treeOrderBuf, bestSplitLeft._featureValue, leftRecord->iStart, leftRecord->n, nLeft, nRight));
                    // printf ("after split, left = %d, right = %d\n", (int)nLeft, (int)nRight);
                    if (nLeft == 0 || nRight == 0)
                    {
                        leftRecord->isFinalized = true;
                        leftRecord->nodeState = ConnectorType::badSplit;
                    }
                    else
                    {
                        leftRecord->isFinalized = true;
                        leftRecord->featureValue = bestSplitLeft._featureValue;
                        leftRecord->featureIdx = bestSplitLeft._featureIndex;
                        connector.createNode(leftRecord->level + 1, leftRecord->nid * 2 + 1, nLeft, leftRecord->iStart, bestSplitLeft._leftGTotal, bestSplitLeft._leftHTotal, nLeft, par);
                        connector.createNode(leftRecord->level + 1, leftRecord->nid * 2 + 2, leftRecord->n - nLeft, leftRecord->iStart + nLeft, bestSplitLeft._rightGTotal, bestSplitLeft._rightHTotal, nRight, par);
                        connector.setSplitLevel(leftRecord->level + 1);
                        connector.getSplitNodesMerged(leftRecord->nid, splits, false);
                    }
                }

                bestSplitRight._impurityDecrease -= (rightRecord->gTotal / (rightRecord->hTotal + par.lambda)) * rightRecord->gTotal;
                //printf ("right node, found overall best split, feature = %d, value = %d, impDec = %.5f, %.5f %.5f | %.5f %.5f\n", bestSplitRight._featureIndex, bestSplitRight._featureValue, bestSplitRight._impurityDecrease, bestSplitRight._leftGTotal, bestSplitRight._leftHTotal, bestSplitRight._rightGTotal, bestSplitRight._rightHTotal);
                if (bestSplitRight._impurityDecrease < par.minSplitLoss || bestSplitRight._featureIndex < 0 || bestSplitRight._featureValue < 0)
                {
                    rightRecord->isFinalized = true;
                    rightRecord->nodeState = ConnectorType::badSplit;
                }
                else
                {
                    size_t nLeft = 0;
                    size_t nRight = 0;
                    DAAL_CHECK_STATUS_VAR(doPartition(indexedFeatures.getFeature(bestSplitRight._featureIndex), treeOrder, treeOrderBuf, bestSplitRight._featureValue, rightRecord->iStart, rightRecord->n, nLeft, nRight));
                    // printf ("after split, left = %d, right = %d\n", (int)nLeft, (int)nRight);
                    if (nLeft == 0 || nRight == 0)
                    {
                        rightRecord->isFinalized = true;
                        rightRecord->nodeState = ConnectorType::badSplit;
                    }
                    else
                    {
                        rightRecord->isFinalized = true;
                        rightRecord->featureValue = bestSplitRight._featureValue;
                        rightRecord->featureIdx = bestSplitRight._featureIndex;
                        connector.createNode(rightRecord->level + 1, rightRecord->nid * 2 + 1, nLeft, rightRecord->iStart, bestSplitRight._leftGTotal, bestSplitRight._leftHTotal, nLeft, par);
                        connector.createNode(rightRecord->level + 1, rightRecord->nid * 2 + 2, rightRecord->n - nLeft, rightRecord->iStart + nLeft, bestSplitRight._rightGTotal, bestSplitRight._rightHTotal, nRight, par);
                        connector.setSplitLevel(rightRecord->level + 1);
                        connector.getSplitNodesMerged(rightRecord->nid, splits, false);
                    }
                }

                treeNodeStorages[parentId].clear();
            }
            else
            {
                TableRecord<algorithmFPType>* record = (split.first ? split.first : split.second);
                DAAL_CHECK_STATUS_VAR(treeNodeStorages[record->nid].allocate(indexedFeatures));
                BestSplitOneAPI<algorithmFPType> bestSplit;
                algorithmFPType gTotal = 0.0;
                algorithmFPType hTotal = 0.0;
                DAAL_CHECK_STATUS_VAR(computeHistogram(indexedFeatures.getFullData(), treeOrder, optCoeffs, partialHistograms,
                                 treeNodeStorages[record->nid].getHistograms(), record->iStart, record->n, indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nFeatures));
                DAAL_CHECK_STATUS_VAR(computeBestSplit(treeNodeStorages[record->nid].getHistograms(), indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nFeatures, par.lambda, bestSplit, &gTotal, &hTotal));
/*                    {
                        printf ("nBins = %d\n", (int)(indexedFeatures.totalBins()));
                        auto col = treeNodeStorages[record->nid].getHistograms().template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
                        for (size_t j = 0; j < indexedFeatures.totalBins(); j++)
                        {
                            for (size_t k = 0; k < 2; k++)
                            {
                                printf ("%.3f ", col.get()[j * 2 + k]);
                            }
                        }
                        printf ("\n"); fflush (stdout);
                    }*/
                if (record->nid == 0)
                {
                    record->gTotal = gTotal;
                    record->hTotal = hTotal;
                    record->nTotal = record->n;
                }
                bestSplit._impurityDecrease -= (record->gTotal / (record->hTotal + par.lambda)) * record->gTotal;
                // printf ("found overall best split, feature = %d, value = %d, impDec = %.5f, %.5f %.5f | %.5f %.5f\n", bestSplit._featureIndex, bestSplit._featureValue, bestSplit._impurityDecrease, bestSplit._leftGTotal, bestSplit._leftHTotal, bestSplit._rightGTotal, bestSplit._rightHTotal);
                if (bestSplit._impurityDecrease < par.minSplitLoss || bestSplit._featureIndex < 0 || bestSplit._featureValue < 0)
                {
                    record->isFinalized = true;
                    record->nodeState = ConnectorType::badSplit;
                }
                else
                {
                    size_t nLeft = 0;
                    size_t nRight = 0;
                    DAAL_CHECK_STATUS_VAR(doPartition(indexedFeatures.getFeature(bestSplit._featureIndex), treeOrder, treeOrderBuf, bestSplit._featureValue, record->iStart, record->n, nLeft, nRight));
                    // printf ("after split, left = %d, right = %d\n", (int)nLeft, (int)nRight);
                    if (nLeft == 0 || nRight == 0)
                    {
                        record->isFinalized = true;
                        record->nodeState = ConnectorType::badSplit;
                    }
                    else
                    {
                        record->isFinalized = true;
                        record->featureValue = bestSplit._featureValue;
                        record->featureIdx = bestSplit._featureIndex;
                        connector.createNode(record->level + 1, record->nid * 2 + 1, nLeft, record->iStart, bestSplit._leftGTotal, bestSplit._leftHTotal, nLeft, par);
                        connector.createNode(record->level + 1, record->nid * 2 + 2, record->n - nLeft, record->iStart + nLeft, bestSplit._rightGTotal, bestSplit._rightHTotal, nRight, par);
                        connector.setSplitLevel(record->level + 1);
                        connector.getSplitNodesMerged(record->nid, splits, false);
                    }
                }
                if (record->nid > 0)
                {
                    size_t parentId = (record->nid - 1) / 2;
                    treeNodeStorages[parentId].clear();
                }
            }
        }

        Collection<TableRecord<algorithmFPType> *> leaves;
        connector.getLeafNodes(0, leaves);
        size_t nLeaves = leaves.size();

        for (size_t leafId = 0; leafId < nLeaves; leafId++)
        {
            TableRecord<algorithmFPType> *node = leaves[leafId];

            const size_t curRows = node->n;
            const size_t curOffset = node->iStart;

            algorithmFPType res = 0;

            algorithmFPType val = node->hTotal + par.lambda;
            if(val != 0.0)
            {
                val = -node->gTotal / val;
                const algorithmFPType inc = val * par.shrinkage;

                res = inc;
                
                DAAL_CHECK_STATUS_VAR(updateResponse(treeOrder, response, node->iStart, node->n, inc));
            }

            node->response = res;
            node->isFinalized = 1;
        }

        services::Collection<SharedPtr<algorithmFPType> > binValuesHost(nFeatures);
        DAAL_CHECK_MALLOC(binValuesHost.data());
        services::Collection<algorithmFPType*> binValues(nFeatures);
        DAAL_CHECK_MALLOC(binValues.data());

        for (size_t i = 0; i < nFeatures; i++)
        {
            binValuesHost[i] = indexedFeatures.binBorders(i).template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
            binValues[i] = binValuesHost[i].get();
        }

        size_t maxLevel = 0;
        connector.getMaxLevel(0, maxLevel);
        const size_t nNodes = (1 << (maxLevel + 1)) - 1;
        const size_t nNodesPresent = connector.getNNodes(0);

        gbt::internal::GbtDecisionTree *pTbl = new gbt::internal::GbtDecisionTree(nNodes, maxLevel, nNodesPresent);

        HomogenNumericTable<double> *pTblImp     = new HomogenNumericTable<double>(1, nNodes, NumericTable::doAllocate);
        HomogenNumericTable<int>    *pTblSmplCnt = new HomogenNumericTable<int>(1, nNodes, NumericTable::doAllocate);

        connector.convertToGbtDecisionTree<sse2>(binValues.data(), nNodes, maxLevel, pTbl, pTblImp->getArray(), pTblSmplCnt->getArray(), initResp, par);
        modelImpl.add(pTbl, pTblImp, pTblSmplCnt);

        initResp = 0.0;
    }

    return services::Status();
}

template <typename algorithmFPType, gbt::regression::training::Method method>
uint32_t RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::getWorkgroupsCount(uint32_t rows)
{
    const uint32_t elementsPerGroup = _maxWorkItemsPerGroup;
    uint32_t workgroupsCount = rows / elementsPerGroup;

    if (workgroupsCount*elementsPerGroup < rows)
        workgroupsCount++;

    return workgroupsCount;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
uint32_t RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::getComputeSquaresWorkgroupsCount(uint32_t nFeatures)
{
    size_t workItemsPerGroup = nFeatures < _maxWorkItemsPerGroup ? nFeatures : _maxWorkItemsPerGroup;
    while (workItemsPerGroup & (workItemsPerGroup - 1))
    {
        workItemsPerGroup++;
    }
    if(nFeatures <= 32)
    {
        workItemsPerGroup = nFeatures;
    }
    else if(nFeatures <= 64)
    {
        workItemsPerGroup = nFeatures / 2;
        if(nFeatures % 2 > 0)
            workItemsPerGroup++;
    }
    else if(nFeatures <= 128)
    {
        workItemsPerGroup = nFeatures / 4;
        if(nFeatures % 4 > 0)
            workItemsPerGroup++;
    }
    return workItemsPerGroup;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
const char * RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::getComputeSquaresKernelName(uint32_t nFeatures)
{
    if(nFeatures <= 32) {
        return  "compute_squares_32";
    } else if(nFeatures <= 64) {
        return "compute_squares_64";
    }
    else if(nFeatures <= 128) {
        return "compute_squares_128";
    }
    return "compute_squares";
}

} /* namespace internal */
} /* namespace training */
} /* namespace regression */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

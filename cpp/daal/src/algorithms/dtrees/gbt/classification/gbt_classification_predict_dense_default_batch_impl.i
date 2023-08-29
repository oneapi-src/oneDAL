/* file: gbt_classification_predict_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Common functions for gradient boosted trees classification predictions calculation
//--
*/

#ifndef __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_predict_kernel.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_model_impl.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/dtrees/regression/dtrees_regression_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_predict_dense_default_batch_impl.i"
#include "src/algorithms/dtrees/gbt/gbt_predict_dense_default_impl.i"
#include "src/algorithms/objective_function/cross_entropy_loss/cross_entropy_loss_dense_default_batch_kernel.h"
#include "src/services/service_algo_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// PredictBinaryClassificationTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictBinaryClassificationTask : public gbt::regression::prediction::internal::PredictRegressionTask<algorithmFPType, cpu>
{
public:
    typedef gbt::regression::prediction::internal::PredictRegressionTask<algorithmFPType, cpu> super;
    PredictBinaryClassificationTask(const NumericTable * x, NumericTable * y, NumericTable * prob) : super(x, y), _prob(prob) {}
    services::Status run(const gbt::classification::internal::ModelImpl * m, size_t nIterations, services::HostAppIface * pHostApp)
    {
        DAAL_ASSERT(!nIterations || nIterations <= m->size());
        DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
        const auto nTreesTotal = (nIterations ? nIterations : m->size());
        this->_aTree.reset(nTreesTotal);
        DAAL_CHECK_MALLOC(this->_aTree.get());
        for (size_t i = 0; i < nTreesTotal; ++i) this->_aTree[i] = m->at(i);
        const auto nRows = this->_data->getNumberOfRows();
        services::Status s;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, sizeof(algorithmFPType));
        //compute raw boosted values
        if (this->_res && _prob)
        {
            WriteOnlyRows<algorithmFPType, cpu> resBD(this->_res, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(resBD);
            const algorithmFPType label[2] = { algorithmFPType(1.), algorithmFPType(0.) };
            algorithmFPType * res          = resBD.get();
            WriteOnlyRows<algorithmFPType, cpu> probBD(_prob, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(probBD);
            algorithmFPType * prob_pred = probBD.get();
            TArray<algorithmFPType, cpu> expValPtr(nRows);
            algorithmFPType * expVal = expValPtr.get();
            DAAL_CHECK_MALLOC(expVal);
            s = super::runInternal(pHostApp, this->_res);
            if (!s) return s;

            auto nBlocks           = daal::threader_get_threads_number();
            const size_t blockSize = nRows / nBlocks;
            nBlocks += (nBlocks * blockSize != nRows);

            daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
                const size_t startRow  = iBlock * blockSize;
                const size_t finishRow = (((iBlock + 1) == nBlocks) ? nRows : (iBlock + 1) * blockSize);
                daal::internal::MathInst<algorithmFPType, cpu>::vExp(finishRow - startRow, res + startRow, expVal + startRow);

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t iRow = startRow; iRow < finishRow; ++iRow)
                {
                    res[iRow]               = label[services::internal::SignBit<algorithmFPType, cpu>::get(res[iRow])];
                    prob_pred[2 * iRow + 1] = expVal[iRow] / (algorithmFPType(1.) + expVal[iRow]);
                    prob_pred[2 * iRow]     = algorithmFPType(1.) - prob_pred[2 * iRow + 1];
                }
            });
        }

        else if ((!this->_res) && _prob)
        {
            WriteOnlyRows<algorithmFPType, cpu> probBD(_prob, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(probBD);
            algorithmFPType * prob_pred = probBD.get();
            TArray<algorithmFPType, cpu> expValPtr(nRows);
            algorithmFPType * expVal = expValPtr.get();
            NumericTablePtr expNT    = HomogenNumericTableCPU<algorithmFPType, cpu>::create(expVal, 1, nRows, &s);
            DAAL_CHECK_MALLOC(expVal);
            s = super::runInternal(pHostApp, expNT.get());
            if (!s) return s;

            auto nBlocks           = daal::threader_get_threads_number();
            const size_t blockSize = nRows / nBlocks;
            nBlocks += (nBlocks * blockSize != nRows);
            daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
                const size_t startRow  = iBlock * blockSize;
                const size_t finishRow = (((iBlock + 1) == nBlocks) ? nRows : (iBlock + 1) * blockSize);
                daal::internal::MathInst<algorithmFPType, cpu>::vExp(finishRow - startRow, expVal + startRow, expVal + startRow);
                for (size_t iRow = startRow; iRow < finishRow; ++iRow)
                {
                    prob_pred[2 * iRow + 1] = expVal[iRow] / (algorithmFPType(1.) + expVal[iRow]);
                    prob_pred[2 * iRow]     = algorithmFPType(1.) - prob_pred[2 * iRow + 1];
                }
            });
        }
        else if (this->_res && (!_prob))
        {
            WriteOnlyRows<algorithmFPType, cpu> resBD(this->_res, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(resBD);
            const algorithmFPType label[2] = { algorithmFPType(1.), algorithmFPType(0.) };
            algorithmFPType * res          = resBD.get();
            s                              = super::runInternal(pHostApp, this->_res);
            if (!s) return s;

            for (size_t iRow = 0; iRow < nRows; ++iRow)
            {
                //probablity is a sigmoid(f) hence sign(f) can be checked
                res[iRow] = label[services::internal::SignBit<algorithmFPType, cpu>::get(res[iRow])];
            }
        }
        return s;
    }

protected:
    NumericTable * _prob;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictMulticlassTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictMulticlassTask
{
public:
    typedef gbt::internal::GbtDecisionTree TreeType;
    typedef gbt::prediction::internal::TileDimensions<algorithmFPType> DimType;
    typedef daal::tls<algorithmFPType *> ClassesRawBoostedTlsBase;
    typedef daal::TlsMem<algorithmFPType, cpu> ClassesRawBoostedTls;

    PredictMulticlassTask(const NumericTable * x, NumericTable * y, NumericTable * prob) : _data(x), _res(y), _prob(prob) {}
    services::Status run(const gbt::classification::internal::ModelImpl * m, size_t nClasses, size_t nIterations, services::HostAppIface * pHostApp);

protected:
    services::Status predictByAllTrees(size_t nTreesTotal, size_t nClasses, const DimType & dim);

    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    using dispatcher_t = gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing>;
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    void predictByTrees(algorithmFPType * val, size_t iFirstTree, size_t nTrees, size_t nClasses, const algorithmFPType * x,
                        const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);
    template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
    void predictByTreesVector(algorithmFPType * val, size_t iFirstTree, size_t nTrees, size_t nClasses, const algorithmFPType * x,
                              const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);

    template <bool val>
    struct BooleanConstant
    {
        typedef BooleanConstant<val> type;
    };

    inline void updateResult(algorithmFPType * res, algorithmFPType * val, size_t iRow, size_t i, size_t nClasses, BooleanConstant<true> dispatcher)
    {
        res[iRow + i] = getMaxClass(val + i * nClasses, nClasses);
    }

    inline void updateResult(algorithmFPType * res, algorithmFPType * val, size_t iRow, size_t i, size_t nClasses, BooleanConstant<false> dispatcher)
    {}

    inline algorithmFPType * updateBuffer(algorithmFPType * buff, size_t buf_shift, size_t buf_size, BooleanConstant<true> dispatcher)
    {
        services::internal::service_memset_seq<algorithmFPType, cpu>(buff, algorithmFPType(0), buf_size);
        return buff;
    }

    inline algorithmFPType * updateBuffer(algorithmFPType * buff, size_t buf_shift, size_t buf_size, BooleanConstant<false> dispatcher)
    {
        return buff + buf_shift;
    }

    inline size_t getNumberOfNodes(size_t nTrees)
    {
        size_t nNodesTotal = 0;
        for (size_t iTree = 0; iTree < nTrees; ++iTree)
        {
            nNodesTotal += this->_aTree[iTree]->getNumberOfNodes();
        }
        return nNodesTotal;
    }

    inline bool checkForMissing(const algorithmFPType * x, size_t nTrees, size_t nRows, size_t nColumns)
    {
        size_t nLvlTotal = 0;
        for (size_t iTree = 0; iTree < nTrees; ++iTree)
        {
            nLvlTotal += this->_aTree[iTree]->getMaxLvl();
        }
        if (nLvlTotal <= nColumns)
        {
            // Checking is compicated. Better to do it during inferense.
            return true;
        }
        else
        {
            for (size_t idx = 0; idx < nRows * nColumns; ++idx)
            {
                if (checkFinitenessByComparison(x[idx])) return true;
            }
        }
        return false;
    }

    template <bool hasUnorderedFeatures, bool hasAnyMissing, bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res)
    {
        dispatcher_t<hasUnorderedFeatures, hasAnyMissing> dispatcher;
        size_t iRow = 0;
        for (; iRow + vectorBlockSize <= nRows; iRow += vectorBlockSize)
        {
            algorithmFPType * val = updateBuffer(buff, iRow * nClasses, nClasses * vectorBlockSize, BooleanConstant<reuseBuffer>());
            predictByTreesVector<hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(val, 0, nTrees, nClasses, x + iRow * nColumns, dispatcher);
            for (size_t i = 0; i < vectorBlockSize; ++i)
            {
                updateResult(res, val, iRow, i, nClasses, BooleanConstant<isResValidPtr>());
            }
        }
        for (; iRow < nRows; ++iRow)
        {
            algorithmFPType * val = updateBuffer(buff, iRow * nClasses, nClasses, BooleanConstant<reuseBuffer>());
            predictByTrees(val, 0, nTrees, nClasses, x + iRow * nColumns, dispatcher);
            updateResult(res, val, iRow, 0, nClasses, BooleanConstant<isResValidPtr>());
        }
    }

    template <bool hasAnyMissing, bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res)
    {
        if (this->_featHelper.hasUnorderedFeatures())
        {
            predict<true, hasAnyMissing, isResValidPtr, reuseBuffer, vectorBlockSize>(nTrees, nClasses, nRows, nColumns, x, buff, res);
        }
        else
        {
            predict<false, hasAnyMissing, isResValidPtr, reuseBuffer, vectorBlockSize>(nTrees, nClasses, nRows, nColumns, x, buff, res);
        }
    }

    template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSize>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res)
    {
        const bool hasAnyMissing = checkForMissing(x, nTrees, nRows, nColumns);
        if (hasAnyMissing)
        {
            predict<true, isResValidPtr, reuseBuffer, vectorBlockSize>(nTrees, nClasses, nRows, nColumns, x, buff, res);
        }
        else
        {
            predict<false, isResValidPtr, reuseBuffer, vectorBlockSize>(nTrees, nClasses, nRows, nColumns, x, buff, res);
        }
    }

    // Recursivelly checking template parameter until it becomes equal to dim.vectorBlockSizeFactor or equal to DimType::minVectorBlockSizeFactor.
    template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSizeFactor>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res, const DimType & dim, BooleanConstant<true> keepLooking)
    {
        constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
        if (dim.vectorBlockSizeFactor == vectorBlockSizeFactor)
        {
            predict<isResValidPtr, reuseBuffer, vectorBlockSizeFactor * vectorBlockSizeStep>(nTrees, nClasses, nRows, nColumns, x, buff, res);
        }
        else
        {
            predict<isResValidPtr, reuseBuffer, vectorBlockSizeFactor - 1>(
                nTrees, nClasses, nRows, nColumns, x, buff, res, dim, BooleanConstant<vectorBlockSizeFactor != DimType::minVectorBlockSizeFactor>());
        }
    }

    template <bool isResValidPtr, bool reuseBuffer, size_t vectorBlockSizeFactor>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res, const DimType & dim, BooleanConstant<false> keepLooking)
    {
        constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
        predict<isResValidPtr, reuseBuffer, vectorBlockSizeFactor * vectorBlockSizeStep>(nTrees, nClasses, nRows, nColumns, x, buff, res);
    }

    template <bool isResValidPtr, bool reuseBuffer>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res, const DimType & dim)
    {
        constexpr size_t maxVectorBlockSizeFactor = DimType::maxVectorBlockSizeFactor;
        if (maxVectorBlockSizeFactor > 1)
        {
            predict<isResValidPtr, reuseBuffer, maxVectorBlockSizeFactor>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim,
                                                                          BooleanConstant<true>());
        }
        else
        {
            predict<isResValidPtr, reuseBuffer, maxVectorBlockSizeFactor>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim,
                                                                          BooleanConstant<false>());
        }
    }

    template <bool reuseBuffer>
    inline void predict(size_t nTrees, size_t nClasses, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * buff,
                        algorithmFPType * res, const DimType & dim)
    {
        if (res)
        {
            predict<true, reuseBuffer>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim);
        }
        else
        {
            predict<false, reuseBuffer>(nTrees, nClasses, nRows, nColumns, x, buff, res, dim);
        }
    }

    void softmax(algorithmFPType * Input, algorithmFPType * Output, size_t nRows, size_t nCols);

    size_t getMaxClass(const algorithmFPType * val, size_t nClasses) const
    {
        return services::internal::getMaxElementIndex<algorithmFPType, cpu>(val, nClasses);
    }

protected:
    const NumericTable * _data;
    NumericTable * _res;
    NumericTable * _prob;
    dtrees::internal::FeatureTypes _featHelper;
    TArray<const TreeType *, cpu> _aTree;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface * pHostApp, const NumericTable * x,
                                                                      const classification::Model * m, NumericTable * r, NumericTable * prob,
                                                                      size_t nClasses, size_t nIterations)
{
    const daal::algorithms::gbt::classification::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::gbt::classification::internal::ModelImpl *>(m);
    if (nClasses == 2)
    {
        PredictBinaryClassificationTask<algorithmFPType, cpu> task(x, r, prob);
        return task.run(pModel, nIterations, pHostApp);
    }
    PredictMulticlassTask<algorithmFPType, cpu> task(x, r, prob);
    return task.run(pModel, nClasses, nIterations, pHostApp);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictMulticlassTask<algorithmFPType, cpu>::run(const gbt::classification::internal::ModelImpl * m, size_t nClasses,
                                                                  size_t nIterations, services::HostAppIface * pHostApp)
{
    DAAL_ASSERT(!nIterations || nClasses * nIterations <= m->size());
    const auto nTreesTotal = (nIterations ? nIterations * nClasses : m->size());
    DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
    this->_aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(this->_aTree.get());
    for (size_t i = 0; i < nTreesTotal; ++i) this->_aTree[i] = m->at(i);

    DimType dim(*_data, nTreesTotal, getNumberOfNodes(nTreesTotal));

    return predictByAllTrees(nTreesTotal, nClasses, dim);
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing>
void PredictMulticlassTask<algorithmFPType, cpu>::predictByTrees(algorithmFPType * val, size_t iFirstTree, size_t nTrees, size_t nClasses,
                                                                 const algorithmFPType * x,
                                                                 const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        val[iTree % nClasses] +=
            gbt::prediction::internal::predictForTree<algorithmFPType, TreeType, cpu>(*this->_aTree[iTree], this->_featHelper, x, dispatcher);
    }
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
void PredictMulticlassTask<algorithmFPType, cpu>::predictByTreesVector(algorithmFPType * val, size_t iFirstTree, size_t nTrees, size_t nClasses,
                                                                       const algorithmFPType * x,
                                                                       const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    algorithmFPType v[vectorBlockSize];
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        gbt::prediction::internal::predictForTreeVector<algorithmFPType, TreeType, cpu, hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(
            *this->_aTree[iTree], this->_featHelper, x, v, dispatcher);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < vectorBlockSize; ++j) val[(iTree % nClasses) + j * nClasses] += v[j];
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictMulticlassTask<algorithmFPType, cpu>::predictByAllTrees(size_t nTreesTotal, size_t nClasses, const DimType & dim)
{
    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, dim.nRowsTotal);
    DAAL_CHECK_BLOCK_STATUS(resBD);

    const size_t nCols(_data->getNumberOfColumns());
    const size_t nRows(_data->getNumberOfRows());
    daal::SafeStatus safeStat;
    if (_prob)
    {
        WriteOnlyRows<algorithmFPType, cpu> probBD(_prob, 0, dim.nRowsTotal);
        DAAL_CHECK_BLOCK_STATUS(probBD);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nClasses);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows * nClasses, sizeof(algorithmFPType));
        TArray<algorithmFPType, cpu> valPtr(nRows * nClasses);
        algorithmFPType * valFull = valPtr.get();
        services::internal::service_memset<algorithmFPType, cpu>(valFull, algorithmFPType(0), nRows * nClasses);

        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock) {
            const size_t iStartRow      = iBlock * dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == (dim.nDataBlocks - 1)) ? dim.nRowsTotal - iStartRow : dim.nRowsInBlock;
            algorithmFPType * valL      = valFull + iStartRow * nClasses;
            algorithmFPType * val       = valL;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            algorithmFPType * res = resBD.get() ? resBD.get() + iStartRow : nullptr;

            predict<false>(nTreesTotal, nClasses, nRowsToProcess, nCols, xBD.get(), valL, res, dim);
        });
        algorithmFPType * prob_pred = probBD.get();
        daal::algorithms::optimization_solver::cross_entropy_loss::internal::CrossEntropyLossKernel<
            algorithmFPType, daal::algorithms::optimization_solver::cross_entropy_loss::defaultDense, cpu>::softmaxThreaded(valFull, prob_pred, nRows,
                                                                                                                            nClasses);
    }
    else if (!_prob && this->_res)
    {
        ClassesRawBoostedTls lsData(nClasses * dim.vectorBlockSizeFactor * dim.vectorBlockSizeStep);
        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock) {
            algorithmFPType * const val = lsData.local();
            const size_t iStartRow      = iBlock * dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == (dim.nDataBlocks - 1)) ? dim.nRowsTotal - iStartRow : dim.nRowsInBlock;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            algorithmFPType * res = resBD.get() + iStartRow;

            predict<true, true>(nTreesTotal, nClasses, nRowsToProcess, nCols, xBD.get(), val, res, dim);
        });
    }

    return safeStat.detach();
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace classification */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

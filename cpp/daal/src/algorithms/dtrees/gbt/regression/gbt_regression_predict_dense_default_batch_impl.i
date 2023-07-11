/* file: gbt_regression_predict_dense_default_batch_impl.i */
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
//  Common functions for gradient boosted trees regression predictions calculation
//--
*/

#ifndef __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "src/algorithms/dtrees/gbt/gbt_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_predict_kernel.h"
#include "src/algorithms/dtrees/gbt/treeshap.h"
#include "src/algorithms/dtrees/regression/dtrees_regression_predict_dense_default_impl.i"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/threading/threading.h"

#include <vector> // TODO: remove

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace prediction
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTask
{
public:
    typedef gbt::internal::GbtDecisionTree TreeType;
    typedef gbt::prediction::internal::TileDimensions<algorithmFPType> DimType;
    PredictRegressionTask(const NumericTable * x, NumericTable * y) : _data(x), _res(y) {}

    services::Status run(const gbt::regression::internal::ModelImpl * m, size_t nIterations, services::HostAppIface * pHostApp,
                         bool predShapContributions, bool predShapInteractions);

protected:
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    using dispatcher_t = gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing>;

    services::Status runInternal(services::HostAppIface * pHostApp, NumericTable * result, bool predShapContributions, bool predShapInteractions);
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    algorithmFPType predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                   const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);
    template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
    void predictByTreesVector(size_t iFirstTree, size_t nTrees, const algorithmFPType * x, algorithmFPType * res,
                              const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher, size_t resIncrement);

    inline size_t getNumberOfNodes(size_t nTrees)
    {
        size_t nNodesTotal = 0;
        for (size_t iTree = 0; iTree < nTrees; ++iTree)
        {
            nNodesTotal += _aTree[iTree]->getNumberOfNodes();
        }
        return nNodesTotal;
    }

    inline bool checkForMissing(const algorithmFPType * x, size_t nTrees, size_t nRows, size_t nColumns) const
    {
        size_t nLvlTotal = 0;
        for (size_t iTree = 0; iTree < nTrees; ++iTree)
        {
            nLvlTotal += _aTree[iTree]->getMaxLvl();
        }
        if (nLvlTotal <= nColumns)
        {
            // Checking is complicated. Better to do it during inference
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

    template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res,
                        size_t resIncrement)
    {
        size_t iRow;
        dispatcher_t<hasUnorderedFeatures, hasAnyMissing> dispatcher;
        for (iRow = 0; iRow + vectorBlockSize <= nRows; iRow += vectorBlockSize)
        {
            predictByTreesVector<hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(iTree, nTrees, x + iRow * nColumns, res + iRow, dispatcher,
                                                                                       resIncrement);
        }
        for (; iRow < nRows; ++iRow)
        {
            // result goes into final columns of current row
            const size_t lastColumn = (iRow + 1) * resIncrement - 1;
            res[lastColumn] += predictByTrees(iTree, nTrees, x + iRow * nColumns, dispatcher);
        }
    }

    template <bool hasAnyMissing, size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res,
                        size_t resIncrement)
    {
        if (_featHelper.hasUnorderedFeatures())
        {
            predict<true, hasAnyMissing, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res, resIncrement);
        }
        else
        {
            predict<false, hasAnyMissing, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res, resIncrement);
        }
    }

    template <size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res,
                        size_t resIncrement)
    {
        const bool hasAnyMissing = checkForMissing(x, nTrees, nRows, nColumns);
        if (hasAnyMissing)
        {
            predict<true, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res, resIncrement);
        }
        else
        {
            predict<false, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res, resIncrement);
        }
    }

    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    inline void predictContributions(size_t iTree, size_t nTrees, size_t nRowsData, size_t nColumnsData, const algorithmFPType * x,
                                     algorithmFPType * res, const DimType & dim);

    template <bool hasAnyMissing>
    inline void predictContributions(size_t iTree, size_t nTrees, size_t nRowsData, size_t nColumnsData, const algorithmFPType * x,
                                     algorithmFPType * res, const DimType & dim)
    {
        if (_featHelper.hasUnorderedFeatures())
        {
            predictContributions<true, hasAnyMissing>(iTree, nTrees, nRowsData, nColumnsData, x, res, dim);
        }
        else
        {
            predictContributions<false, hasAnyMissing>(iTree, nTrees, nRowsData, nColumnsData, x, res, dim);
        }
    }

    // TODO: Add vectorBlockSize templates, similar to predict
    // template <size_t vectorBlockSize>
    inline void predictContributions(size_t iTree, size_t nTrees, size_t nRowsData, size_t nColumnsData, const algorithmFPType * x,
                                     algorithmFPType * res, const DimType & dim)
    {
        const bool hasAnyMissing = checkForMissing(x, nTrees, nRowsData, nColumnsData);
        if (hasAnyMissing)
        {
            predictContributions<true>(iTree, nTrees, nRowsData, nColumnsData, x, res, dim);
        }
        else
        {
            predictContributions<false>(iTree, nTrees, nRowsData, nColumnsData, x, res, dim);
        }
    }

    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    inline void predictContributionInteractions(size_t iTree, size_t nTrees, size_t nRowsData, size_t nColumnsData, const algorithmFPType * x,
                                                algorithmFPType * res, const DimType & dim);

    template <bool hasAnyMissing>
    inline void predictContributionInteractions(size_t iTree, size_t nTrees, size_t nRowsData, size_t nColumnsData, const algorithmFPType * x,
                                                algorithmFPType * res, const DimType & dim)
    {
        if (_featHelper.hasUnorderedFeatures())
        {
            predictContributionInteractions<true, hasAnyMissing>(iTree, nTrees, nRowsData, nColumnsData, x, res, dim);
        }
        else
        {
            predictContributionInteractions<false, hasAnyMissing>(iTree, nTrees, nRowsData, nColumnsData, x, res, dim);
        }
    }

    // TODO: Add vectorBlockSize templates, similar to predict
    // template <size_t vectorBlockSize>
    inline void predictContributionInteractions(size_t iTree, size_t nTrees, size_t nRowsData, size_t nColumnsData, const algorithmFPType * x,
                                                algorithmFPType * res, const DimType & dim)
    {
        const bool hasAnyMissing = checkForMissing(x, nTrees, nRowsData, nColumnsData);
        if (hasAnyMissing)
        {
            predictContributionInteractions<true>(iTree, nTrees, nRowsData, nColumnsData, x, res, dim);
        }
        else
        {
            predictContributionInteractions<false>(iTree, nTrees, nRowsData, nColumnsData, x, res, dim);
        }
    }

    template <bool val>
    struct BooleanConstant
    {
        typedef BooleanConstant<val> type;
    };

    // Recursivelly checking template parameter until it becomes equal to dim.vectorBlockSizeFactor or equal to DimType::minVectorBlockSizeFactor.
    template <size_t vectorBlockSizeFactor>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res,
                        const DimType & dim, BooleanConstant<true> keepLooking, size_t resIncrement)
    {
        constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
        if (dim.vectorBlockSizeFactor == vectorBlockSizeFactor)
        {
            predict<vectorBlockSizeFactor * vectorBlockSizeStep>(iTree, nTrees, nRows, nColumns, x, res, resIncrement);
        }
        else
        {
            predict<vectorBlockSizeFactor - 1>(iTree, nTrees, nRows, nColumns, x, res, dim,
                                               BooleanConstant<vectorBlockSizeFactor != DimType::minVectorBlockSizeFactor>(), resIncrement);
        }
    }

    template <size_t vectorBlockSizeFactor>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res,
                        const DimType & dim, BooleanConstant<false> keepLooking, size_t resIncrement)
    {
        constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
        predict<vectorBlockSizeFactor * vectorBlockSizeStep>(iTree, nTrees, nRows, nColumns, x, res, resIncrement);
    }

    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res,
                        const DimType & dim, size_t resIncrement)
    {
        constexpr size_t maxVectorBlockSizeFactor = DimType::maxVectorBlockSizeFactor;
        if (maxVectorBlockSizeFactor > 1)
        {
            predict<maxVectorBlockSizeFactor>(iTree, nTrees, nRows, nColumns, x, res, dim, BooleanConstant<true>(), resIncrement);
        }
        else
        {
            predict<maxVectorBlockSizeFactor>(iTree, nTrees, nRows, nColumns, x, res, dim, BooleanConstant<false>(), resIncrement);
        }
    }

protected:
    dtrees::internal::FeatureTypes _featHelper;
    TArray<const TreeType *, cpu> _aTree;
    const NumericTable * _data;
    NumericTable * _res;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface * pHostApp, const NumericTable * x,
                                                                      const regression::Model * m, NumericTable * r, size_t nIterations,
                                                                      bool predShapContributions, bool predShapInteractions)
{
    const daal::algorithms::gbt::regression::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::gbt::regression::internal::ModelImpl *>(m);
    PredictRegressionTask<algorithmFPType, cpu> task(x, r);
    return task.run(pModel, nIterations, pHostApp, predShapContributions, predShapInteractions);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::run(const gbt::regression::internal::ModelImpl * m, size_t nIterations,
                                                                  services::HostAppIface * pHostApp, bool predShapContributions,
                                                                  bool predShapInteractions)
{
    DAAL_ASSERT(nIterations || nIterations <= m->size());
    DAAL_CHECK_MALLOC(_featHelper.init(*_data));
    const auto nTreesTotal = (nIterations ? nIterations : m->size());
    _aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(_aTree.get());
    for (size_t i = 0; i < nTreesTotal; ++i) _aTree[i] = m->at(i);
    return runInternal(pHostApp, this->_res, predShapContributions, predShapInteractions);
}

/**
 * Helper to predict SHAP contribution values
 * \param[in] iTree index of start tree for the calculation
 * \param[in] nTrees number of trees in block included in calculation
 * \param[in] nRowsData number of rows to process
 * \param[in] nColumnsData number of columns in data, i.e. features
 *                         note: 1 SHAP value per feature + bias term
 * \param[in] x pointer to the start of observation data
 * \param[out] res pointer to the start of memory where results are written to
 * \param[in] dim DimType helper
*/
template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing>
void PredictRegressionTask<algorithmFPType, cpu>::predictContributions(size_t iTree, size_t nTrees, size_t nRowsData, size_t nColumnsData,
                                                                       const algorithmFPType * x, algorithmFPType * res, const DimType & dim)
{
    // TODO: Make use of vectorBlockSize, similar to predictByTreesVector

    const size_t nColumnsPhi   = nColumnsData + 1;
    const size_t biasTermIndex = nColumnsPhi - 1;

    for (size_t iRow = 0; iRow < nRowsData; ++iRow)
    {
        const algorithmFPType * currentX = x + (iRow * nColumnsData);
        algorithmFPType * phi            = res + (iRow * nColumnsPhi);
        for (size_t currentTreeIndex = iTree; currentTreeIndex < iTree + nTrees; ++currentTreeIndex)
        {
            // regression model builder tree 0 contains only the base_score and must be skipped
            if (currentTreeIndex == 0) continue;

            // prepare memory for unique path data
            const int depth = _aTree[currentTreeIndex]->getMaxLvl() + 2;
            std::vector<gbt::internal::PathElement> uniquePathData((depth * (depth + 1)) / 2);

            // TODO: Not using a separate variable (test) for the phi values causes a lot of cache misses and the code
            //       runs 10x slower - why?
            // std::vector<algorithmFPType> test(nColumnsData, 0);

            const gbt::internal::GbtDecisionTree * currentTree = _aTree[currentTreeIndex];
            gbt::internal::treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(currentTree, currentX, phi, nColumnsData, &_featHelper, 1,
                                                                                          0, 0, uniquePathData.data(), 1, 1, -1, 0, 0, 1);

            // PRAGMA_VECTOR_ALWAYS
            // PRAGMA_IVDEP
            // for (int iFeature = 0; iFeature < nColumnsData; ++iFeature)
            // {
            //     phi[iFeature] += test[iFeature];
            // }
        }

        PRAGMA_VECTOR_ALWAYS
        PRAGMA_IVDEP
        for (int iFeature = 0; iFeature < nColumnsData; ++iFeature)
        {
            phi[biasTermIndex] -= phi[iFeature];
        }
    }
}

/**
 * Helper to predict SHAP contribution interactions
 * \param[in] iTree index of start tree for the calculation
 * \param[in] nTrees number of trees in block included in calculation
 * \param[in] nRowsData number of rows to process
 * \param[in] nColumnsData number of columns in data, i.e. features
 *                         note: 1 SHAP value per feature + bias term
 * \param[in] x pointer to the start of observation data
 * \param[out] res pointer to the start of memory where results are written to
 * \param[in] dim DimType helper
*/
template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing>
void PredictRegressionTask<algorithmFPType, cpu>::predictContributionInteractions(size_t iTree, size_t nTrees, size_t nRowsData, size_t nColumnsData,
                                                                                  const algorithmFPType * x, algorithmFPType * res,
                                                                                  const DimType & dim)
{}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::runInternal(services::HostAppIface * pHostApp, NumericTable * result,
                                                                          bool predShapContributions, bool predShapInteractions)
{
    // assert we're not requesting both contributions and interactions
    DAAL_ASSERT(!(predShapContributions && predShapInteractions));

    const size_t nTreesTotal    = _aTree.size();
    const int dataNColumns      = _data->getNumberOfColumns();
    const size_t resultNColumns = result->getNumberOfColumns();
    const size_t resultNRows    = result->getNumberOfRows();

    DimType dim(*_data, nTreesTotal, getNumberOfNodes(nTreesTotal));
    WriteOnlyRows<algorithmFPType, cpu> resMatrix(result, 0, resultNRows); // select all rows for writing
    DAAL_CHECK_BLOCK_STATUS(resMatrix);
    services::internal::service_memset<algorithmFPType, cpu>(resMatrix.get(), 0, resultNRows * resultNColumns); // set nRows * nCols to 0
    SafeStatus safeStat;
    services::Status s;
    HostAppHelper host(pHostApp, 100);

    const size_t predictionIndex = resultNColumns - 1;
    for (size_t iTree = 0; iTree < nTreesTotal; iTree += dim.nTreesInBlock)
    {
        if (!s || host.isCancelled(s, 1)) return s;
        size_t nTreesToUse = ((iTree + dim.nTreesInBlock) < nTreesTotal ? dim.nTreesInBlock : (nTreesTotal - iTree));

        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock) {
            const size_t iStartRow      = iBlock * dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == dim.nDataBlocks - 1) ? dim.nRowsTotal - iBlock * dim.nRowsInBlock : dim.nRowsInBlock;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);

            if (predShapContributions)
            {
                // thread-local write rows into global result buffer
                WriteOnlyRows<algorithmFPType, cpu> resRow(result, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(resRow);

                // bias term: prediction - sum_i phi_i (subtraction in predictContributions)
                predict(iTree, nTreesToUse, nRowsToProcess, dim.nCols, xBD.get(), resRow.get(), dim, resultNColumns);

                // TODO: support tree weights
                predictContributions(iTree, nTreesToUse, nRowsToProcess, dim.nCols, xBD.get(), resRow.get(), dim);
            }
            else if (predShapInteractions)
            {
                // thread-local write rows into global result buffer
                WriteOnlyRows<algorithmFPType, cpu> resRow(result, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(resRow);

                // bias term: prediction - sum_i phi_i (subtraction in predictContributions)
                predict(iTree, nTreesToUse, nRowsToProcess, dim.nCols, xBD.get(), resRow.get(), dim, resultNColumns);

                // TODO: support tree weights
                predictContributionInteractions(iTree, nTreesToUse, nRowsToProcess, dim.nCols, xBD.get(), resRow.get(), dim);
            }
            else
            {
                algorithmFPType * res = resMatrix.get() + iStartRow;
                predict(iTree, nTreesToUse, nRowsToProcess, dim.nCols, xBD.get(), res, dim, 1);
            }
        });

        s = safeStat.detach();
    }

    return s;
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing>
algorithmFPType PredictRegressionTask<algorithmFPType, cpu>::predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                                                            const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    algorithmFPType val = 0;
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
        val += gbt::prediction::internal::predictForTree<algorithmFPType, TreeType, cpu>(*_aTree[iTree], _featHelper, x, dispatcher);
    return val;
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
void PredictRegressionTask<algorithmFPType, cpu>::predictByTreesVector(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                                                       algorithmFPType * res,
                                                                       const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher,
                                                                       size_t resIncrement)
{
    algorithmFPType v[vectorBlockSize];
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        gbt::prediction::internal::predictForTreeVector<algorithmFPType, TreeType, cpu, hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(
            *_aTree[iTree], _featHelper, x, v, dispatcher);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t row = 0; row < vectorBlockSize; ++row)
        {
            const size_t lastColumn = (row + 1) * resIncrement - 1;
            res[lastColumn] += v[row];
        }
    }
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

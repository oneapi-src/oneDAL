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
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_predict_kernel.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/dtrees/regression/dtrees_regression_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/gbt_predict_dense_default_impl.i"

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
    services::Status run(const gbt::regression::internal::ModelImpl * m, size_t nIterations, services::HostAppIface * pHostApp);

protected:
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    using dispatcher_t = gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing>;

    services::Status runInternal(services::HostAppIface * pHostApp, NumericTable * result);
    template <bool hasUnorderedFeatures, bool hasAnyMissing>
    algorithmFPType predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                   const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);
    template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
    void predictByTreesVector(size_t iFirstTree, size_t nTrees, const algorithmFPType * x, algorithmFPType * res,
                              const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher);

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

    template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res)
    {
        size_t iRow;
        dispatcher_t<hasUnorderedFeatures, hasAnyMissing> dispatcher;
        for (iRow = 0; iRow + vectorBlockSize <= nRows; iRow += vectorBlockSize)
        {
            predictByTreesVector<hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(iTree, nTrees, x + iRow * nColumns, res + iRow, dispatcher);
        }
        for (; iRow < nRows; ++iRow)
        {
            res[iRow] += predictByTrees(iTree, nTrees, x + iRow * nColumns, dispatcher);
        }
    }

    template <bool hasAnyMissing, size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res)
    {
        if (this->_featHelper.hasUnorderedFeatures())
        {
            predict<true, hasAnyMissing, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res);
        }
        else
        {
            predict<false, hasAnyMissing, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res);
        }
    }

    template <size_t vectorBlockSize>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res)
    {
        const bool hasAnyMissing = checkForMissing(x, nTrees, nRows, nColumns);
        if (hasAnyMissing)
        {
            predict<true, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res);
        }
        else
        {
            predict<false, vectorBlockSize>(iTree, nTrees, nRows, nColumns, x, res);
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
                        const DimType & dim, BooleanConstant<true> keepLooking)
    {
        constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
        if (dim.vectorBlockSizeFactor == vectorBlockSizeFactor)
        {
            predict<vectorBlockSizeFactor * vectorBlockSizeStep>(iTree, nTrees, nRows, nColumns, x, res);
        }
        else
        {
            predict<vectorBlockSizeFactor - 1>(iTree, nTrees, nRows, nColumns, x, res, dim,
                                               BooleanConstant<vectorBlockSizeFactor != DimType::minVectorBlockSizeFactor>());
        }
    }

    template <size_t vectorBlockSizeFactor>
    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res,
                        const DimType & dim, BooleanConstant<false> keepLooking)
    {
        constexpr size_t vectorBlockSizeStep = DimType::vectorBlockSizeStep;
        predict<vectorBlockSizeFactor * vectorBlockSizeStep>(iTree, nTrees, nRows, nColumns, x, res);
    }

    inline void predict(size_t iTree, size_t nTrees, size_t nRows, size_t nColumns, const algorithmFPType * x, algorithmFPType * res,
                        const DimType & dim)
    {
        constexpr size_t maxVectorBlockSizeFactor = DimType::maxVectorBlockSizeFactor;
        if (maxVectorBlockSizeFactor > 1)
        {
            predict<maxVectorBlockSizeFactor>(iTree, nTrees, nRows, nColumns, x, res, dim, BooleanConstant<true>());
        }
        else
        {
            predict<maxVectorBlockSizeFactor>(iTree, nTrees, nRows, nColumns, x, res, dim, BooleanConstant<false>());
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
                                                                      const regression::Model * m, NumericTable * r, size_t nIterations)
{
    const daal::algorithms::gbt::regression::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::gbt::regression::internal::ModelImpl *>(m);
    PredictRegressionTask<algorithmFPType, cpu> task(x, r);
    return task.run(pModel, nIterations, pHostApp);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::run(const gbt::regression::internal::ModelImpl * m, size_t nIterations,
                                                                  services::HostAppIface * pHostApp)
{
    DAAL_ASSERT(nIterations || nIterations <= m->size());
    DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
    const auto nTreesTotal = (nIterations ? nIterations : m->size());
    this->_aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(this->_aTree.get());
    for (size_t i = 0; i < nTreesTotal; ++i) this->_aTree[i] = m->at(i);
    return runInternal(pHostApp, this->_res);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::runInternal(services::HostAppIface * pHostApp, NumericTable * result)
{
    const auto nTreesTotal = this->_aTree.size();

    DimType dim(*this->_data, nTreesTotal, getNumberOfNodes(nTreesTotal));
    WriteOnlyRows<algorithmFPType, cpu> resBD(result, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    services::internal::service_memset<algorithmFPType, cpu>(resBD.get(), 0, dim.nRowsTotal);
    SafeStatus safeStat;
    services::Status s;
    HostAppHelper host(pHostApp, 100);
    for (size_t iTree = 0; iTree < nTreesTotal; iTree += dim.nTreesInBlock)
    {
        if (!s || host.isCancelled(s, 1)) return s;
        size_t nTreesToUse = ((iTree + dim.nTreesInBlock) < nTreesTotal ? dim.nTreesInBlock : (nTreesTotal - iTree));

        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock) {
            const size_t iStartRow      = iBlock * dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == dim.nDataBlocks - 1) ? dim.nRowsTotal - iBlock * dim.nRowsInBlock : dim.nRowsInBlock;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(this->_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            algorithmFPType * res = resBD.get() + iStartRow;

            predict(iTree, nTreesTotal, nRowsToProcess, dim.nCols, xBD.get(), res, dim);
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
        val += gbt::prediction::internal::predictForTree<algorithmFPType, TreeType, cpu>(*this->_aTree[iTree], this->_featHelper, x, dispatcher);
    return val;
}

template <typename algorithmFPType, CpuType cpu>
template <bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
void PredictRegressionTask<algorithmFPType, cpu>::predictByTreesVector(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                                                       algorithmFPType * res,
                                                                       const dispatcher_t<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    algorithmFPType v[vectorBlockSize];
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        gbt::prediction::internal::predictForTreeVector<algorithmFPType, TreeType, cpu, hasUnorderedFeatures, hasAnyMissing, vectorBlockSize>(
            *this->_aTree[iTree], this->_featHelper, x, v, dispatcher);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < vectorBlockSize; ++j) res[j] += v[j];
    }
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

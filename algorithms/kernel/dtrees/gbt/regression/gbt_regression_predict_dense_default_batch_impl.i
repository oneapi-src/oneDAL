/* file: gbt_regression_predict_dense_default_batch_impl.i */
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
//  Common functions for gradient boosted trees regression predictions calculation
//--
*/

#ifndef __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "gbt_regression_predict_kernel.h"
#include "threading.h"
#include "daal_defines.h"
#include "gbt_regression_model_impl.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "service_memory.h"
#include "dtrees_regression_predict_dense_default_impl.i"
#include "gbt_predict_dense_default_impl.i"

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
using gbt::prediction::internal::VECTOR_BLOCK_SIZE;

//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTask
{
public:
    typedef gbt::internal::GbtDecisionTree TreeType;
    PredictRegressionTask(const NumericTable * x, NumericTable * y) : _data(x), _res(y) {}
    services::Status run(const gbt::regression::internal::ModelImpl * m, size_t nIterations, services::HostAppIface * pHostApp);

protected:
    services::Status runInternal(services::HostAppIface * pHostApp, NumericTable * result);
    algorithmFPType predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType * x);
    void predictByTreesVector(size_t iFirstTree, size_t nTrees, const algorithmFPType * x, algorithmFPType * res);

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

    gbt::prediction::internal::TileDimensions<algorithmFPType> dim(*this->_data, nTreesTotal);
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

            size_t iRow;
            for (iRow = 0; iRow + VECTOR_BLOCK_SIZE <= nRowsToProcess; iRow += VECTOR_BLOCK_SIZE)
            {
                predictByTreesVector(iTree, nTreesToUse, xBD.get() + iRow * dim.nCols, res + iRow);
            }
            for (; iRow < nRowsToProcess; ++iRow)
            {
                res[iRow] += predictByTrees(iTree, nTreesToUse, xBD.get() + iRow * dim.nCols);
            }
        });

        s = safeStat.detach();
    }

    return s;
}

template <typename algorithmFPType, CpuType cpu>
algorithmFPType PredictRegressionTask<algorithmFPType, cpu>::predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType * x)
{
    algorithmFPType val = 0;
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
        val += gbt::prediction::internal::predictForTree<algorithmFPType, TreeType, cpu>(*this->_aTree[iTree], this->_featHelper, x);
    return val;
}

template <typename algorithmFPType, CpuType cpu>
void PredictRegressionTask<algorithmFPType, cpu>::predictByTreesVector(size_t iFirstTree, size_t nTrees, const algorithmFPType * x,
                                                                       algorithmFPType * res)
{
    algorithmFPType v[VECTOR_BLOCK_SIZE];
    for (size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        gbt::prediction::internal::predictForTreeVector<algorithmFPType, TreeType, cpu>(*this->_aTree[iTree], this->_featHelper, x, v);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < VECTOR_BLOCK_SIZE; ++j) res[j] += v[j];
    }
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

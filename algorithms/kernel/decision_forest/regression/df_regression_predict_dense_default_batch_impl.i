/* file: df_regression_predict_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Common functions for decision forest regression predictions calculation
//--
*/

#ifndef __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "df_regression_predict_dense_default_batch.h"
#include "threading.h"
#include "daal_defines.h"
#include "df_regression_model_impl.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "service_memory.h"
#include "df_predict_dense_default_impl.i"

using namespace daal::internal;
using namespace daal::services::internal;

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

static const size_t nRowsInBlock = 500;

//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTask
{
public:
    typedef decision_forest::internal::TreeImpRegression<> TreeType;
    PredictRegressionTask(const NumericTable *x, NumericTable *y, const decision_forest::regression::internal::ModelImpl* m) :
        _data(x), _res(y), _model(m){}

    services::Status run();

protected:
    static algorithmFPType predict(const decision_forest::internal::Tree& t, const algorithmFPType* x)
    {
        const typename TreeType::NodeType::Base* pNode = decision_forest::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x);
        DAAL_ASSERT(pNode);
        return pNode ? TreeType::NodeType::castLeaf(pNode)->response : 0.;
    }

protected:
    const NumericTable* _data;
    NumericTable* _res;
    const decision_forest::internal::ModelImpl* _model;
};

//////////////////////////////////////////////////////////////////////////////////////////
// RandomForestPredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(const NumericTable *x,
    const regression::Model *m, NumericTable *r)
{
    const daal::algorithms::decision_forest::regression::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::decision_forest::regression::internal::ModelImpl*>(m);
    PredictRegressionTask<algorithmFPType, cpu> task(x, r, pModel);
    return task.run();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::run()
{
    const auto nRows = _data->getNumberOfRows();
    const auto nCols = _data->getNumberOfColumns();
    size_t nBlocks = nRows / nRowsInBlock;
    nBlocks += (nBlocks * nRowsInBlock != nRows);

    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    daal::services::internal::service_memset<algorithmFPType, cpu>(resBD.get(), 0, nRows);

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock)
    {
        const size_t iStartRow = iBlock*nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == nBlocks - 1) ? nRows - iBlock * nRowsInBlock : nRowsInBlock;
        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        algorithmFPType* res = resBD.get() + iStartRow;
        daal::threader_for(nRowsToProcess, nRowsToProcess, [&](size_t iRow)
        {
            const auto size = _model->size();
            res[iRow] = predict(*_model->at(0), xBD.get() + iRow*nCols);
            for(size_t iTree = 1; iTree < size; ++iTree)
            {
                const algorithmFPType val = predict(*_model->at(iTree), xBD.get() + iRow*nCols);
                //recalculate response incrementally, as a mean of all trees responses
                algorithmFPType delta = val - res[iRow];
                res[iRow] += delta / algorithmFPType(iTree + 1);
            }
        });
    });
    return safeStat.detach();
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

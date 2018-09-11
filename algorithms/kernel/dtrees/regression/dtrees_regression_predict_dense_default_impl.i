/* file: dtrees_regression_predict_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of auxiliary functions for dtrees regression predict algorithms
//  (defaultDense) method.
//--
*/

#ifndef __DTREES_REGRESSION_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __DTREES_REGRESSION_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "dtrees_model_impl.h"
#include "service_data_utils.h"
#include "dtrees_feature_type_helper.h"
#include "service_environment.h"
#include "dtrees_predict_dense_default_impl.i"
#include "service_algo_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace regression
{
namespace prediction
{
namespace internal
{

using namespace dtrees::internal;
//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTaskBase
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTaskBase
{
public:
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    PredictRegressionTaskBase(const NumericTable *x, NumericTable *y) : _data(x), _res(y){}

protected:
    static algorithmFPType predict(const dtrees::internal::DecisionTreeTable& t,
        const dtrees::internal::FeatureTypes& featTypes, const algorithmFPType* x)
    {
        const typename dtrees::internal::DecisionTreeNode* pNode =
            dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, featTypes, x);
        DAAL_ASSERT(pNode);
        return pNode ? pNode->featureValueOrResponse : 0.;
    }

    algorithmFPType predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType* x)
    {
        algorithmFPType val = 0;
        for(size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
            val += predict(*_aTree[iTree], _featHelper, x);
        return val;
    }
    services::Status run(services::HostAppIface* pHostApp, algorithmFPType factor);

protected:
    dtrees::internal::FeatureTypes _featHelper;
    TArray<const dtrees::internal::DecisionTreeTable*, cpu> _aTree;
    const NumericTable* _data;
    NumericTable* _res;
};

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTaskBase<algorithmFPType, cpu>::run(services::HostAppIface* pHostApp, algorithmFPType factor)
{
    const auto nTreesTotal = _aTree.size();
    const auto treeSize = _aTree[0]->getNumberOfRows()*sizeof(dtrees::internal::DecisionTreeNode);

    dtrees::prediction::internal::TileDimensions<algorithmFPType> dim(*_data, nTreesTotal, treeSize);
    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    services::internal::service_memset<algorithmFPType, cpu>(resBD.get(), 0, dim.nRowsTotal);
    const size_t nThreads = daal::threader_get_threads_number();
    SafeStatus safeStat;
    services::Status s;
    HostAppHelper host(pHostApp, 100);
    for(size_t iTree = 0; iTree < nTreesTotal; iTree += dim.nTreesInBlock)
    {
        if(!s || host.isCancelled(s, 1))
            return s;
        size_t nTreesToUse = ((iTree + dim.nTreesInBlock) < nTreesTotal ? dim.nTreesInBlock : (nTreesTotal - iTree));
        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock)
        {
            const size_t iStartRow = iBlock*dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == dim.nDataBlocks - 1) ? dim.nRowsTotal - iBlock * dim.nRowsInBlock : dim.nRowsInBlock;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            algorithmFPType* res = resBD.get() + iStartRow;
            if(nRowsToProcess < 2 * nThreads)
            {
                for(size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
                    res[iRow] += factor*predictByTrees(iTree, nTreesToUse, xBD.get() + iRow*dim.nCols);
            }
            else
            {
                daal::threader_for(nRowsToProcess, nRowsToProcess, [&](size_t iRow)
                {
                    res[iRow] += factor*predictByTrees(iTree, nTreesToUse, xBD.get() + iRow*dim.nCols);
                });
            }
        });
        s = safeStat.detach();
    }
    return s;
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */

#endif

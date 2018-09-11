/* file: df_classification_predict_dense_default_batch_impl.i */
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
//  Common functions for decision forest classification predictions calculation
//--
*/

#ifndef __DF_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __DF_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "df_classification_predict_dense_default_batch.h"
#include "threading.h"
#include "daal_defines.h"
#include "df_classification_model_impl.h"
#include "service_numeric_table.h"
#include "service_memory.h"
#include "dtrees_predict_dense_default_impl.i"
#include "service_error_handling.h"
#include "service_arrays.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;
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

//////////////////////////////////////////////////////////////////////////////////////////
// PredictClassificationTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictClassificationTask
{
protected:
    typedef dtrees::internal::TreeImpClassification<> TreeType;
    typedef dtrees::prediction::internal::TileDimensions<algorithmFPType> DimType;
    typedef daal::tls<ClassIndexType *> ClassesCounterTlsBase;
    class ClassesCounterTls : public ClassesCounterTlsBase
    {
    public:
        ClassesCounterTls(size_t nClasses) : ClassesCounterTlsBase([=]()-> ClassIndexType*
        {
            return service_scalable_malloc<ClassIndexType, cpu>(nClasses);
        })
        {}
        ~ClassesCounterTls()
        {
            reduce([](ClassIndexType* ptr)-> void
            {
                if(ptr)
                    service_scalable_free<ClassIndexType, cpu>(ptr);
            });
        }
    };

public:
    PredictClassificationTask(const NumericTable *x, NumericTable *y, const dtrees::internal::ModelImpl* m,
        size_t nClasses) : _data(x), _res(y), _model(m), _nClasses(nClasses){}
    Status run(services::HostAppIface* pHostApp);

protected:
    void predictByTrees(ClassIndexType* res, size_t iFirstTree, size_t nTrees, const algorithmFPType* x);
    Status predictByAllTrees(size_t nTreesTotal, const DimType& dim);
    Status predictByBlocksOfTrees(services::HostAppIface* pHostApp,
        size_t nTreesTotal, const DimType& dim, ClassIndexType* aClsCounters);
    size_t getMaxClass(const ClassIndexType* counts) const
    {
        return services::internal::getMaxElementIndex<ClassIndexType, cpu>(counts, _nClasses);
    }

protected:
    dtrees::internal::FeatureTypes _featHelper;
    TArray<const dtrees::internal::DecisionTreeTable*, cpu> _aTree;
    const NumericTable* _data;
    NumericTable* _res;
    const dtrees::internal::ModelImpl* _model;
    size_t _nClasses;
    static const size_t s_cMaxClassesBufSize = 32;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface* pHostApp,
    const NumericTable *x, const decision_forest::classification::Model *m, NumericTable *r, size_t nClasses)
{
    const daal::algorithms::decision_forest::classification::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::decision_forest::classification::internal::ModelImpl*>(m);
    PredictClassificationTask<algorithmFPType, cpu> task(x, r, pModel, nClasses);
    return task.run(pHostApp);
}

template <typename algorithmFPType, CpuType cpu>
void PredictClassificationTask<algorithmFPType, cpu>::predictByTrees(ClassIndexType* res,
    size_t iFirstTree, size_t nTrees, const algorithmFPType* x)
{
    for(size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        const dtrees::internal::DecisionTreeNode* pNode =
            dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(*_aTree[iTree], _featHelper, x);
        DAAL_ASSERT(pNode);
        res[pNode->leftIndexOrClass]++;
    }
}

template <typename algorithmFPType, CpuType cpu>
Status PredictClassificationTask<algorithmFPType, cpu>::predictByAllTrees(size_t nTreesTotal,
    const DimType& dim)
{
    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);

    const bool bUseTLS(_nClasses > s_cMaxClassesBufSize);
    const size_t nCols(_data->getNumberOfColumns());
    ClassesCounterTls lsData(_nClasses);
    daal::SafeStatus safeStat;
    daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock)
    {
        const size_t iStartRow = iBlock*dim.nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == dim.nDataBlocks - 1) ? dim.nRowsTotal - iStartRow : dim.nRowsInBlock;
        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        algorithmFPType* res = resBD.get() + iStartRow;
        daal::threader_for(nRowsToProcess, nRowsToProcess, [&](size_t iRow)
        {
            ClassIndexType buf[s_cMaxClassesBufSize];
            ClassIndexType* val = bUseTLS ? lsData.local() : buf;
            for(size_t i = 0; i < _nClasses; ++i)
                val[i] = 0;
            predictByTrees(val, 0, nTreesTotal, xBD.get() + iRow*nCols);
            res[iRow] = algorithmFPType(getMaxClass(val));
        });
    });
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
Status PredictClassificationTask<algorithmFPType, cpu>::run(services::HostAppIface* pHostApp)
{
    DAAL_CHECK_MALLOC(_featHelper.init(*_data));
    const auto nTreesTotal = _model->size();
    _aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(_aTree.get());
    for(size_t i = 0; i < nTreesTotal; ++i)
        _aTree[i] = _model->at(i);

    const auto treeSize = _aTree[0]->getNumberOfRows()*sizeof(dtrees::internal::DecisionTreeNode);
    DimType dim(*_data, nTreesTotal, treeSize, _nClasses);

    if(dim.nTreeBlocks == 1) //all fit into LL cache
        return predictByAllTrees(nTreesTotal, dim);

    services::internal::TArrayCalloc<ClassIndexType, cpu> aClsCounters(dim.nRowsTotal*_nClasses);
    if(!aClsCounters.get())
        return predictByAllTrees(nTreesTotal, dim);

    return predictByBlocksOfTrees(pHostApp, nTreesTotal, dim, aClsCounters.get());
}

template <typename algorithmFPType, CpuType cpu>
Status PredictClassificationTask<algorithmFPType, cpu>::predictByBlocksOfTrees(
    services::HostAppIface* pHostApp, size_t nTreesTotal,
    const DimType& dim, ClassIndexType* aClsCount)
{
    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);

    const size_t nThreads = daal::threader_get_threads_number();
    daal::SafeStatus safeStat;
    services::Status s;
    HostAppHelper host(pHostApp, 100);
    for(size_t iTree = 0; iTree < nTreesTotal; iTree += dim.nTreesInBlock)
    {
        if(!s || host.isCancelled(s, 1))
            return s;
        const bool bLastGroup(nTreesTotal <= (iTree + dim.nTreesInBlock));
        const size_t nTreesToUse = (bLastGroup ? (nTreesTotal - iTree) : dim.nTreesInBlock);
        daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&, nTreesToUse, bLastGroup](size_t iBlock)
        {
            const size_t iStartRow = iBlock*dim.nRowsInBlock;
            const size_t nRowsToProcess = (iBlock == dim.nDataBlocks - 1) ? dim.nRowsTotal - iStartRow : dim.nRowsInBlock;
            ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            algorithmFPType* res = resBD.get() + iStartRow;
            ClassIndexType* counts = aClsCount + iStartRow*_nClasses;
            if(nRowsToProcess < 2 * nThreads)
            {
                for(size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
                {
                    ClassIndexType* countsForTheRow = counts + iRow*_nClasses;
                    predictByTrees(countsForTheRow, iTree, nTreesToUse, xBD.get() + iRow*dim.nCols);
                    if(bLastGroup)
                        //find winning class now
                        res[iRow] = algorithmFPType(getMaxClass(countsForTheRow));
                }
            }
            else
            {
                daal::threader_for(nRowsToProcess, nRowsToProcess, [&](size_t iRow)
                {
                    ClassIndexType* countsForTheRow = counts + iRow*_nClasses;
                    predictByTrees(countsForTheRow, iTree, nTreesToUse, xBD.get() + iRow*dim.nCols);
                    if(bLastGroup)
                        //find winning class now
                        res[iRow] = algorithmFPType(getMaxClass(countsForTheRow));
                });
            }
        });
        s = safeStat.detach();
    }
    return s;
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace classification */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

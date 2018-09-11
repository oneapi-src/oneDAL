/* file: gbt_classification_predict_dense_default_batch_impl.i */
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
//  Common functions for gradient boosted trees classification predictions calculation
//--
*/

#ifndef __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "gbt_classification_predict_kernel.h"
#include "threading.h"
#include "daal_defines.h"
#include "gbt_classification_model_impl.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "service_memory.h"
#include "dtrees_regression_predict_dense_default_impl.i"
#include "gbt_regression_predict_dense_default_batch_impl.i"
#include "gbt_predict_dense_default_impl.i"

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

using gbt::prediction::internal::VECTOR_BLOCK_SIZE;

//////////////////////////////////////////////////////////////////////////////////////////
// PredictBinaryClassificationTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictBinaryClassificationTask : public gbt::regression::prediction::internal::PredictRegressionTask<algorithmFPType, cpu>
{
public:
    typedef gbt::regression::prediction::internal::PredictRegressionTask<algorithmFPType, cpu> super;
    PredictBinaryClassificationTask(const NumericTable *x, NumericTable *y) : super(x, y){}
    services::Status run(const gbt::classification::internal::ModelImpl* m, size_t nIterations, services::HostAppIface* pHostApp)
    {
        DAAL_ASSERT(!nIterations || nIterations <= m->size());
        DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
        const auto nTreesTotal = (nIterations ? nIterations : m->size());
        this->_aTree.reset(nTreesTotal);
        DAAL_CHECK_MALLOC(this->_aTree.get());
        for(size_t i = 0; i < nTreesTotal; ++i)
            this->_aTree[i] = m->at(i);
        //compute raw boosted values
        auto s = super::runInternal(pHostApp);
        if(!s)
            return s;
        WriteOnlyRows<algorithmFPType, cpu> resBD(this->_res, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(resBD);
        const algorithmFPType label[2] = { algorithmFPType(1.), algorithmFPType(0.) };
        const auto nRows = this->_data->getNumberOfRows();
        algorithmFPType* res = resBD.get();
        //res contains raw boosted values
        for(size_t iRow = 0; iRow < nRows; ++iRow)
        {
            //probablity is a sigmoid(f) hence sign(f) can be checked
            res[iRow] = label[services::internal::SignBit<algorithmFPType, cpu>::get(res[iRow])];
        }
        return s;
    }
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

    PredictMulticlassTask(const NumericTable *x, NumericTable *y) : _data(x), _res(y){}
    services::Status run(const gbt::classification::internal::ModelImpl* m, size_t nClasses, size_t nIterations,
        services::HostAppIface* pHostApp);

protected:
    services::Status predictByAllTrees(size_t nTreesTotal, size_t nClasses, const DimType& dim);

    void predictByTrees(algorithmFPType* res, size_t iFirstTree, size_t nTrees, size_t nClasses, const algorithmFPType* x);
    void predictByTreesVector(algorithmFPType* val, size_t iFirstTree, size_t nTrees, size_t nClasses, const algorithmFPType* x);

    size_t getMaxClass(const algorithmFPType* val, size_t nClasses) const
    {
        return services::internal::getMaxElementIndex<algorithmFPType, cpu>(val, nClasses);
    }

protected:
    const NumericTable* _data;
    NumericTable* _res;
    dtrees::internal::FeatureTypes _featHelper;
    TArray<const TreeType*, cpu> _aTree;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface* pHostApp,
    const NumericTable *x, const classification::Model *m, NumericTable *r, size_t nClasses, size_t nIterations)
{
    const daal::algorithms::gbt::classification::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::gbt::classification::internal::ModelImpl*>(m);
    if(nClasses == 2)
    {
        PredictBinaryClassificationTask<algorithmFPType, cpu> task(x, r);
        return task.run(pModel, nIterations, pHostApp);
    }
    PredictMulticlassTask<algorithmFPType, cpu> task(x, r);
    return task.run(pModel, nClasses, nIterations, pHostApp);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictMulticlassTask<algorithmFPType, cpu>::run(const gbt::classification::internal::ModelImpl* m,
    size_t nClasses, size_t nIterations, services::HostAppIface* pHostApp)
{
    DAAL_ASSERT(!nIterations || nClasses*nIterations <= m->size());
    const auto nTreesTotal = (nIterations ? nIterations*nClasses : m->size());
    DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
    this->_aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(this->_aTree.get());
    for(size_t i = 0; i < nTreesTotal; ++i)
        this->_aTree[i] = m->at(i);

    DimType dim(*_data, nTreesTotal);

    return predictByAllTrees(nTreesTotal, nClasses, dim);
}

template <typename algorithmFPType, CpuType cpu>
void PredictMulticlassTask<algorithmFPType, cpu>::predictByTrees(algorithmFPType* val,
    size_t iFirstTree, size_t nTrees, size_t nClasses, const algorithmFPType* x)
{
    for(size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        val[iTree%nClasses] += gbt::prediction::internal::predictForTree<algorithmFPType, TreeType, cpu>(*this->_aTree[iTree], this->_featHelper, x);
    }
}

template <typename algorithmFPType, CpuType cpu>
void PredictMulticlassTask<algorithmFPType, cpu>::predictByTreesVector(algorithmFPType* val, size_t iFirstTree, size_t nTrees, size_t nClasses, const algorithmFPType* x)
{
    algorithmFPType v[VECTOR_BLOCK_SIZE];
    for(size_t iTree = iFirstTree, iLastTree = iFirstTree + nTrees; iTree < iLastTree; ++iTree)
    {
        gbt::prediction::internal::predictForTreeVector<algorithmFPType, TreeType, cpu>(*this->_aTree[iTree], this->_featHelper, x, v);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t j = 0; j < VECTOR_BLOCK_SIZE; ++j)
            val[(iTree%nClasses) + j*nClasses] += v[j];
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictMulticlassTask<algorithmFPType, cpu>::predictByAllTrees(size_t nTreesTotal, size_t nClasses,
    const DimType& dim)
{
    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    const size_t nCols(_data->getNumberOfColumns());
    ClassesRawBoostedTls lsData(nClasses*VECTOR_BLOCK_SIZE);

    daal::SafeStatus safeStat;
    daal::threader_for(dim.nDataBlocks, dim.nDataBlocks, [&](size_t iBlock)
    {
        algorithmFPType* const val = lsData.local();
        const size_t iStartRow = iBlock*dim.nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == (dim.nDataBlocks - 1)) ? dim.nRowsTotal - iStartRow : dim.nRowsInBlock;
        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        algorithmFPType* res = resBD.get() + iStartRow;

        size_t iRow = 0;
        for(; iRow + VECTOR_BLOCK_SIZE <= nRowsToProcess; iRow += VECTOR_BLOCK_SIZE)
        {
            services::internal::service_memset_seq<algorithmFPType, cpu>(val, algorithmFPType(0), nClasses*VECTOR_BLOCK_SIZE);
            predictByTreesVector(val, 0, nTreesTotal, nClasses, xBD.get() + iRow*nCols);

            for(size_t i = 0; i < gbt::prediction::internal::VECTOR_BLOCK_SIZE; ++i)
            {
                res[iRow + i] = getMaxClass(val + i*nClasses, nClasses);
            }

        }
        for(; iRow < nRowsToProcess; ++iRow)
        {
            services::internal::service_memset_seq<algorithmFPType, cpu>(val, algorithmFPType(0), nClasses);
            predictByTrees(val, 0, nTreesTotal, nClasses, xBD.get() + iRow*nCols);
            res[iRow] = algorithmFPType(getMaxClass(val, nClasses));
        }
    });
    return safeStat.detach();
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace classification */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

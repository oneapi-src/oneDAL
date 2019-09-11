/* file: df_classification_predict_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

typedef int32_t leftOrClassType; /* tree size and number of classes are fit in to 2^31 */
typedef int32_t featureIndexType;
#define _DEFAULT_BLOCK_SIZE 32
#define _DEFAULT_BLOCK_SIZE_COMMON 22
#define _MIN_TREES_FOR_THREADING 100
#define _SCALE_FACTOR_FOR_VECT_PARALLEL_COMPUTE 0.3  /* scale tree size to chose whethever vectorized or not compute path in parallel mode */
#define _MIN_NUMBER_OF_ROWS_FOR_VECT_SEQ_COMPUTE 1024 /* min number of rows to be predicted by vectorized compute path in sequential mode */

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
    void predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType* x, ClassIndexType* res);
    void predictByTree(const algorithmFPType* x, size_t sizeOfBlock, size_t nCols, const featureIndexType* tFI, const leftOrClassType* tLC, const algorithmFPType* tFV, ClassIndexType* res);
    void predictByTreeCommon(const algorithmFPType* x, size_t sizeOfBlock, size_t nCols, const featureIndexType* tFI, const leftOrClassType* tLC, const algorithmFPType* tFV, ClassIndexType* res);

    void parallelPredict(const algorithmFPType* aX, const DecisionTreeNode* aNode, size_t treeSize, size_t nBlocks, size_t nCols, size_t blockSize, size_t residualSize, ClassIndexType* bufVal);
    Status predictByAllTrees(size_t nTreesTotal, const DimType& dim);
    Status predictAllPointsByAllTrees(size_t nTreesTotal);
    Status predictByBlocksOfTrees(services::HostAppIface* pHostApp,
        size_t nTreesTotal, const DimType& dim, ClassIndexType* aClsCounters);
    size_t getMaxClass(const ClassIndexType* counts) const
    {
        return services::internal::getMaxElementIndex<ClassIndexType, cpu>(counts, _nClasses);
    }

    DAAL_FORCEINLINE void predictByTreeInternal(size_t check, size_t blockSize, size_t nCols, uint32_t* currentNodes, bool* isSplits, const algorithmFPType* x, const featureIndexType* fi, const leftOrClassType* lc, const algorithmFPType* fv, ClassIndexType* res)
    {
            for(;check > 0;)
            {
                check = 0;
                for(size_t i = 0; i < blockSize; i++)
                {
                    const algorithmFPType* currentSample = x + i * nCols;
                    const uint32_t cnIdx = currentNodes[i];
                    size_t idx = isSplits[i] * fi[cnIdx];
                    bool sn = currentSample[idx] > fv[cnIdx];
                    currentNodes[i] -= isSplits[i] * (cnIdx - lc[cnIdx] - sn);
                    isSplits[i] = (fi[currentNodes[i]] != -1);
                    check += isSplits[i];
                }
            }
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < blockSize; i++)
            {
                const size_t cl = lc[currentNodes[i]];
                res[i*_nClasses + cl]++;
            }
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
void PredictClassificationTask<algorithmFPType, cpu>::predictByTrees(size_t iFirstTree, size_t nTrees, const algorithmFPType* x, ClassIndexType* res)
{
    const size_t iLastTree = iFirstTree + nTrees;
    for(size_t iTree = iFirstTree; iTree < iLastTree; ++iTree)
    {
        const dtrees::internal::DecisionTreeNode* pNode =
            dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(*_aTree[iTree], _featHelper, x);
        DAAL_ASSERT(pNode);
        res[pNode->leftIndexOrClass]++;
    }
}

template <typename algorithmFPType, CpuType cpu>
void PredictClassificationTask<algorithmFPType, cpu>::parallelPredict(const algorithmFPType* aX, const DecisionTreeNode* aNode, size_t treeSize, size_t nBlocks,
                     size_t nCols, size_t blockSize, size_t residualSize, ClassIndexType* bufVal)
{
    services::internal::TArray<featureIndexType, cpu> tFI(treeSize);
    services::internal::TArray<leftOrClassType, cpu>  tLC(treeSize);
    services::internal::TArray<algorithmFPType, cpu>  tFV(treeSize);

    featureIndexType* fi =  tFI.get();
    leftOrClassType* lc  =  tLC.get();
    algorithmFPType* fv  =  tFV.get();

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(size_t i = 0; i < treeSize; i++)
    {
        fi[i] = aNode[i].featureIndex;
        lc[i] = aNode[i].leftIndexOrClass;
        fv[i] = (algorithmFPType)aNode[i].featureValueOrResponse;
    }

    daal::threader_for(nBlocks,nBlocks,[&,nCols](size_t iBlock){
       predictByTree(aX + iBlock*blockSize*nCols, blockSize, nCols, fi, lc, fv, bufVal + iBlock*blockSize*_nClasses);
    });

    if(residualSize != 0)
    {
        predictByTree(aX + nBlocks*blockSize*nCols, residualSize, nCols, fi, lc, fv, bufVal + nBlocks*blockSize*_nClasses);
    }

}

template <typename algorithmFPType, CpuType cpu>
void PredictClassificationTask<algorithmFPType, cpu>::predictByTreeCommon(const algorithmFPType* x, const size_t sizeOfBlock, const size_t nCols,
                                                      const featureIndexType* fi, const leftOrClassType* lc, const algorithmFPType* fv, ClassIndexType* res)
{
    size_t check = 0;
    check = fi[0] != -1;

    /* done for unrollig */
    if(sizeOfBlock == _DEFAULT_BLOCK_SIZE_COMMON)
    {
        uint32_t currentNodes[_DEFAULT_BLOCK_SIZE_COMMON];
        bool isSplits[_DEFAULT_BLOCK_SIZE_COMMON];
        services::internal::service_memset_seq<uint32_t, avx512>(currentNodes, uint32_t(0), _DEFAULT_BLOCK_SIZE_COMMON);
        services::internal::service_memset_seq<bool, avx512>(isSplits, bool(1), _DEFAULT_BLOCK_SIZE_COMMON);
        predictByTreeInternal(check, _DEFAULT_BLOCK_SIZE_COMMON, nCols, currentNodes, isSplits, x, fi, lc, fv, res);
    }
    else
    {
        services::internal::TArray<uint32_t, cpu> currentNodesT(sizeOfBlock);
        services::internal::TArray<bool, cpu> isSplitsT(sizeOfBlock);
        uint32_t* const currentNodes = currentNodesT.get();
        bool* isSplits = isSplitsT.get();
        if(isSplits && currentNodes)
        {
            services::internal::service_memset_seq<uint32_t, avx512>(currentNodes, uint32_t(0), sizeOfBlock);
            services::internal::service_memset_seq<bool, avx512>(isSplits, bool(1), sizeOfBlock);
            predictByTreeInternal(check, sizeOfBlock, nCols, currentNodes, isSplits, x, fi, lc, fv, res);
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
void PredictClassificationTask<algorithmFPType, cpu>::predictByTree(const algorithmFPType* x, const size_t sizeOfBlock, const size_t nCols,
    const featureIndexType* tFI, const leftOrClassType* tLC, const algorithmFPType* tFV, ClassIndexType* res)
{
    predictByTreeCommon(x, sizeOfBlock, nCols, tFI, tLC, tFV, res);
}

#if defined (__INTEL_COMPILER)
template <>
void PredictClassificationTask<float, avx512>::predictByTree(const float* x, const size_t sizeOfBlock, const size_t nCols, const featureIndexType* feat_idx,
    const leftOrClassType* left_son, const float* split_point, ClassIndexType* res)
{
    if(sizeOfBlock == _DEFAULT_BLOCK_SIZE)
    {
        uint32_t idx[_DEFAULT_BLOCK_SIZE];
        services::internal::service_memset_seq<uint32_t, avx512>(idx, uint32_t(0), _DEFAULT_BLOCK_SIZE);

        __mmask16 isSplit = 0xffff;

        __m512i offset = _mm512_set_epi32(15*nCols, 14*nCols, 13*nCols,12*nCols, 11*nCols, 10*nCols, 9*nCols, 8*nCols,
                                          7*nCols, 6*nCols, 5*nCols, 4*nCols, 3*nCols, 2*nCols, nCols, 0);

        __mmask16 checkMask = feat_idx[0] != -1;

        __m512i nOne = _mm512_set1_epi32(-1);
        __m512i zero = _mm512_set1_epi32(0);
        __m512 zero_ps = _mm512_set1_ps(0);
        __m512i one  = _mm512_set1_epi32(1);

        while(checkMask)
        {

            checkMask = 0x0000;
            size_t i = 0;
            for(size_t i = 0; i < _DEFAULT_BLOCK_SIZE; i += 16) {
                __m512i idxr =_mm512_castps_si512(_mm512_loadu_ps((float*)(idx + i)));
                __m512  sp   = _mm512_i32gather_ps(idxr, split_point, 4);

                __m512i left = _mm512_i32gather_epi32(idxr, left_son, 4);

                __m512i fi   = _mm512_i32gather_epi32(idxr, feat_idx, 4);

                isSplit = _mm512_cmp_epi32_mask(fi, nOne, _MM_CMPINT_NE);

                __m512 X  = _mm512_mask_i32gather_ps(zero_ps, isSplit, _mm512_add_epi32(offset, fi), x + i*nCols, 4);

                __mmask16 res = _mm512_cmp_ps_mask(X, sp, _CMP_GT_OS);
                __m512i next_indexes = _mm512_mask_add_epi32(zero, res, one, zero);
                __m512i reservedLeft = _mm512_add_epi32(next_indexes, left);

                _mm512_mask_storeu_epi32(idx + i, isSplit, reservedLeft);

                checkMask =  _kor_mask16(checkMask, isSplit);
            }
        }
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < sizeOfBlock; i++)
        {
            const size_t cl = left_son[idx[i]];
            res[i*_nClasses + cl]++;
        }
    }
    else
    {
        predictByTreeCommon(x, sizeOfBlock, nCols, feat_idx, left_son, split_point, res);
    }
}


template <>
void PredictClassificationTask<double, avx512>::predictByTree(const double* x, const size_t sizeOfBlock, const size_t nCols,
    const featureIndexType* feat_idx, const leftOrClassType* left_son, const double* split_point, ClassIndexType* res)
{
    if(sizeOfBlock == _DEFAULT_BLOCK_SIZE)
    {
        uint32_t idx[_DEFAULT_BLOCK_SIZE];
        services::internal::service_memset_seq<uint32_t, avx512>(idx, uint32_t(0), _DEFAULT_BLOCK_SIZE);

        __mmask8 isSplit = 1;

        __m256i offset = _mm256_set_epi32(7*nCols, 6*nCols, 5*nCols, 4*nCols, 3*nCols, 2*nCols, nCols, 0);

        __mmask8 checkMask = feat_idx[0] != -1;

        while(checkMask)
        {
            checkMask = 0;
            size_t i = 0;
            for(size_t i = 0; i < _DEFAULT_BLOCK_SIZE; i += 8) {
                __m256i idxr = _mm256_castps_si256(_mm256_loadu_ps((float*)(idx + i)));
                __m512d sp   = _mm512_i32gather_pd(idxr, split_point, 8);

                __m256i left = _mm256_i32gather_epi32(left_son, idxr, 4);

                __m256i fi   = _mm256_i32gather_epi32(feat_idx, idxr, 4);

                isSplit = _mm256_cmp_epi32_mask(fi, _mm256_set1_epi32(-1), _MM_CMPINT_NE);

                __m512d X  = _mm512_mask_i32gather_pd(_mm512_set1_pd(0), isSplit, _mm256_add_epi32(offset, fi), x + i*nCols, 8);

                __mmask8 res = _mm512_cmp_pd_mask(X, sp, _CMP_GT_OS);

                __m256i next_indexes = _mm256_mask_add_epi32(_mm256_set1_epi32(0), res, _mm256_set1_epi32(1), _mm256_set1_epi32(0));
                __m256i reservedLeft = _mm256_add_epi32(next_indexes, left);

                _mm256_mask_storeu_epi32(idx + i, isSplit, reservedLeft);

                checkMask =  _kor_mask8(checkMask, isSplit);
            }
        }
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < sizeOfBlock; i++)
        {
            const size_t cl = left_son[idx[i]];
            res[i*_nClasses + cl]++;
        }
    }
    else
    {
        predictByTreeCommon(x, sizeOfBlock, nCols, feat_idx, left_son, split_point, res);
    }
}
#endif

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
            predictByTrees(0, nTreesTotal, xBD.get() + iRow*nCols, val);
            res[iRow] = algorithmFPType(getMaxClass(val));
        });
    });
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
Status PredictClassificationTask<algorithmFPType, cpu>::predictAllPointsByAllTrees(size_t nTreesTotal)
{
    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    const size_t numberOfTrees = nTreesTotal;
    const size_t nCols = _data->getNumberOfColumns();

    daal::SafeStatus safeStat;
    const size_t nRowsOfRes= _res->getNumberOfRows();
    const size_t blockSize = cpu == avx512 ? _DEFAULT_BLOCK_SIZE : _DEFAULT_BLOCK_SIZE_COMMON;
    const size_t nBlocks = nRowsOfRes / blockSize;
    const size_t residualSize = nRowsOfRes - nBlocks * blockSize;

    services::internal::TArray<ClassIndexType, cpu> commonBufValT(_nClasses*nRowsOfRes);
    ClassIndexType* commonBufVal = commonBufValT.get();

    services::internal::service_memset<ClassIndexType, cpu>(commonBufVal, ClassIndexType(0), _nClasses*nRowsOfRes);

    algorithmFPType* const res = resBD.get();
    ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), 0, nRowsOfRes);
    DAAL_CHECK_BLOCK_STATUS(xBD);
    const algorithmFPType* const aX = xBD.get();

    daal::TlsMem<ClassIndexType, cpu, services::internal::ScalableCalloc<ClassIndexType, cpu>> tlsData(_nClasses*nRowsOfRes);

    if(numberOfTrees > _MIN_TREES_FOR_THREADING)
    {
        daal::threader_for(numberOfTrees, numberOfTrees, [&,nCols](const size_t iTree)
        {
            const size_t treeSize = _aTree[iTree]->getNumberOfRows();
            const DecisionTreeNode* aNode = (const DecisionTreeNode*)(*_aTree[iTree]).getArray();
            parallelPredict(aX, aNode, treeSize, nBlocks, nCols, blockSize, residualSize, tlsData.local());
        });
        if(threader_get_threads_number())
        {
            tlsData.reduce([&](ClassIndexType* buf)
            {
                for(size_t i = 0; i < nRowsOfRes; i++)
                    for(size_t j = 0; j < _nClasses; j++)
                        commonBufVal[i*_nClasses + j] += buf[i*_nClasses + j];
            });
        }
        else
        {
            commonBufVal = tlsData.local();
        }
    }
    else
    {
        for(size_t iTree = 0; iTree < numberOfTrees; iTree++)
        {
            const size_t treeSize = _aTree[iTree]->getNumberOfRows();
            const DecisionTreeNode* aNode = (const DecisionTreeNode*)(*_aTree[iTree]).getArray();
            parallelPredict(aX, aNode, treeSize, nBlocks, nCols, blockSize, residualSize, commonBufVal);
        }
    }

    for(size_t iRes = 0; iRes < nRowsOfRes; iRes++)
        res[iRes] = algorithmFPType(getMaxClass(commonBufVal + iRes*_nClasses));

    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
Status PredictClassificationTask<algorithmFPType, cpu>::run(services::HostAppIface* pHostApp)
{
    DAAL_CHECK_MALLOC(_featHelper.init(*_data));
    const auto nTreesTotal = _model->size();
    _aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(_aTree.get());
    size_t averageTreeSize = 0;
    for(size_t i = 0; i < nTreesTotal; ++i)
    {
        _aTree[i] = _model->at(i);
        averageTreeSize += _aTree[i]->getNumberOfRows();
    }
    averageTreeSize = averageTreeSize / nTreesTotal;

    if(_featHelper.hasUnorderedFeatures() || (_res->getNumberOfRows() < averageTreeSize*_SCALE_FACTOR_FOR_VECT_PARALLEL_COMPUTE && daal::threader_get_threads_number() > 1)
        || (_res->getNumberOfRows() < _MIN_NUMBER_OF_ROWS_FOR_VECT_SEQ_COMPUTE && daal::threader_get_threads_number() == 1))
    {
        const auto treeSize = _aTree[0]->getNumberOfRows()*sizeof(dtrees::internal::DecisionTreeNode);
        DimType dim(*_data, nTreesTotal, treeSize, _nClasses);

        if(dim.nTreeBlocks == 1) //all fit into LL cache
            return predictByAllTrees(nTreesTotal, dim);

        services::internal::TArrayCalloc<ClassIndexType, cpu> aClsCounters(dim.nRowsTotal*_nClasses);
        if(!aClsCounters.get())
            return predictByAllTrees(nTreesTotal, dim);

        return predictByBlocksOfTrees(pHostApp, nTreesTotal, dim, aClsCounters.get());
    }
    else
    {
        return predictAllPointsByAllTrees(nTreesTotal);
    }
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
        DAAL_CHECK_STATUS_VAR(s);
        if(host.isCancelled(s, 1))
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

            if(nRowsToProcess < 2 * nThreads || cpu == __avx512_mic__)
            {
                for(size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
                {
                    ClassIndexType* countsForTheRow = counts + iRow*_nClasses;
                    predictByTrees(iTree, nTreesToUse, xBD.get() + iRow*dim.nCols, countsForTheRow);
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
                    predictByTrees(iTree, nTreesToUse, xBD.get() + iRow*dim.nCols, countsForTheRow);
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

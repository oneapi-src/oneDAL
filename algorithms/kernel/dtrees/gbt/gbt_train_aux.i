/* file: gbt_train_aux.i */
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
//  Implementation of auxiliary functions for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_AUX_I__
#define __GBT_TRAIN_AUX_I__

#include "dtrees_model_impl.h"
#include "dtrees_train_data_helper.i"
#include "gbt_internal.h"
#include "threading.h"
#include "gbt_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::internal;

template <CpuType cpu>
void deleteTables(gbt::internal::GbtDecisionTree** aTbl, HomogenNumericTable<double>** aTblImp, HomogenNumericTable<int>** aTblSmplCnt, size_t n)
{
    for(size_t i = 0; i < n; ++i)
    {
        delete aTbl[i];
        delete aTblImp[i];
        delete aTblSmplCnt[i];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Data helper class for regression
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, CpuType cpu>
class OrderedRespHelper : public DataHelperBase<algorithmFPType, cpu>
{
public:
    typedef DataHelperBase<algorithmFPType, cpu> super;
public:
    OrderedRespHelper(const dtrees::internal::IndexedFeatures* indexedFeatures) : super(indexedFeatures), _aIdxToRow(nullptr){}
    const algorithmFPType* y() const { return _y.get(); }

    services::Status init(const NumericTable* data, const NumericTable* resp, const IndexType* aIdxToRow)
    {
        super::init(data, resp);
        _y.reset(data->getNumberOfRows());
        DAAL_CHECK_MALLOC(_y.get());
        ReadRows<algorithmFPType, cpu> bd(const_cast<NumericTable*>(resp), 0, _y.size());
        services::internal::tmemcpy<algorithmFPType, cpu>(_y.get(), bd.get(), _y.size());
        setIdxToRowMapping(aIdxToRow);
        return services::Status();
    }
    void setIdxToRowMapping(const IndexType* aIdxToRow) { _aIdxToRow = aIdxToRow; }

    void getColumnValues(size_t iCol, const IndexType* aIdx, size_t n, algorithmFPType* aVal) const
    {
        if(this->_dataDirect)
        {
            if(_aIdxToRow)
            {
                for(size_t i = 0; i < n; ++i)
                    aVal[i] = this->_dataDirect[_aIdxToRow[aIdx[i]] * this->_nCols + iCol];
            }
            else
            {
                for(size_t i = 0; i < n; ++i)
                    aVal[i] = this->_dataDirect[aIdx[i] * this->_nCols + iCol];
            }
        }
        else
        {
            data_management::BlockDescriptor<algorithmFPType> bd;
            if(_aIdxToRow)
            {
                for(size_t i = 0; i < n; ++i)
                {
                    this->_data->getBlockOfColumnValues(iCol, _aIdxToRow[aIdx[i]], 1, readOnly, bd);
                    aVal[i] = *bd.getBlockPtr();
                    this->_data->releaseBlockOfColumnValues(bd);
                }
            }
            else
            {
                for(size_t i = 0; i < n; ++i)
                {
                    this->_data->getBlockOfColumnValues(iCol, aIdx[i], 1, readOnly, bd);
                    aVal[i] = *bd.getBlockPtr();
                    this->_data->releaseBlockOfColumnValues(bd);
                }
            }
        }
    }

    bool hasDiffFeatureValues(IndexType iFeature, const IndexType* aIdx, size_t n) const
    {
        if(this->indexedFeatures().numIndices(iFeature) == 1)
            return false; //single value only
        const IndexedFeatures::IndexType* indexedFeature = this->indexedFeatures().data(iFeature);
        size_t i = 1;
        if(_aIdxToRow)
        {
            const IndexedFeatures::IndexType idx0 = indexedFeature[_aIdxToRow[aIdx[0]]];
            for(; i < n; ++i)
            {
                const IndexedFeatures::IndexType idx = indexedFeature[_aIdxToRow[aIdx[i]]];
                if(idx != idx0)
                    break;
            }
        }
        else
        {
            const IndexedFeatures::IndexType idx0 = indexedFeature[aIdx[0]];
            for(; i < n; ++i)
            {
                const IndexedFeatures::IndexType idx = indexedFeature[aIdx[i]];
                if(idx != idx0)
                    break;
            }
        }
        return (i != n);
    }

protected:
    TArray<algorithmFPType, cpu> _y;
    const IndexType* _aIdxToRow; //for the methods that take in array of indices, this is the
                                 // mapping of the index to the row, if required
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, pair (gradient, hessian) of algorithmFPType values
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, CpuType cpu>
struct gh
{
    algorithmFPType g; //gradient
    algorithmFPType h; //hessian
    gh() : g(0), h(0){}
    gh(algorithmFPType _g, algorithmFPType _h) : g(_g), h(_h){}
    gh(const gh& o) : g(o.g), h(o.h){}
    gh(const gh& total, const gh& part) : g(total.g - part.g), h(total.h - part.h){}
    gh& operator =(const gh& o) { g = o.g;  h = o.h; return *this; }
    void reset(algorithmFPType _g, algorithmFPType _h) { g = _g;  h = _h; }
    void add(const gh& o) { g += o.g;  h += o.h; }
    algorithmFPType value(algorithmFPType regLambda) const { return (g / (h + regLambda))*g; }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Impurity data
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, CpuType cpu>
using ImpurityData = gh<algorithmFPType, cpu>;

//////////////////////////////////////////////////////////////////////////////////////////
// Base class for loss function L(y,f), where y is a response value,
// f is its current approximation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class LossFunction : public Base
{
public:
    virtual void getGradients(size_t n,
        const algorithmFPType* y, const algorithmFPType* f,
        const IndexType* sampleInd,
        algorithmFPType* gh) = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, the sum of pairs (gradient, hessian) corresponding to the same value of indexed feature
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, CpuType cpu>
struct ghSum : public gh<algorithmFPType, cpu>
{
    ghSum() : gh<algorithmFPType, cpu>(), n(0){}
    IndexType n;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Base memory helper class
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperBase : public Base
{
protected:
    MemHelperBase(size_t nFeaturesIdx) : _nFeaturesIdx(nFeaturesIdx){}

public:
    typedef gh<algorithmFPType, cpu> ghType;
    typedef ghSum<algorithmFPType, cpu> ghSumType;
    typedef TVector<IndexType, cpu> IndexTypeVector;
    typedef TVector<ghType, cpu> ghTypeVector;
    typedef TVector<ghSumType, cpu> ghSumTypeVector;
    typedef TVector<algorithmFPType, cpu> algorithmFPTypeVector;

    virtual bool init() = 0;
    //get buffer for the indices of features to be used for the split at the current level
    virtual IndexType* getFeatureSampleBuf() = 0;
    //release the buffer
    virtual void releaseFeatureSampleBuf(IndexType* buf) = 0;

    //get buffer for ghSums to be used for the split of an indexed feature at the current level
    virtual ghSumType* getGHSumBuf(size_t size) = 0;

    //get buffer for the feature values to be used for the split at the current level
    virtual algorithmFPTypeVector* getFeatureValueBuf(size_t size) = 0;
    //release the buffer
    virtual void releaseFeatureValueBuf(algorithmFPTypeVector* buf) = 0;

    //get buffer for the indexes of the sorted feature values to be used for the split at the current level
    virtual IndexTypeVector* getSortedFeatureIdxBuf(size_t size) = 0;
    //release the buffer
    virtual void releaseSortedFeatureIdxBuf(IndexTypeVector* p) = 0;

protected:
    const size_t _nFeaturesIdx;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation of memory helper for sequential version
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperSeq : public MemHelperBase<algorithmFPType, cpu>
{
public:
    typedef MemHelperBase<algorithmFPType, cpu> super;
    MemHelperSeq(size_t nFeaturesIdx, size_t nDiffFeaturesMax, size_t nFeatureValuesMax) :
        super(nFeaturesIdx), _featureSample(nFeaturesIdx), _aGHSum(nDiffFeaturesMax), _aFeatureValue(nFeatureValuesMax){}

    virtual bool init() DAAL_C11_OVERRIDE
    {
        return (!_featureSample.size() || _featureSample.get()) && //not required to allocate or allocated
        (!_aGHSum.size() || _aGHSum.get()) && (!_aFeatureValue.size() || _aFeatureValue.get());
    }

    virtual IndexType* getFeatureSampleBuf() DAAL_C11_OVERRIDE { return _featureSample.get(); }
    virtual void releaseFeatureSampleBuf(IndexType* buf) DAAL_C11_OVERRIDE{}

    virtual typename super::ghSumType* getGHSumBuf(size_t size) DAAL_C11_OVERRIDE
    {
        if(_aGHSum.size() < size)
            _aGHSum.reset(size);
        return _aGHSum.get();
    }

    //get buffer for the feature values to be used for the split at the current level
    virtual typename super::algorithmFPTypeVector* getFeatureValueBuf(size_t size) DAAL_C11_OVERRIDE
    { DAAL_ASSERT(_aFeatureValue.size() >= size);  return &_aFeatureValue; }
    //release the buffer
    virtual void releaseFeatureValueBuf(typename super::algorithmFPTypeVector* buf) DAAL_C11_OVERRIDE{}

    virtual typename super::IndexTypeVector* getSortedFeatureIdxBuf(size_t size) DAAL_C11_OVERRIDE
    { DAAL_ASSERT(false);  return nullptr; }//should never be called

    virtual void releaseSortedFeatureIdxBuf(typename super::IndexTypeVector* p) DAAL_C11_OVERRIDE{}

protected:
    typename super::IndexTypeVector _featureSample;
    typename super::ghSumTypeVector _aGHSum;
    typename super::algorithmFPTypeVector _aFeatureValue;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, keeps an array in ls and resizes it in local()
//////////////////////////////////////////////////////////////////////////////////////////
template<typename VectorType>
class lsVector : public ls<VectorType*>
{
public:
    typedef ls<VectorType*> super;
    explicit lsVector() : super([=]()->VectorType*{ return new VectorType(); }){}
    ~lsVector() { this->reduce([](VectorType* ptr) { if(ptr) delete ptr; }); }
    VectorType* local(size_t size)
    {
        auto ptr = super::local();
        if(ptr && (ptr->size() < size))
        {
            ptr->reset(size);
            if(!ptr->get())
            {
                this->release(ptr);
                ptr = nullptr;
            }
        }
        return ptr;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, keeps an array in tls and resizes it in local()
//////////////////////////////////////////////////////////////////////////////////////////
template<typename VectorType>
class tlsVector : public tls<VectorType*>
{
public:
    typedef tls<VectorType*> super;
    explicit tlsVector() : super([=]()->VectorType*{ return new VectorType(); }){}
    ~tlsVector() { this->reduce([](VectorType* ptr) { if(ptr) delete ptr; }); }
    VectorType* local(size_t size)
    {
        auto ptr = super::local();
        if(ptr && (ptr->size() < size))
        {
            ptr->reset(size);
            if(!ptr->get())
                ptr = nullptr;
        }
        return ptr;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation of memory helper for threaded version
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperThr : public MemHelperBase<algorithmFPType, cpu>
{
public:
    typedef MemHelperBase<algorithmFPType, cpu> super;
    MemHelperThr(size_t nFeaturesIdx) : super(nFeaturesIdx),
        _lsFeatureSample([=]()->IndexType*{ return services::internal::service_scalable_malloc<IndexType, cpu>(this->_nFeaturesIdx); })
    {
    }
    ~MemHelperThr()
    {
        _lsFeatureSample.reduce([](IndexType* ptr){ if(ptr) services::internal::service_scalable_free<IndexType, cpu>(ptr); });
    }
public:
    virtual bool init() DAAL_C11_OVERRIDE { return true; }
    virtual IndexType* getFeatureSampleBuf() DAAL_C11_OVERRIDE
    {
        return _lsFeatureSample.local();
    }

    virtual void releaseFeatureSampleBuf(IndexType* p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsFeatureSample.release(p);
    }

    virtual typename super::ghSumType* getGHSumBuf(size_t size) DAAL_C11_OVERRIDE
    {
        auto ptr = _tlsGHSum.local(size);
        return ptr ? ptr->get() : nullptr;
    }

    //get buffer for the feature values to be used for the split at the current level
    virtual typename super::algorithmFPTypeVector* getFeatureValueBuf(size_t size) DAAL_C11_OVERRIDE
    {
        return _lsFeatureValueBuf.local(size);
    }

    //release the buffer
    virtual void releaseFeatureValueBuf(typename super::algorithmFPTypeVector* p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsFeatureValueBuf.release(p);
    }

    virtual typename super::IndexTypeVector* getSortedFeatureIdxBuf(size_t size) DAAL_C11_OVERRIDE
    {
        return _lsSortedFeatureIdxBuf.local(size);
    }

    virtual void releaseSortedFeatureIdxBuf(typename super::IndexTypeVector* p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsSortedFeatureIdxBuf.release(p);
    }

protected:
    ls<IndexType*> _lsFeatureSample;
    tlsVector<typename super::ghSumTypeVector> _tlsGHSum;
    lsVector<typename super::algorithmFPTypeVector> _lsFeatureValueBuf;
    lsVector<typename super::IndexTypeVector> _lsSortedFeatureIdxBuf;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Job to be performed in one node
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
struct SplitJob
{
public:
    using TreeType = gbt::internal::TreeImpRegression<>;
    using NodeType = TreeType::NodeType;
    using ImpurityType = ImpurityData<algorithmFPType, cpu>;

    SplitJob(const SplitJob& o): iStart(o.iStart), n(o.n), level(o.level), imp(o.imp), res(o.res){}
    SplitJob(size_t _iStart, size_t _n, size_t _level, const ImpurityType& _imp, NodeType::Base*& _res) :
        iStart(_iStart), n(_n), level(_level), imp(_imp), res(_res){}
public:
    const size_t iStart;
    const size_t n;
    const size_t level;
    const ImpurityType imp;
    NodeType::Base*& res;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Base tree builder class
//////////////////////////////////////////////////////////////////////////////////////////
class TreeBuilderBase : public Base
{
public:
    virtual services::Status init() = 0;
    virtual services::Status run(gbt::internal::GbtDecisionTree*& pRes, HomogenNumericTable<double>*& pTblImp,
        HomogenNumericTable<int>*& pTblSmplCnt, size_t iTree) = 0;
};

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

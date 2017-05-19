/* file: df_classification_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for decision forest classification
//  (defaultDense) method.
//--
*/

#ifndef __DF_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DF_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "df_train_dense_default_impl.i"
#include "df_classification_train_kernel.h"
#include "df_classification_model_impl.h"
#include "df_predict_dense_default_impl.i"

#define OOBClassificationData size_t

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace internal
{

using namespace decision_forest::training::internal;

//////////////////////////////////////////////////////////////////////////////////////////
// UnorderedRespHelper
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class UnorderedRespHelper : public DataHelper<algorithmFPType, ClassIndexType, cpu>
{
public:
    typedef ClassIndexType TResponse;
    typedef DataHelper<algorithmFPType, ClassIndexType, cpu> super;
    typedef typename decision_forest::internal::TreeImpClassification<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef typename daal::algorithms::decision_forest::internal::TVector<size_t, cpu> Histogramm;

    struct ImpurityData
    {
        algorithmFPType var; //impurity is a variance
        Histogramm      hist;
        algorithmFPType value() const { return var; }

        ImpurityData(){}
        ImpurityData(const ImpurityData& o): var(o.var), hist(o.hist){}

        void init(size_t nClasses) { var = 0; hist.resize(nClasses, 0); }
    };
    typedef SplitData<algorithmFPType, ImpurityData> TSplitData;

public:
    UnorderedRespHelper(size_t nClasses): _nClasses(nClasses) {}

    void calcImpurity(const IndexType* aIdx, size_t n, ImpurityData& imp) const;
    bool findBestSplitForFeature(const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, size_t nMinSplitPart, const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const
    {
        return split.featureUnordered ? findBestSplitCategoricalFeature(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split) :
            findBestSplitOrderedFeature(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split);
    }
    void simpleSplit(const algorithmFPType* featureVal, const IndexType* aIdx, TSplitData& split) const;

    TResponse predict(const decision_forest::internal::Tree& t, const algorithmFPType* x) const
    {
        const TreeType& tree = static_cast<const TreeType&>(t);
        const typename TreeType::NodeType::Base* pNode = decision_forest::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x);
        DAAL_ASSERT(pNode);
        return TreeType::NodeType::castLeaf(pNode)->response.value;
    }

    algorithmFPType predictionError(TResponse prediction, TResponse response) const
    {
        return algorithmFPType(prediction != response);
    }

    algorithmFPType predictionError(const decision_forest::internal::Tree& t, const algorithmFPType* x,
        const NumericTable* resp, size_t iRow, byte* oobBuf) const
    {
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable*>(resp), iRow, 1);
        const TResponse response(this->predict(t, x));
        if(oobBuf)
        {
            OOBClassificationData* ptr = ((OOBClassificationData*)oobBuf) + _nClasses*iRow;
            ptr[response]++;
        }
        return this->predictionError(response, *y.get());
    }

    typename TreeType::NodeType::Leaf* makeLeaf(const IndexType* idx, size_t n, ImpurityData& imp) const
    {
        DAAL_ASSERT(n > 0);
        typename NodeType::Leaf* pNode = TreeType::Allocator::allocLeaf();
        pNode->count = n;
        pNode->impurity = imp.var;
#ifdef DEBUG_CHECK_IMPURITY
        {
            Histogramm res(_nClasses, 0);
            for(size_t i = 0; i < n; ++i)
            {
                const ClassIndexType iClass = this->_aResponse[idx[i]].val;
                res[iClass] += 1;
            }
            for(size_t i = 0; i < _nClasses; ++i)
                DAAL_ASSERT(res[i] == imp.hist[i]);
        }
#endif
        auto maxVal = imp.hist[0];
        ClassIndexType maxClass = 0;
        for(size_t i = 1; i < _nClasses; ++i)
        {
            if(maxVal < imp.hist[i])
            {
                maxVal = imp.hist[i];
                maxClass = i;
            }
        }
        pNode->response.value = maxClass;
#ifdef KEEP_CLASSES_PROBABILITIIES
        pNode->response.size = imp.hist.size();
        pNode->response.hist = imp.hist.detach();
#endif
        return pNode;
    }

#ifdef DEBUG_CHECK_IMPURITY
    void checkImpurity(const IndexType* ptrIdx, size_t n, const ImpurityData& expected) const;
#endif

private:
    void calcGini(size_t n, ImpurityData& imp) const
    {
        const algorithmFPType cDiv(1. / (algorithmFPType(n)*algorithmFPType(n)));
        algorithmFPType var(1.);
        for(size_t i = 0; i < _nClasses; ++i)
            var -= cDiv*algorithmFPType(imp.hist[i]) * algorithmFPType(imp.hist[i]);
        imp.var = var;
        if(imp.var < 0)
            imp.var = 0; //roundoff error
    }

    static void calcPrevImpurity(ImpurityData& imp, ClassIndexType x, size_t n, size_t l)
    {
        algorithmFPType delta = (2.*algorithmFPType(n) - algorithmFPType(l))*imp.var + 2.*(algorithmFPType(imp.hist[x]) - algorithmFPType(n));
        imp.var += algorithmFPType(l)*delta / (algorithmFPType(n - l)*algorithmFPType(n - l));
        imp.hist[x] -= l;
    }
    static void flush(ImpurityData& left, ImpurityData& right, ClassIndexType xi, size_t n, size_t k, size_t& ll)
    {
        algorithmFPType tmp = algorithmFPType(k)*(2.*algorithmFPType(ll) + left.var*algorithmFPType(k)) - 2. *algorithmFPType(ll)*algorithmFPType(left.hist[xi]);
        left.hist[xi] += ll;
        left.var = tmp / (algorithmFPType(k + ll)*algorithmFPType(k + ll));
        calcPrevImpurity(right, xi, n - k, ll);
        ll = 0;
    }

    bool findBestSplitOrderedFeature(const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, size_t nMinSplitPart, const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const;
    bool findBestSplitCategoricalFeature(const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, size_t nMinSplitPart, const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const;
private:
    size_t _nClasses;
};

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu>
void UnorderedRespHelper<algorithmFPType, cpu>::checkImpurity(const IndexType* ptrIdx, size_t n, const ImpurityData& expected) const
{
    Histogramm hist;
    hist.resize(_nClasses, 0);
    for(size_t i = 0; i < n; ++i)
    {
        const ClassIndexType iClass = _aResponse[ptrIdx[i]].val;
        hist[iClass] += 1;
    }
    const algorithmFPType cDiv(1. / (algorithmFPType(n)*algorithmFPType(n)));
    algorithmFPType var(1.);
    for(size_t i = 0; i < _nClasses; ++i)
        var -= cDiv*algorithmFPType(hist[i]) * algorithmFPType(hist[i]);
    for(size_t i = 0; i < _nClasses; ++i)
        DAAL_ASSERT(hist[i] == expected.hist[i]);
    DAAL_ASSERT(!(fabs(var - expected.var) > 0.001));
}
#endif

template <typename algorithmFPType, CpuType cpu>
void UnorderedRespHelper<algorithmFPType, cpu>::calcImpurity(const IndexType* aIdx, size_t n, ImpurityData& imp) const
{
    imp.init(_nClasses);
    for(size_t i = 0; i < n; ++i)
    {
        const ClassIndexType iClass = this->_aResponse[aIdx[i]].val;
        imp.hist[iClass] += 1;
    }
    calcGini(n, imp);
}

template <typename algorithmFPType, CpuType cpu>
void UnorderedRespHelper<algorithmFPType, cpu>::simpleSplit(const algorithmFPType* featureVal,
    const IndexType* aIdx, TSplitData& split) const
{
    split.featureValue = featureVal[0];
    split.left.init(_nClasses);
    const ClassIndexType iClass1(this->_aResponse[aIdx[0]].val);
    split.left.hist[iClass1] = 1;

    split.right.init(_nClasses);
    const ClassIndexType iClass2(this->_aResponse[aIdx[1]].val);
    split.right.hist[iClass2] = 1;
    split.nLeft = 1;
    split.iStart = 0;
}

template <typename algorithmFPType, CpuType cpu>
bool UnorderedRespHelper<algorithmFPType, cpu>::findBestSplitOrderedFeature(const algorithmFPType* featureVal, const IndexType* aIdx,
    size_t n, size_t nMinSplitPart, const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const
{
    ClassIndexType xi = this->_aResponse[aIdx[0]].val;
    ImpurityData left;
    left.init(_nClasses);
    left.hist[xi] = 1;

    ImpurityData right(curImpurity);
    this->calcPrevImpurity(right, xi, n, 1);
    if(right.var < 0) //roundoff error
        right.var = 0;

#ifdef DEBUG_CHECK_IMPURITY
    checkImpurity(aIdx + 1, n - 1, right);
#endif
    algorithmFPType vBest = -1;
    split.left.var = left.var;
    split.right.var = right.var;
    split.left.hist = left.hist;
    split.right.hist = right.hist;
    split.nLeft = 1;
    IndexType iBest = -1;

    size_t ll = 0;
    const algorithmFPType last = featureVal[n - nMinSplitPart];
    for(size_t i = 1, ll = 0, k = 1; (i < n - nMinSplitPart) && (featureVal[i] < last); ++i)
    {
        const bool bSameFeatureNext(featureVal[i + 1] < featureVal[i] + accuracy);
        if(!ll)
        {
            xi = this->_aResponse[aIdx[i]].val;
            ll = 1;
            k = i;
            if(bSameFeatureNext)
                continue;
        }
        else if(xi == this->_aResponse[aIdx[i]].val)
        {
            ++ll;
            if(bSameFeatureNext)
                continue;
        }
        else
        {
            flush(left, right, xi, n, k, ll);
#ifdef DEBUG_CHECK_IMPURITY
            checkImpurity(aIdx, i, left);
            checkImpurity(aIdx + i, n - i, right);
#endif
            xi = this->_aResponse[aIdx[i]].val;
            ll = 1;
            k = i;
        }

        flush(left, right, xi, n, k, ll);
#ifdef DEBUG_CHECK_IMPURITY
        checkImpurity(aIdx, i + 1, left);
        checkImpurity(aIdx + i + 1, n - i - 1, right);
#endif
        if(bSameFeatureNext)
            continue;

        if(left.var < 0)
            left.var = 0;
        if(right.var < 0)
            right.var = 0;

        if((iBest < 0) && (i + 1 < nMinSplitPart))
            continue;//skip first nMinSplitPart - 1 samples

        const algorithmFPType v = algorithmFPType(i)*left.var + algorithmFPType(n - i)*right.var;
        if(iBest < 0 || v < vBest)
        {
            vBest = v;
            split.left.var = left.var;
            split.right.var = right.var;
            split.left.hist = left.hist;
            split.right.hist = right.hist;
            iBest = i;
            split.nLeft = i + 1;
        }
    }

    if(iBest < 0)
    {
        if(split.nLeft < nMinSplitPart)
            return false;
        iBest = 0;
    }

    DAAL_ASSERT(!ll);

#ifdef DEBUG_CHECK_IMPURITY
    checkImpurity(aIdx, split.nLeft, split.left);
    checkImpurity(aIdx + split.nLeft, n - split.nLeft, split.right);
#endif
    split.featureValue = featureVal[iBest];
    split.iStart = 0;
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool UnorderedRespHelper<algorithmFPType, cpu>::findBestSplitCategoricalFeature(const algorithmFPType* featureVal, const IndexType* aIdx,
    size_t n, size_t nMinSplitPart, const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const
{
    DAAL_ASSERT(n >= 2*nMinSplitPart);
    ImpurityData left;
    ImpurityData right;
    right.init(_nClasses);
    bool bFound = false;
    algorithmFPType vBest;
    for(size_t i = 0; i < n - nMinSplitPart;)
    {
        size_t count = 1;
        left.init(_nClasses);
        const algorithmFPType first = featureVal[i];
        ClassIndexType xi = this->_aResponse[aIdx[i]].val;
        left.hist[xi] = 1;
        const size_t iStart = i;
        for(++i; (i < n) && (featureVal[i] == first); ++count, ++i)
        {
            xi = this->_aResponse[aIdx[i]].val;
            ++left.hist[xi];
        }
        if(count < nMinSplitPart)
            continue;
        for(size_t j = 0; j < _nClasses; ++j)
            right.hist[j] = curImpurity.hist[j] - left.hist[j];
        calcGini(count, left);
        calcGini(n - count, right);
        const algorithmFPType v = algorithmFPType(count)*left.var + algorithmFPType(n - count)*right.var;
        if(!bFound || v < vBest)
        {
            vBest = v;
            split.left.var = left.var;
            split.right.var = right.var;
            split.left.hist = left.hist;
            split.right.hist = right.hist;
            split.nLeft = count;
            split.iStart = iStart;
            split.featureValue = first;
            bFound = true;
        }
    }
    return bFound;
}

//////////////////////////////////////////////////////////////////////////////////////////
// TreeThreadCtx class for classification
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class TreeThreadCtx : public TreeThreadCtxBase<algorithmFPType, cpu>
{
public:
    typedef TreeThreadCtxBase<algorithmFPType, cpu> super;
    TreeThreadCtx(algorithmFPType* _varImp = nullptr) : super(_varImp){}
    bool init(const decision_forest::training::Parameter& par, const NumericTable* x, size_t nClasses)
    {
        if(!super::init(par, x))
            return false;
        _nClasses = nClasses;
        if(par.resultsToCompute & decision_forest::training::computeOutOfBagError)
        {
            size_t sz = sizeof(OOBClassificationData)*nClasses*x->getNumberOfRows();
            this->oobBuf = service_calloc<byte, cpu>(sz);
            if(!this->oobBuf)
                return false;
        }
        return true;
    }

    void reduceTo(TreeThreadCtx& other, size_t nVars, size_t nSamples) const
    {
        super::reduceTo(other, nVars, nSamples);
        if(this->oobBuf)
        {
            OOBClassificationData* dst = (OOBClassificationData*)other.oobBuf;
            const OOBClassificationData* src = (const OOBClassificationData*)this->oobBuf;
            for(size_t i = 0, n = _nClasses*nSamples; i < n; ++i)
                dst[i] += src[i];
        }
    }

    Status finalizeOOBError(const NumericTable* resp, algorithmFPType& res) const
    {
        DAAL_ASSERT(this->oobBuf);
        const size_t nSamples = resp->getNumberOfRows();
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable*>(resp), 0, nSamples);
        DAAL_CHECK_BLOCK_STATUS(y);
        size_t nPredicted = 0.;
        size_t nError = 0.;
        for(size_t i = 0; i < nSamples; ++i)
        {
            const OOBClassificationData* ptr = ((const OOBClassificationData*)this->oobBuf) + i*_nClasses;
            size_t maxIdx = 0;
            OOBClassificationData maxVal = ptr[0];
            for(size_t j = 1; j < _nClasses; ++j)
            {
                if(maxVal < ptr[j])
                {
                    maxVal = ptr[j];
                    maxIdx = j;
                }
            }
            if(maxVal == 0)
                continue;
            nPredicted++;
            if(maxIdx != size_t(y.get()[i]))
                ++nError;
        }
        res = algorithmFPType(nError) / algorithmFPType(nPredicted);
        return Status();
    }


private:
    size_t _nClasses;
};


//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchTask for classification
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, decision_forest::classification::training::Method method, CpuType cpu>
class TrainBatchTask : public TrainBatchTaskBase<algorithmFPType, UnorderedRespHelper<algorithmFPType, cpu>, cpu>
{
    typedef TrainBatchTaskBase<algorithmFPType, UnorderedRespHelper<algorithmFPType, cpu>, cpu> super;
public:
    typedef TreeThreadCtx<algorithmFPType, cpu> ThreadCtxType;
    TrainBatchTask(size_t seed, const NumericTable *x, const NumericTable *y,
        const decision_forest::training::Parameter& par,
        const FeatureHelper<cpu>& featHelper, typename super::ThreadCtxType& ctx, size_t dummy) :
        super(seed, x, y, par, featHelper, ctx, dummy)
    {
        if(!this->_nFeaturesPerNode)
        {
            size_t nF(daal::internal::Math<algorithmFPType, cpu>::sSqrt(x->getNumberOfColumns()));
            const_cast<size_t&>(this->_nFeaturesPerNode) = (nF < 1 ? 1 : nF);
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// ClassificationTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status ClassificationTrainBatchKernel<algorithmFPType, method, cpu>::compute(
    const NumericTable *x, const NumericTable *y, decision_forest::classification::Model& m,
    Result& res,
    const decision_forest::classification::training::Parameter& par)
{
    ResultData rd(par, res.get(variableImportance).get(), res.get(outOfBagError).get());
    return computeImpl<algorithmFPType, cpu,
        daal::algorithms::decision_forest::classification::internal::ModelImpl,
        TrainBatchTask<algorithmFPType, method, cpu> >
        (x, y, *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl*>(&m),
        rd, par, par.nClasses);
}

} /* namespace internal */
} /* namespace training */
} /* namespace classification */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

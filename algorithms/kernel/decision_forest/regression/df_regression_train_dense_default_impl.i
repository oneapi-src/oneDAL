/* file: df_regression_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for decision forest regression
//  (defaultDense) method.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DF_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "df_train_dense_default_impl.i"
#include "df_regression_train_kernel.h"
#include "df_regression_model_impl.h"
#include "df_predict_dense_default_impl.i"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace internal
{

using namespace decision_forest::training::internal;

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains regression error data for OOB calculation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
struct RegErr
{
    RegErr() : count(0), value(0){}
    algorithmFPType value;
    size_t count;
    void add(const RegErr& o) { count += o.count; value += o.value; }
};

//////////////////////////////////////////////////////////////////////////////////////////
// OrderedRespHelper
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class OrderedRespHelper : public DataHelper<algorithmFPType, algorithmFPType, cpu>
{
public:
    typedef algorithmFPType TResponse;
    typedef DataHelper<algorithmFPType, algorithmFPType, cpu> super;
    typedef decision_forest::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;

    struct ImpurityData
    {
        algorithmFPType var; //impurity is a variance
        algorithmFPType mean;
        algorithmFPType value() const { return var; }
    };
    typedef SplitData<algorithmFPType, ImpurityData> TSplitData;

public:
    OrderedRespHelper(size_t dummy){}
    void calcImpurity(const IndexType* aIdx, size_t n, ImpurityData& imp) const;
    bool findBestSplitForFeature(const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, size_t nMinSplitPart, const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const;
    void simpleSplit(const algorithmFPType* featureVal, const IndexType* aIdx, TSplitData& split) const;

    TResponse predict(const decision_forest::internal::Tree& t, const algorithmFPType* x) const
    {
        const typename TreeType::NodeType::Base* pNode =
            decision_forest::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x);
        DAAL_ASSERT(pNode);
        return pNode ? TreeType::NodeType::castLeaf(pNode)->response : 0.;
    }

    algorithmFPType predictionError(TResponse prediction, TResponse response) const
    {
        return (prediction - response)*(prediction - response);
    }

    algorithmFPType predictionError(const decision_forest::internal::Tree& t, const algorithmFPType* x,
        const NumericTable* resp, size_t iRow, byte* oobBuf) const
    {
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable*>(resp), iRow, 1);
        const TResponse response(this->predict(t, x));
        algorithmFPType val = this->predictionError(response, *y.get());
        if(oobBuf)
        {
            ((RegErr<algorithmFPType, cpu>*)oobBuf)[iRow].value += response;
            ((RegErr<algorithmFPType, cpu>*)oobBuf)[iRow].count++;
        }
        return val;
    }

    typename TreeType::NodeType::Leaf* makeLeaf(const IndexType* idx, size_t n, ImpurityData& imp) const
    {
        DAAL_ASSERT(n > 0);
        typename TreeType::NodeType::Leaf* pNode = TreeType::Allocator::allocLeaf();
        pNode->response = imp.mean;
#ifdef DEBUG_CHECK_IMPURITY
        algorithmFPType val = calcResponse(pNode->response, idx, n);
        DAAL_ASSERT(fabs(val - imp.mean) < 0.001);
#endif
        pNode->count = n;
        pNode->impurity = imp.var;
        return pNode;
    }

#ifdef DEBUG_CHECK_IMPURITY
    void checkImpurity(const IndexType* ptrIdx, size_t n, const ImpurityData& expected) const;
#endif

private:
#ifdef DEBUG_CHECK_IMPURITY
    algorithmFPType calcResponse(algorithmFPType& res, const IndexType* idx, size_t n) const;
#endif
    bool findBestSplitOrderedFeature(const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, size_t nMinSplitPart, const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const;
    bool findBestSplitCategoricalFeature(const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, size_t nMinSplitPart, const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const;
};

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu>
void OrderedRespHelper<algorithmFPType, cpu>::checkImpurity(const IndexType* ptrIdx, size_t n, const ImpurityData& expected) const
{
    algorithmFPType div = 1. / algorithmFPType(n);
    TResponse cMean = this->_aResponse[ptrIdx[0]].val*div;
    for(size_t i = 1; i < n; ++i)
        cMean += this->_aResponse[ptrIdx[i]].val*div;
    algorithmFPType cVar = 0;
    for(size_t i = 0; i < n; ++i)
        cVar += (this->_aResponse[ptrIdx[i]].val - cMean)*(this->_aResponse[ptrIdx[i]].val - cMean);
    DAAL_ASSERT(fabs(cMean - expected.mean) < 0.001);
    DAAL_ASSERT(fabs(cVar - expected.var) < 0.001);
}
#endif

template <typename algorithmFPType, CpuType cpu>
void OrderedRespHelper<algorithmFPType, cpu>::calcImpurity(const IndexType* aIdx, size_t n, ImpurityData& imp) const
{
    imp.var = 0;
    imp.mean = this->_aResponse[aIdx[0]].val;
    for(size_t i = 1; i < n; ++i)
    {
        algorithmFPType delta = this->_aResponse[aIdx[i]].val - imp.mean; //x[i] - mean
        imp.mean += delta / algorithmFPType(i + 1);
        imp.var += delta*(this->_aResponse[aIdx[i]].val - imp.mean);
    }

#ifdef DEBUG_CHECK_IMPURITY
    TResponse mean1 = this->_aResponse[aIdx[0]].val / algorithmFPType(n);
    for(size_t i = 1; i < n; ++i)
        mean1 += this->_aResponse[aIdx[i]].val / algorithmFPType(n);
    algorithmFPType var1 = 0;
    for(size_t i = 0; i < n; ++i)
        var1 += (this->_aResponse[aIdx[i]].val - mean1)*(this->_aResponse[aIdx[i]].val - mean1);
    DAAL_ASSERT(fabs(mean1 - imp.mean) < 0.001);
    DAAL_ASSERT(fabs(var1 - imp.var) < 0.001);
#endif
}

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu>
algorithmFPType OrderedRespHelper<algorithmFPType, cpu>::calcResponse(algorithmFPType& res, const IndexType* idx, size_t n) const
{
    const algorithmFPType cDiv = 1. / algorithmFPType(n);
    res = this->_aResponse[idx[0]].val*cDiv;
    for(size_t i = 1; i < n; ++i)
        res += this->_aResponse[idx[i]].val*cDiv;
    return res;
}
#endif

template <typename algorithmFPType, CpuType cpu>
void calcPrevImpurity(algorithmFPType var, algorithmFPType mean,
    algorithmFPType& varPrev, algorithmFPType& meanPrev, algorithmFPType x, size_t n)
{
    algorithmFPType delta = (x - mean)*algorithmFPType(n) / algorithmFPType(n - 1);
    varPrev = var - delta*(x - mean);
    meanPrev = mean - delta / algorithmFPType(n);
}

template <typename algorithmFPType, CpuType cpu>
void OrderedRespHelper<algorithmFPType, cpu>::simpleSplit(const algorithmFPType* featureVal, const IndexType* aIdx, TSplitData& split) const
{
    split.featureValue = featureVal[0];
    split.left.var = 0;
    split.left.mean = this->_aResponse[aIdx[0]].val;
    split.right.var = 0;
    split.right.mean = this->_aResponse[aIdx[1]].val;
    split.nLeft = 1;
    split.iStart = 1;
}

template <typename algorithmFPType, CpuType cpu>
bool OrderedRespHelper<algorithmFPType, cpu>::findBestSplitForFeature(const algorithmFPType* featureVal,
    const IndexType* aIdx, size_t n, size_t nMinSplitPart,
    const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const
{
    return split.featureUnordered ? findBestSplitCategoricalFeature(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split) :
        findBestSplitOrderedFeature(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split);
}

template <typename algorithmFPType, CpuType cpu>
bool OrderedRespHelper<algorithmFPType, cpu>::findBestSplitOrderedFeature(const algorithmFPType* featureVal,
    const IndexType* aIdx, size_t n, size_t nMinSplitPart,
    const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const
{
    //https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    algorithmFPType xi = this->_aResponse[aIdx[0]].val;
    ImpurityData left;
    left.var = 0;
    left.mean = xi;

    ImpurityData right;
    calcPrevImpurity<algorithmFPType, cpu>(curImpurity.var, curImpurity.mean, right.var, right.mean, xi, n);
    if(right.var < 0)
        right.var = 0;

#ifdef DEBUG_CHECK_IMPURITY
    checkImpurity(aIdx + 1, n - 1, right);
#endif
    algorithmFPType vBest = left.var + right.var;
    split.left.var = left.var;
    split.right.var = right.var;
    split.left.mean = left.mean;
    split.right.mean = right.mean;
    IndexType iBest = -1;

    const algorithmFPType last = featureVal[n - 1];
    for(size_t i = 1; (i < n - 1) && (featureVal[i] < last); ++i)
    {
        xi = this->_aResponse[aIdx[i]].val;
        algorithmFPType delta = xi - left.mean;
        left.mean += delta / algorithmFPType(i + 1);
        left.var += delta * (xi - left.mean);

#ifdef DEBUG_CHECK_IMPURITY
        checkImpurity(aIdx, i + 1, left);
#endif
        calcPrevImpurity<algorithmFPType, cpu>(right.var, right.mean, right.var, right.mean, xi, n - i);

#ifdef DEBUG_CHECK_IMPURITY
        checkImpurity(aIdx + i + 1, n - i - 1, right);
#endif
        if(featureVal[i + 1] < featureVal[i] + accuracy)
            continue;

        if(left.var < 0)
            left.var = 0;
        if(right.var < 0)
            right.var = 0;
        const algorithmFPType v = algorithmFPType(i + 1)*left.var + algorithmFPType(n - i - 1)*right.var;
        if(iBest < 0 || v < vBest)
        {
            vBest = v;
            split.left.var = left.var;
            split.right.var = right.var;
            split.left.mean = left.mean;
            split.right.mean = right.mean;
            iBest = i;
        }
    }
    if(iBest < 0)
        iBest = 0;
    split.nLeft = iBest + 1;
    split.iStart = 0;
#ifdef DEBUG_CHECK_IMPURITY
    checkImpurity(aIdx, split.nLeft, split.left);
    checkImpurity(aIdx + split.nLeft, n - split.nLeft, split.right);
#endif
    split.featureValue = featureVal[iBest];
    return true;
}

template <typename algorithmFPType, CpuType cpu>
void subtractVariance(algorithmFPType var, algorithmFPType mean,
    algorithmFPType var1, algorithmFPType mean1, size_t n1,
    algorithmFPType& var2, algorithmFPType& mean2, size_t n2)
{
    mean2 = mean + (algorithmFPType(n1)*(mean - mean1)) / algorithmFPType(n2);
    const algorithmFPType m1 = mean1 - mean;
    const algorithmFPType m2 = mean2 - mean;
    var2 = var - var1 - algorithmFPType(n1)*m1*m1 - algorithmFPType(n2)*m2*m2;
}

template <typename algorithmFPType, CpuType cpu>
bool OrderedRespHelper<algorithmFPType, cpu>::findBestSplitCategoricalFeature(const algorithmFPType* featureVal,
    const IndexType* aIdx, size_t n, size_t nMinSplitPart,
    const algorithmFPType accuracy, const ImpurityData& curImpurity, TSplitData& split) const
{
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    ImpurityData left;
    ImpurityData right;
    bool bFound = false;
    algorithmFPType vBest;
    size_t nDiffFeatureValues = 0;
    for(size_t i = 0; i < n - nMinSplitPart;)
    {
        ++nDiffFeatureValues;
        size_t count = 1;
        const algorithmFPType first = featureVal[i];
        ClassIndexType xi = this->_aResponse[aIdx[i]].val;
        left.var = 0;
        left.mean = first;
        const size_t iStart = i;
        for(++i; (i < n) && (featureVal[i] == first); ++count, ++i);
        if(count < nMinSplitPart)
            continue;

        if((i == n) && (nDiffFeatureValues == 2) && bFound)
            break; //only 2 feature values, one possible split, already found

        calcImpurity(aIdx + iStart, count, left);
        subtractVariance<algorithmFPType, cpu>(curImpurity.var, curImpurity.mean, left.var, left.mean, count,
            right.var, right.mean, n - count);
#ifdef DEBUG_CHECK_IMPURITY
        if(iStart == 0)
            checkImpurity(aIdx + count, n - count, right);
#endif
        const algorithmFPType v = algorithmFPType(count)*left.var + algorithmFPType(n - count)*right.var;
        if(!bFound || v < vBest)
        {
            vBest = v;
            split.left.var = left.var;
            split.right.var = right.var;
            split.left.mean = left.mean;
            split.right.mean = right.mean;
            split.nLeft = count;
            split.iStart = iStart;
            split.featureValue = first;
            bFound = true;
        }
    }
    return bFound;
}

//////////////////////////////////////////////////////////////////////////////////////////
// TreeThreadCtx class for regression
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class TreeThreadCtx : public TreeThreadCtxBase<algorithmFPType, cpu>
{
public:
    typedef TreeThreadCtxBase<algorithmFPType, cpu> super;
    TreeThreadCtx(algorithmFPType* _varImp = nullptr) : super(_varImp){}
    bool init(const decision_forest::training::Parameter& par, const NumericTable* x, size_t /*dummy*/)
    {
        if(!super::init(par, x))
            return false;
        if(par.resultsToCompute & decision_forest::training::computeOutOfBagError)
        {
            size_t sz = sizeof(RegErr<algorithmFPType, cpu>)*x->getNumberOfRows();
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
            RegErr<algorithmFPType, cpu>* dst = (RegErr<algorithmFPType, cpu>*)other.oobBuf;
            const RegErr<algorithmFPType, cpu>* src = (const RegErr<algorithmFPType, cpu>*)this->oobBuf;
            for(size_t i = 0; i < nSamples; ++i)
                dst[i].add(src[i]);
        }
    }

    Status finalizeOOBError(const NumericTable* resp, algorithmFPType& res) const
    {
        DAAL_ASSERT(this->oobBuf);
        const size_t nSamples = resp->getNumberOfRows();
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable*>(resp), 0, nSamples);
        DAAL_CHECK_BLOCK_STATUS(y);
        size_t nPredicted = 0.;
        res = 0;
        RegErr<algorithmFPType, cpu>* ptr = (RegErr<algorithmFPType, cpu>*)this->oobBuf;
        for(size_t i = 0; i < nSamples; ++i)
        {
            if(ptr[i].count)
            {
                ptr[i].value /= algorithmFPType(ptr[i].count);
                res += (ptr[i].value - y.get()[i])*(ptr[i].value - y.get()[i]);
                ++nPredicted;
            }
        }
        res /= algorithmFPType(nPredicted);
        return Status();
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchTask for regression
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, decision_forest::regression::training::Method method, CpuType cpu>
class TrainBatchTask : public TrainBatchTaskBase<algorithmFPType, OrderedRespHelper<algorithmFPType, cpu>, cpu>
{
    typedef TrainBatchTaskBase<algorithmFPType, OrderedRespHelper<algorithmFPType, cpu>, cpu> super;
public:
    typedef TreeThreadCtx<algorithmFPType, cpu> ThreadCtxType;
    TrainBatchTask(size_t seed, const NumericTable *x, const NumericTable *y,
        const decision_forest::training::Parameter& par,
        const FeatureHelper<cpu>& featHelper, typename super::ThreadCtxType& ctx, size_t dummy) :
        super(seed, x, y, par, featHelper, ctx, dummy)
    {
        if(!this->_nFeaturesPerNode)
        {
            size_t nF = x->getNumberOfColumns()/3;
            const_cast<size_t&>(this->_nFeaturesPerNode) = (nF < 1 ? 1 : nF);
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, decision_forest::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainBatchKernel<algorithmFPType, method, cpu>::compute(
    const NumericTable *x, const NumericTable *y, decision_forest::regression::Model& m, Result& res, const Parameter& par)
{
    ResultData rd(par, res.get(variableImportance).get(), res.get(outOfBagError).get());
    return computeImpl<algorithmFPType, cpu,
        daal::algorithms::decision_forest::regression::internal::ModelImpl,
        TrainBatchTask<algorithmFPType, method, cpu> >
        (x, y, *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl*>(&m),
        rd, par, 0);
}

} /* namespace internal */
} /* namespace training */
} /* namespace regression */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

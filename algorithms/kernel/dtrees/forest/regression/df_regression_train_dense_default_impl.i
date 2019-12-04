/* file: df_regression_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for decision forest regression
//  (defaultDense) method.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DF_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "df_train_dense_default_impl.i"
#include "df_regression_train_kernel.h"
#include "df_regression_model_impl.h"
#include "dtrees_predict_dense_default_impl.i"
#include "df_regression_training_types_result.h"

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
using namespace dtrees::internal;
using namespace dtrees::training::internal;

//computes mean2 and var2 as the mean and mse for the set of elements s2, s2 = s - s1
//where mean, var are mean and mse for s,
//where mean1, var1 are mean and mse for s1
template <typename algorithmFPType, CpuType cpu>
void subtractImpurity(algorithmFPType var, algorithmFPType mean, algorithmFPType var1, algorithmFPType mean1, size_t n1, algorithmFPType & var2,
                      algorithmFPType & mean2, size_t n2)
{
    mean2                   = mean + (algorithmFPType(n1) * (mean - mean1)) / algorithmFPType(n2);
    const algorithmFPType a = (var + mean * mean);
    //const algorithmFPType a1 = (var1 + mean1*mean1);
    const algorithmFPType b = algorithmFPType(n1) / algorithmFPType(n2);
    //var2 = a + (a - a1)*b - mean2*mean2;
    //var2 = var + mean*mean - mean2*mean2 + (var - var1 + mean*mean - mean1*mean1)*b;
    var2 = var + (mean - mean2) * (mean + mean2) + (var - var1 + (mean - mean1) * (mean + mean1)) * b;
    if (var2 < 0) var2 = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains regression error data for OOB calculation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
struct RegErr
{
    algorithmFPType value = 0;
    size_t count          = 0;
    void add(const RegErr & o)
    {
        count += o.count;
        value += o.value;
    }
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
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;

    struct ImpurityData
    {
        algorithmFPType var; //impurity is a variance
        algorithmFPType mean;
        algorithmFPType value() const { return var; }
    };
    typedef SplitData<algorithmFPType, ImpurityData> TSplitData;

public:
    OrderedRespHelper(const dtrees::internal::IndexedFeatures * indexedFeatures, size_t dummy) : super(indexedFeatures) {}
    virtual bool init(const NumericTable * data, const NumericTable * resp, const IndexType * aSample) DAAL_C11_OVERRIDE;
    void convertLeftImpToRight(size_t n, const ImpurityData & total, TSplitData & split)
    {
        subtractImpurity<algorithmFPType, cpu>(total.var, total.mean, split.left.var, split.left.mean, split.nLeft, split.left.var, split.left.mean,
                                               n - split.nLeft);
        split.nLeft = n - split.nLeft;
    }

    void calcImpurity(const IndexType * aIdx, size_t n, ImpurityData & imp) const;
    bool findBestSplitForFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                 const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split) const;
    int findBestSplitForFeatureSorted(algorithmFPType * featureBuf, IndexType iFeature, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                      const ImpurityData & curImpurity, TSplitData & split) const;
    void finalizeBestSplit(const IndexType * aIdx, size_t n, IndexType iFeature, size_t idxFeatureValueBestSplit, TSplitData & bestSplit,
                           IndexType * bestSplitIdx) const;
    void simpleSplit(const algorithmFPType * featureVal, const IndexType * aIdx, TSplitData & split) const;
    bool terminateCriteria(ImpurityData & imp, algorithmFPType impurityThreshold, size_t nSamples) const { return imp.value() < impurityThreshold; }

    TResponse predict(const dtrees::internal::Tree & t, const algorithmFPType * x) const
    {
        const typename TreeType::NodeType::Base * pNode = dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x);
        DAAL_ASSERT(pNode);
        return pNode ? TreeType::NodeType::castLeaf(pNode)->response : 0.;
    }

    algorithmFPType predictionError(TResponse prediction, TResponse response) const { return (prediction - response) * (prediction - response); }

    algorithmFPType predictionError(const dtrees::internal::Tree & t, const algorithmFPType * x, const NumericTable * resp, size_t iRow,
                                    byte * oobBuf) const
    {
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable *>(resp), iRow, 1);
        const TResponse response(this->predict(t, x));
        algorithmFPType val = this->predictionError(response, *y.get());
        if (oobBuf)
        {
            ((RegErr<algorithmFPType, cpu> *)oobBuf)[iRow].value += response;
            ((RegErr<algorithmFPType, cpu> *)oobBuf)[iRow].count++;
        }
        return val;
    }

    void setLeafData(typename TreeType::NodeType::Leaf & node, const IndexType * idx, size_t n, ImpurityData & imp) const
    {
        DAAL_ASSERT(n > 0);
        node.response = imp.mean;
#ifdef DEBUG_CHECK_IMPURITY
        algorithmFPType response;
        algorithmFPType val = calcResponse(response, idx, n);
        node.response       = response;
        DAAL_ASSERT(fabs(val - imp.mean) < 0.001);
#endif
        node.count    = n;
        node.impurity = imp.var;
    }

#ifdef DEBUG_CHECK_IMPURITY
    void checkImpurity(const IndexType * ptrIdx, size_t n, const ImpurityData & expected) const { checkImpurityInternal(ptrIdx, n, expected, false); }
    void checkImpurityInternal(const IndexType * ptrIdx, size_t n, const ImpurityData & expected, bool bInternal = true) const;
#endif

private:
#ifdef DEBUG_CHECK_IMPURITY
    algorithmFPType calcResponse(algorithmFPType & res, const IndexType * idx, size_t n) const;
#endif
    bool findBestSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                     const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split) const;
    bool findBestSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n, size_t nMinSplitPart,
                                         const algorithmFPType accuracy, const ImpurityData & curImpurity, TSplitData & split) const;

private:
    //buffer for the computation using indexed features
    mutable TVector<IndexType, cpu, DefaultAllocator<cpu> > _idxFeatureBuf;
};

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu>
void OrderedRespHelper<algorithmFPType, cpu>::checkImpurityInternal(const IndexType * ptrIdx, size_t n, const ImpurityData & expected,
                                                                    bool bInternal) const
{
    algorithmFPType div = 1. / algorithmFPType(n);
    TResponse cMean     = this->_aResponse[ptrIdx[0]].val * div;
    for (size_t i = 1; i < n; ++i) cMean += this->_aResponse[ptrIdx[i]].val * div;
    algorithmFPType cVar = 0;
    for (size_t i = 0; i < n; ++i) cVar += (this->_aResponse[ptrIdx[i]].val - cMean) * (this->_aResponse[ptrIdx[i]].val - cMean);
    if (!bInternal) cVar *= div;
    DAAL_ASSERT(fabs(cMean - expected.mean) < 0.001);
    DAAL_ASSERT(fabs(cVar - expected.var) < 0.001);
}
#endif

template <typename algorithmFPType, CpuType cpu>
bool OrderedRespHelper<algorithmFPType, cpu>::init(const NumericTable * data, const NumericTable * resp, const IndexType * aSample)
{
    DAAL_CHECK_STATUS_VAR(super::init(data, resp, aSample));
    if (this->_indexedFeatures)
    {
        //init work buffer for the computation using indexed features
        const auto nDiffFeatMax = this->indexedFeatures().maxNumIndices();
        _idxFeatureBuf.reset(nDiffFeatMax);
        return _idxFeatureBuf.get();
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
void OrderedRespHelper<algorithmFPType, cpu>::calcImpurity(const IndexType * aIdx, size_t n, ImpurityData & imp) const
{
    imp.var  = 0;
    imp.mean = this->_aResponse[aIdx[0]].val;
    for (size_t i = 1; i < n; ++i)
    {
        algorithmFPType delta = this->_aResponse[aIdx[i]].val - imp.mean; //x[i] - mean
        imp.mean += delta / algorithmFPType(i + 1);
        imp.var += delta * (this->_aResponse[aIdx[i]].val - imp.mean);
    }
    imp.var /= algorithmFPType(n); //impurity is MSE

#ifdef DEBUG_CHECK_IMPURITY
    TResponse mean1 = this->_aResponse[aIdx[0]].val / algorithmFPType(n);
    for (size_t i = 1; i < n; ++i) mean1 += this->_aResponse[aIdx[i]].val / algorithmFPType(n);
    algorithmFPType var1 = 0;
    for (size_t i = 0; i < n; ++i) var1 += (this->_aResponse[aIdx[i]].val - mean1) * (this->_aResponse[aIdx[i]].val - mean1);
    var1 /= algorithmFPType(n); //impurity is MSE
    DAAL_ASSERT(fabs(mean1 - imp.mean) < 0.001);
    DAAL_ASSERT(fabs(var1 - imp.var) < 0.001);
#endif
}

#ifdef DEBUG_CHECK_IMPURITY
template <typename algorithmFPType, CpuType cpu>
algorithmFPType OrderedRespHelper<algorithmFPType, cpu>::calcResponse(algorithmFPType & res, const IndexType * idx, size_t n) const
{
    const algorithmFPType cDiv = 1. / algorithmFPType(n);
    res                        = this->_aResponse[idx[0]].val * cDiv;
    for (size_t i = 1; i < n; ++i) res += this->_aResponse[idx[i]].val * cDiv;
    return res;
}
#endif

//computes meanPrev as the mean of n-1 elements after removing of element x (based on mean of n elements passed as 'mean' argument)
//instead of impurity, computes the sum of (xi - meanPrev)(xi - meanPrev) for n-1 elements
//(based on the sum of (xi - mean)*(xi - mean) of n elements passed as 'var' argument)
template <typename algorithmFPType, CpuType cpu>
void calcPrevImpurity(algorithmFPType var, algorithmFPType mean, algorithmFPType & varPrev, algorithmFPType & meanPrev, algorithmFPType x, size_t n)
{
    algorithmFPType delta = (x - mean) / algorithmFPType(n - 1);
    varPrev               = var - delta * algorithmFPType(n) * (x - mean);
    meanPrev              = mean - delta;
    if (varPrev < 0) varPrev = 0;
}

template <typename algorithmFPType, CpuType cpu>
void OrderedRespHelper<algorithmFPType, cpu>::simpleSplit(const algorithmFPType * featureVal, const IndexType * aIdx, TSplitData & split) const
{
    split.featureValue = featureVal[0];
    split.left.var     = 0;
    split.left.mean    = this->_aResponse[aIdx[0]].val;
    split.nLeft        = 1;
    split.iStart       = 0;
}

template <typename algorithmFPType, CpuType cpu>
bool OrderedRespHelper<algorithmFPType, cpu>::findBestSplitForFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                      size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                      const ImpurityData & curImpurity, TSplitData & split) const
{
    return split.featureUnordered ? findBestSplitCategoricalFeature(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split) :
                                    findBestSplitOrderedFeature(featureVal, aIdx, n, nMinSplitPart, accuracy, curImpurity, split);
}

template <typename algorithmFPType, CpuType cpu>
void OrderedRespHelper<algorithmFPType, cpu>::finalizeBestSplit(const IndexType * aIdx, size_t n, IndexType iFeature, size_t idxFeatureValueBestSplit,
                                                                TSplitData & bestSplit, IndexType * bestSplitIdx) const
{
    DAAL_ASSERT(bestSplit.nLeft > 0);
    const algorithmFPType divL = algorithmFPType(1.) / algorithmFPType(bestSplit.nLeft);
    bestSplit.left.mean *= divL;
    bestSplit.left.var                                = 0;
    IndexType * bestSplitIdxRight                     = bestSplitIdx + bestSplit.nLeft;
    size_t iLeft                                      = 0;
    size_t iRight                                     = 0;
    int iRowSplitVal                                  = -1;
    const auto aResponse                              = this->_aResponse.get();
    const IndexedFeatures::IndexType * indexedFeature = this->indexedFeatures().data(iFeature);
    for (size_t i = 0; i < n; ++i)
    {
        const auto iSample = aIdx[i];
        const auto idx     = indexedFeature[aResponse[iSample].idx];
        if ((bestSplit.featureUnordered && (idx != idxFeatureValueBestSplit)) || ((!bestSplit.featureUnordered) && (idx > idxFeatureValueBestSplit)))
        {
            DAAL_ASSERT(iRight < n - bestSplit.nLeft);
            bestSplitIdxRight[iRight++] = iSample;
        }
        else
        {
            if (idx == idxFeatureValueBestSplit) iRowSplitVal = aResponse[iSample].idx;
            DAAL_ASSERT(iLeft < bestSplit.nLeft);
            bestSplitIdx[iLeft++]   = iSample;
            const algorithmFPType y = aResponse[iSample].val;
            bestSplit.left.var += (y - bestSplit.left.mean) * (y - bestSplit.left.mean);
        }
    }
    DAAL_ASSERT(iRight == n - bestSplit.nLeft);
    DAAL_ASSERT(iLeft == bestSplit.nLeft);
    bestSplit.left.var *= divL;
    bestSplit.iStart = 0;
    DAAL_ASSERT(iRowSplitVal >= 0);
    bestSplit.featureValue = this->getValue(iFeature, iRowSplitVal);
}

template <typename algorithmFPType, CpuType cpu>
int OrderedRespHelper<algorithmFPType, cpu>::findBestSplitForFeatureSorted(algorithmFPType * buf, IndexType iFeature, const IndexType * aIdx,
                                                                           size_t n, size_t nMinSplitPart, const ImpurityData & curImpurity,
                                                                           TSplitData & split) const
{
    const auto nDiffFeatMax = this->indexedFeatures().numIndices(iFeature);
    _idxFeatureBuf.setValues(nDiffFeatMax, 0);

    //the buffer keeps sums of responses for each of unique feature values
    for (size_t i = 0; i < nDiffFeatMax; ++i) buf[i] = algorithmFPType(0);

    typedef double intermSummFPType;
    //below we calculate only part of the impurity decrease dependent on split itself
    intermSummFPType bestImpDecreasePart =
        split.impurityDecrease < 0 ? -1 : (split.impurityDecrease + curImpurity.mean * curImpurity.mean) * algorithmFPType(n);

    auto nFeatIdx             = _idxFeatureBuf.get(); //number of indexed feature values, array
    intermSummFPType sumTotal = 0;                    //total sum of responses in the set being split
    {
        const IndexedFeatures::IndexType * indexedFeature = this->indexedFeatures().data(iFeature);
        auto aResponse                                    = this->_aResponse.get();
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < n; ++i)
        {
            const IndexType iSample              = aIdx[i];
            const typename super::Response & r   = aResponse[aIdx[i]];
            const IndexedFeatures::IndexType idx = indexedFeature[r.idx];
            ++nFeatIdx[idx];
            buf[idx] += aResponse[iSample].val;
            sumTotal += aResponse[iSample].val;
        }
    }
    size_t nLeft             = 0;
    intermSummFPType sumLeft = 0;
    int idxFeatureBestSplit  = -1; //index of best feature value in the array of sorted feature values
    for (size_t i = 0; i < nDiffFeatMax; ++i)
    {
        if (!nFeatIdx[i]) continue;
        nLeft = (split.featureUnordered ? nFeatIdx[i] : nLeft + nFeatIdx[i]);
        if ((nLeft == n) //last split
            || ((n - nLeft) < nMinSplitPart))
            break;
        sumLeft = (split.featureUnordered ? buf[i] : sumLeft + buf[i]);
        if (nLeft < nMinSplitPart) continue;
        intermSummFPType sumRight = sumTotal - sumLeft;
        //the part of the impurity decrease dependent on split itself
        const intermSummFPType impDecreasePart = sumLeft * sumLeft / intermSummFPType(nLeft) + sumRight * sumRight / intermSummFPType(n - nLeft);
        if (impDecreasePart > bestImpDecreasePart)
        {
            split.left.mean     = algorithmFPType(sumLeft);
            split.nLeft         = nLeft;
            idxFeatureBestSplit = i;
            bestImpDecreasePart = impDecreasePart;
        }
    }
    if (idxFeatureBestSplit >= 0)
    {
        split.impurityDecrease = (bestImpDecreasePart / intermSummFPType(n) - curImpurity.mean * curImpurity.mean);
        //note, left.mean and right.mean are not actually the means but the sums
    }
    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
bool OrderedRespHelper<algorithmFPType, cpu>::findBestSplitOrderedFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                          size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                          const ImpurityData & curImpurity, TSplitData & split) const
{
    algorithmFPType xi = this->_aResponse[aIdx[0]].val;
    ImpurityData left;
    left.var  = 0;
    left.mean = xi;

    ImpurityData right;
    calcPrevImpurity<algorithmFPType, cpu>(curImpurity.var * algorithmFPType(n), curImpurity.mean, right.var, right.mean, xi, n);

#ifdef DEBUG_CHECK_IMPURITY
    checkImpurityInternal(aIdx + 1, n - 1, right);
#endif
    algorithmFPType vBest = split.impurityDecrease < 0 ? daal::services::internal::MaxVal<algorithmFPType>::get() :
                                                         (curImpurity.var - split.impurityDecrease) * algorithmFPType(n);
    IndexType iBest            = -1;
    for (size_t i = 1; i < (n - nMinSplitPart + 1); ++i)
    {
        const bool bSameFeaturePrev(featureVal[i] <= featureVal[i - 1] + accuracy);
        if (!(bSameFeaturePrev || i < nMinSplitPart))
        {
            //can make a split
            //nLeft == i, nRight == n - i
            const algorithmFPType v = left.var + right.var;
            if (v < vBest)
            {
                vBest           = v;
                split.left.var  = left.var;
                split.left.mean = left.mean;
                iBest           = i;
            }
        }

        //update impurity and continue
        xi                    = this->_aResponse[aIdx[i]].val;
        algorithmFPType delta = xi - left.mean;
        left.mean += delta / algorithmFPType(i + 1);
        left.var += delta * (xi - left.mean);
        if (left.var < 0) left.var = 0;
        calcPrevImpurity<algorithmFPType, cpu>(right.var, right.mean, right.var, right.mean, xi, n - i);
#ifdef DEBUG_CHECK_IMPURITY
        checkImpurityInternal(aIdx, i + 1, left);
        checkImpurityInternal(aIdx + i + 1, n - i - 1, right);
#endif
    }
    if (iBest < 0) return false;

    split.impurityDecrease = curImpurity.var - vBest / algorithmFPType(n);
    split.nLeft            = iBest;
    split.left.var /= split.nLeft;
    split.iStart       = 0;
    split.featureValue = featureVal[iBest - 1];
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool OrderedRespHelper<algorithmFPType, cpu>::findBestSplitCategoricalFeature(const algorithmFPType * featureVal, const IndexType * aIdx, size_t n,
                                                                              size_t nMinSplitPart, const algorithmFPType accuracy,
                                                                              const ImpurityData & curImpurity, TSplitData & split) const
{
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    ImpurityData left;
    ImpurityData right;
    bool bFound = false;
    algorithmFPType vBest;
    size_t nDiffFeatureValues = 0;
    for (size_t i = 0; i < n - nMinSplitPart;)
    {
        ++nDiffFeatureValues;
        size_t count                = 1;
        const algorithmFPType first = featureVal[i];
        const size_t iStart         = i;
        for (++i; (i < n) && (featureVal[i] == first); ++count, ++i)
            ;
        if ((count < nMinSplitPart) || ((n - count) < nMinSplitPart)) continue;

        if ((i == n) && (nDiffFeatureValues == 2) && bFound) break; //only 2 feature values, one possible split, already found

        calcImpurity(aIdx + iStart, count, left);
        subtractImpurity<algorithmFPType, cpu>(curImpurity.var, curImpurity.mean, left.var, left.mean, count, right.var, right.mean, n - count);
#ifdef DEBUG_CHECK_IMPURITY
        if (iStart == 0) checkImpurityInternal(aIdx + count, n - count, right);
#endif
        const algorithmFPType v = algorithmFPType(count) * left.var + algorithmFPType(n - count) * right.var;
        if (!bFound || v < vBest)
        {
            vBest              = v;
            split.left.var     = left.var;
            split.left.mean    = left.mean;
            split.nLeft        = count;
            split.iStart       = iStart;
            split.featureValue = first;
            bFound             = true;
        }
    }
    if (bFound)
    {
        algorithmFPType impurityDecrease = curImpurity.var - vBest / algorithmFPType(n);
        if (split.impurityDecrease < 0 || split.impurityDecrease < impurityDecrease)
        {
            split.impurityDecrease = impurityDecrease;
            return true;
        }
    }
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////
// TreeThreadCtx class for regression
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class TreeThreadCtx : public TreeThreadCtxBase<algorithmFPType, cpu>
{
public:
    typedef TreeThreadCtxBase<algorithmFPType, cpu> super;
    TreeThreadCtx(algorithmFPType * _varImp = nullptr) : super(_varImp) {}
    bool init(const decision_forest::training::Parameter & par, const NumericTable * x, size_t /*dummy*/)
    {
        DAAL_CHECK_STATUS_VAR(super::init(par, x));
        using namespace decision_forest::training;
        if (par.resultsToCompute & (computeOutOfBagError | computeOutOfBagErrorPerObservation))
        {
            size_t sz    = sizeof(RegErr<algorithmFPType, cpu>) * x->getNumberOfRows();
            this->oobBuf = service_calloc<byte, cpu>(sz);
            DAAL_CHECK_STATUS_VAR(this->oobBuf);
        }
        return true;
    }

    void reduceTo(decision_forest::training::VariableImportanceMode mode, TreeThreadCtx & other, size_t nVars, size_t nSamples) const
    {
        super::reduceTo(mode, other, nVars, nSamples);
        if (this->oobBuf)
        {
            RegErr<algorithmFPType, cpu> * dst       = (RegErr<algorithmFPType, cpu> *)other.oobBuf;
            const RegErr<algorithmFPType, cpu> * src = (const RegErr<algorithmFPType, cpu> *)this->oobBuf;
            for (size_t i = 0; i < nSamples; ++i) dst[i].add(src[i]);
        }
    }

    Status finalizeOOBError(const NumericTable * resp, algorithmFPType * res, algorithmFPType * resPerObs) const
    {
        DAAL_ASSERT(this->oobBuf);
        const size_t nSamples = resp->getNumberOfRows();
        ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable *>(resp), 0, nSamples);
        DAAL_CHECK_BLOCK_STATUS(y);
        const algorithmFPType * py         = y.get();
        size_t nPredicted                  = 0.;
        algorithmFPType _res               = 0;
        RegErr<algorithmFPType, cpu> * ptr = (RegErr<algorithmFPType, cpu> *)this->oobBuf;
        for (size_t i = 0; i < nSamples; ++i)
        {
            if (ptr[i].count)
            {
                ptr[i].value /= algorithmFPType(ptr[i].count);
                const algorithmFPType oobForObs = (ptr[i].value - py[i]) * (ptr[i].value - py[i]);
                if (resPerObs) resPerObs[i] = oobForObs;
                _res += oobForObs;
                ++nPredicted;
            }
            else if (resPerObs)
                resPerObs[i] = algorithmFPType(-1); //was not in OOB set of any tree and hence not predicted
        }
        if (res) *res = _res / algorithmFPType(nPredicted);
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
    TrainBatchTask(HostAppIface * hostApp, const NumericTable * x, const NumericTable * y, const decision_forest::training::Parameter & par,
                   const dtrees::internal::FeatureTypes & featTypes, const dtrees::internal::IndexedFeatures * indexedFeatures,
                   typename super::ThreadCtxType & ctx, size_t dummy)
        : super(hostApp, x, y, par, featTypes, indexedFeatures, ctx, dummy)
    {
        if (!this->_nFeaturesPerNode)
        {
            size_t nF                                     = x->getNumberOfColumns() / 3;
            const_cast<size_t &>(this->_nFeaturesPerNode) = (nF < 1 ? 1 : nF);
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, decision_forest::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainBatchKernel<algorithmFPType, method, cpu>::compute(HostAppIface * pHostApp, const NumericTable * x,
                                                                                   const NumericTable * y, decision_forest::regression::Model & m,
                                                                                   Result & res, const Parameter & par)
{
    ResultData rd(par, res.get(variableImportance).get(), res.get(outOfBagError).get(), res.get(outOfBagErrorPerObservation).get());
    services::Status s = computeImpl<algorithmFPType, cpu, daal::algorithms::decision_forest::regression::internal::ModelImpl,
                                     TrainBatchTask<algorithmFPType, method, cpu> >(
        pHostApp, x, y, *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m), rd, par, 0);
    if (s.ok()) res.impl()->setEngine(rd.updatedEngine);
    return s;
}

} /* namespace internal */
} /* namespace training */
} /* namespace regression */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

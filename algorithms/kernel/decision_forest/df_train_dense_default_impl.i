/* file: df_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for decision forest train algorithms
//  (defaultDense) method.
//--
*/

#ifndef __DF_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DF_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_memory.h"
#include "service_rng.h"
#include "service_numeric_table.h"
#include "service_data_utils.h"
#include "service_sort.h"
#include "service_math.h"
#include "df_model_impl.h"

//#define DEBUG_CHECK_IMPURITY

typedef int IndexType;

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// Service function, generates permutation of indices from 0 to n - 1
//////////////////////////////////////////////////////////////////////////////////////////
template <CpuType cpu>
void generatePermutation(BaseRNGs<cpu>& brng, size_t n, IndexType* dst)
{
    RNGs<IndexType, cpu> rng;
    IndexType idx[2];
    for(size_t i = 0; i < n; ++i)
    {
        rng.uniform(2, idx, brng, 0, n);
        daal::services::internal::swap<cpu, IndexType>(dst[idx[0]], dst[idx[1]]);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, keeps response-dependent split data
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TImpurityData>
struct SplitData
{
    TImpurityData left;
    TImpurityData right;
    algorithmFPType featureValue;
    size_t nLeft;
    size_t iStart;
    bool featureUnordered;

    algorithmFPType totalImpurity(size_t n) const
    {
        return algorithmFPType(nLeft)*left.value() + algorithmFPType(n - nLeft)*right.value();
    }
    void copyTo(SplitData& o)
    {
        o.featureValue = featureValue;
        o.nLeft = nLeft;
        o.iStart = iStart;
        o.left = left;
        o.right = right;
        o.featureUnordered = featureUnordered;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// FeatureHelper, provides optimal access to the feature types
//////////////////////////////////////////////////////////////////////////////////////////
template <CpuType cpu>
class FeatureHelper
{
public:
    FeatureHelper() : _bAllUnordered(false){}
    bool init(const NumericTable* data)
    {
        size_t count = 0;
        _firstUnordered = -1;
        _lastUnordered = -1;
        const size_t p = data->getNumberOfColumns();
        for(size_t i = 0; i < p; ++i)
        {
            if(data->getFeatureType(i) != data_management::data_feature_utils::DAAL_CATEGORICAL)
                continue;
            if(_firstUnordered < 0)
                _firstUnordered = i;
            _lastUnordered = i;
            ++count;
        }
        _bAllUnordered = (p == count);
        if(_bAllUnordered)
        {
            _aCatFeatures.reset(0);
            return true;
        }
        if(!count)
            return true;
        _aCatFeatures.reset(_lastUnordered - _firstUnordered + 1);
        if(!_aCatFeatures.get())
            return false;
        for(size_t i = _firstUnordered; i < _lastUnordered + 1; ++i)
        {
            _aCatFeatures[i - _firstUnordered] = (data->getFeatureType(i) == data_management::data_feature_utils::DAAL_CATEGORICAL);
        }
        return true;
    }

    bool isUnordered(size_t iFeature) const
    {
        if(_bAllUnordered)
            return true;
        if(!_aCatFeatures.size() || iFeature < _firstUnordered)
            return false;
        const size_t i = iFeature - _firstUnordered;
        if(i < _aCatFeatures.size())
            return _aCatFeatures[i];
        DAAL_ASSERT(iFeature > _lastUnordered);
        return false;
    }

private:
    TArray<bool, cpu> _aCatFeatures;
    bool _bAllUnordered;
    int _firstUnordered;
    int _lastUnordered;
};

//////////////////////////////////////////////////////////////////////////////////////////
// DataHelper. Base class for response-specific services classes.
// Keeps indices of the bootstrap samples and provides optimal access to columns in case
// of homogenious numeric table
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TResponse, CpuType cpu>
class DataHelper
{
protected:
    struct Response
    {
        TResponse val;
        IndexType idx;
    };

public:
    DataHelper() : _data(nullptr), _dataDirect(nullptr), _nCols(0){}
    const NumericTable* data() const { return _data; }
    TResponse response(size_t i) const { return _aResponse[i].val; }
    bool reset(size_t n)
    {
        _aResponse.reset(n);
        return _aResponse.get() != nullptr;
    }
    void init(const NumericTable* data, const NumericTable* resp, const IndexType* aSample)
    {
        if(!_aResponse.size())
            return;
        _data = const_cast<NumericTable*>(data);
        _nCols = data->getNumberOfColumns();
        const HomogenNumericTable<algorithmFPType>* hmg = dynamic_cast<const HomogenNumericTable<algorithmFPType>*>(data);
        _dataDirect = (hmg ? hmg->getArray() : nullptr);
        const IndexType firstRow = aSample[0];
        const IndexType lastRow = aSample[_aResponse.size() - 1];
        ReadRows<algorithmFPType, cpu> bd(const_cast<NumericTable*>(resp), firstRow, lastRow - firstRow + 1);
        for(size_t i = 0; i < _aResponse.size(); ++i)
        {
            _aResponse[i].idx = aSample[i];
            _aResponse[i].val = TResponse(bd.get()[aSample[i] - firstRow]);
        }
    }

    void getColumnValues(size_t iCol, const IndexType* aIdx, size_t n, algorithmFPType* aVal)
    {
        if(_dataDirect)
        {
            for(size_t i = 0; i < n; ++i)
            {
                auto iRow = getObsIdx(aIdx[i]);
                aVal[i] = _dataDirect[iRow*_nCols + iCol];
            }
        }
        else
        {
            data_management::BlockDescriptor<algorithmFPType> bd;
            for(size_t i = 0; i < n; ++i)
            {
                _data->getBlockOfColumnValues(iCol, getObsIdx(aIdx[i]), 1, readOnly, bd);
                aVal[i] = *bd.getBlockPtr();
                _data->releaseBlockOfColumnValues(bd);
            }
        }
    }

    size_t getNumOOBIndices() const
    {
        if(!_aResponse.size())
            return 0;

        size_t count = _aResponse[0].idx;
        size_t prev = count;
        for(size_t i = 1; i < _aResponse.size(); prev = _aResponse[i++].idx)
            count += (_aResponse[i].idx > (prev + 1) ? (_aResponse[i].idx - prev - 1) : 0);
        const size_t nRows = _data->getNumberOfRows();
        count += (nRows > (prev + 1) ? (nRows - prev - 1) : 0);
        return count;
    }

    void getOOBIndices(IndexType* dst) const
    {
        if(!_aResponse.size())
            return;

        const IndexType* savedDst = dst;
        size_t idx = _aResponse[0].idx;
        size_t iDst = 0;
        for(; iDst < idx; dst[iDst] = iDst, ++iDst);

        for(size_t iResp = 1; iResp < _aResponse.size(); idx = _aResponse[iResp].idx, ++iResp)
        {
            for(++idx; idx < _aResponse[iResp].idx; ++idx, ++iDst)
                dst[iDst] = idx;
        }

        const size_t nRows = _data->getNumberOfRows();
        for(++idx; idx < nRows; ++idx, ++iDst)
            dst[iDst] = idx;
        DAAL_ASSERT(iDst == getNumOOBIndices());
    }

protected:
    IndexType getObsIdx(size_t i) const { DAAL_ASSERT(i < _aResponse.size());  return _aResponse[i].idx; }

protected:
    TArray<Response, cpu> _aResponse;
    const algorithmFPType* _dataDirect;
    NumericTable* _data;
    size_t _nCols;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains numeric tables to be calculated as result
//////////////////////////////////////////////////////////////////////////////////////////
struct ResultData
{
public:
    ResultData(const Parameter& par, NumericTable* _varImp, NumericTable* _oobError) :
        oobError(nullptr), varImp(nullptr)
    {
        if(par.varImportance != decision_forest::training::none)
            varImp = _varImp;
        if(par.resultsToCompute & decision_forest::training::computeOutOfBagError)
            oobError = _oobError;
    }
    NumericTable* varImp; //if needed then allocated outside kernel
    NumericTable* oobError; //if needed then allocated outside kernel
    NumericTablePtr oobIndicesNum; //if needed then allocated in kernel
    NumericTablePtr oobIndices; //if needed then allocated in kernel
};

template <typename algorithmFPType, CpuType cpu>
bool isPositive(algorithmFPType val)
{
    return (val > algorithmFPType(10)*daal::data_feature_utils::internal::EpsilonVal<algorithmFPType, cpu>::get());
}

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains workset required for tree calculation in one thread
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class TreeThreadCtxBase
{
public:
    TreeThreadCtxBase(algorithmFPType* _varImp = nullptr) : varImp(_varImp), varImpVariance(nullptr), nTrees(0), oobBuf(nullptr){}
    ~TreeThreadCtxBase()
    {
        if(varImpVariance)
            service_free<algorithmFPType, cpu>(varImpVariance);
        if(oobBuf)
            service_free<byte, cpu>(oobBuf);
    }
    void finalizeVarImp(size_t nVars, training::VariableImportanceMode mode);

protected:
    bool init(const Parameter& par, const NumericTable* x)
    {
        if(par.varImportance == training::MDA_Scaled)
        {
            varImpVariance = service_calloc<algorithmFPType, cpu>(x->getNumberOfColumns());
            if(!varImpVariance)
                return false;
        }
        return true;
    }

    void reduceTo(TreeThreadCtxBase& other, size_t nVars, size_t nSamples) const
    {
        // Reduces tls variable importance results
        if(varImp)
        {
            for(size_t i = 0; i < nVars; ++i)
                other.varImp[i] += varImp[i];
        }
        if(varImpVariance)
        {
            for(size_t i = 0; i < nVars; ++i)
                other.varImpVariance[i] += varImpVariance[i];
        }
        other.nTrees += nTrees;
    }

public:
    algorithmFPType* varImp;
    algorithmFPType* varImpVariance;
    size_t nTrees;
    byte* oobBuf;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Finalizes calculation of variable importance results
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
void TreeThreadCtxBase<algorithmFPType, cpu>::finalizeVarImp(size_t nVars, training::VariableImportanceMode mode)
{
    const algorithmFPType div = 1. / algorithmFPType(nTrees);
    if(mode == training::MDA_Scaled)
    {
        //average over all trees and scale by its variance
        for(size_t i = 0; i < nVars; ++i)
        {
            varImp[i] *= div;
            varImpVariance[i] = varImpVariance[i] * div - varImp[i] * varImp[i];
            if(isPositive<algorithmFPType, cpu>(varImpVariance[i]))
                varImp[i] /= daal::internal::Math<algorithmFPType, cpu>::sSqrt(varImpVariance[i] * div);
        }
    }
    else
    {
        //average over all trees
        for(size_t i = 0; i < nVars; ++i)
            varImp[i] *= div;
    }

#if ENABLE_VAR_IMP_NORMALIZATION
    algorithmFPType sum = 0;
    for(size_t i = 0; i < nVars; ++i)
        sum += varImp[i];
    //normalize by division to the sum of all values
    if(!isPositive<algorithmFPType, cpu>(sum))
    {
        algorithmFPType maxVal = 0;
        for(size_t i = 0; i < nVars; ++i)
        {
            const algorithmFPType val = daal::internal::Math<algorithmFPType, cpu>::sFabs(varImp[i]);
            maxVal = daal::internal::Math<algorithmFPType, cpu>::sMax(maxVal, val);
        }
        if(!isPositive<algorithmFPType, cpu>(maxVal))
            return;
        const algorithmFPType div = 1. / maxVal;
        for(size_t i = 0; i < nVars; varImp[i++] *= div);
    }
    else
    {
        sum = 1. / sum;
        for(size_t i = 0; i < nVars; varImp[i++] *= sum);
    }
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////
// compute() implementation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu, typename ModelType, typename TaskType>
services::Status computeImpl(const NumericTable *x, const NumericTable *y, ModelType& md, ResultData& res,
    const Parameter& par, size_t nClasses)
{
    DAAL_CHECK(md.reserve(par.nTrees), ErrorMemoryAllocationFailed);
    FeatureHelper<cpu> featHelper;
    DAAL_CHECK(featHelper.init(x), ErrorMemoryAllocationFailed);

    const auto nFeatures = x->getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> varImpBD(res.varImp, 0, 1);
    if(res.varImp)
        DAAL_CHECK_BLOCK_STATUS(varImpBD);

    //set of auxiliary data used by single thread or a target for reduction by multiple threads
    typename TaskType::ThreadCtxType mainCtx(varImpBD.get());
    DAAL_CHECK(mainCtx.init(par, x, nClasses), ErrorMemoryAllocationFailed);

    if(mainCtx.varImp)
        //initialize its data
        daal::services::internal::service_memset<algorithmFPType, cpu>(mainCtx.varImp, 0, nFeatures);

    //use tls in case of multiple threads
    const bool bThreaded = (threader_get_max_threads_number() > 1) && (par.nTrees > 1);
    typedef typename TaskType::ThreadCtxType Ctx;
    daal::tls<Ctx*> tlsData([=, &par]()->Ctx*
    {
        const size_t szVarImp = (par.varImportance == decision_forest::training::none ? 0 : nFeatures*sizeof(algorithmFPType));
        const size_t sz = sizeof(Ctx) + szVarImp;
        byte* ptr = service_scalable_calloc<byte, cpu>(sz);
        Ctx* ctx = new (ptr)Ctx();
        if(ctx)
        {
            if(szVarImp)
                ctx->varImp = (algorithmFPType*)(ptr + sizeof(Ctx));
            if(!ctx->init(par, x, nClasses))
            {
                //allocation of extra data hasn't succeeded
                ctx->~Ctx();
                service_scalable_free<byte, cpu>(ptr);
                return nullptr;
            }
        }
        return ctx;
    });

    bool bMemoryAllocationFailed = false;
    daal::threader_for(par.nTrees, par.nTrees, [&](size_t i)
    {
        //in case of single thread no need to allocate
        typename TaskType::ThreadCtxType* ctx = (bThreaded ? tlsData.local() : &mainCtx);
        if(!ctx)
        {
            bMemoryAllocationFailed = true;
            return;
        }
        TaskType task(size_t(par.seed)*(i + 1), x, y, par, featHelper, *ctx, nClasses);
        decision_forest::internal::Tree* pTree = task.run();
        if(pTree)
            md.add(pTree);
    });
    if(bThreaded)
        tlsData.reduce([=, &mainCtx](typename TaskType::ThreadCtxType* ptr)-> void
    {
        if(!ptr)
            return;
        ptr->reduceTo(mainCtx, nFeatures, x->getNumberOfRows());
        Ctx* ctx = (Ctx*)ptr;
        ctx->~Ctx();
        service_scalable_free<byte, cpu>((byte*)ptr);
    });
    DAAL_CHECK(md.size() == par.nTrees, ErrorMemoryAllocationFailed);
    DAAL_CHECK(!bMemoryAllocationFailed, ErrorMemoryAllocationFailed);

    //finalize results computation
    //variable importance
    if(varImpBD.get())
        mainCtx.finalizeVarImp(nFeatures, par.varImportance);

    services::Status s;
    //OOB error
    if(par.resultsToCompute & computeOutOfBagError)
    {
        WriteOnlyRows<algorithmFPType, cpu> oobErr(res.oobError, 0, 1);
        s = mainCtx.finalizeOOBError(y, *oobErr.get());
    }
    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Base task class. Implements general pipeline of tree building
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename DataHelper, CpuType cpu>
class TrainBatchTaskBase
{
public:
    decision_forest::internal::Tree* run();

protected:
    typedef TArray<algorithmFPType, cpu> algorithmFPTypeArray;
    typedef TArray<IndexType, cpu> IndexTypeArray;
    typedef TreeThreadCtxBase<algorithmFPType, cpu> ThreadCtxType;
    TrainBatchTaskBase(size_t seed, const NumericTable *x, const NumericTable *y, const Parameter& par,
        const FeatureHelper<cpu>& featHelper, ThreadCtxType& threadCtx, size_t nClasses) :
        _data(x), _resp(y), _par(par), _brng(seed),
        _nSamples(par.observationsPerTreeFraction*x->getNumberOfRows()),
        _nFeaturesPerNode(par.featuresPerNode),
        _helper(nClasses),
        _impurityThreshold(_par.impurityThreshold),
        _nFeatureBufs(1), //for sequential processing
        _featHelper(featHelper),
        _threadCtx(threadCtx),
        _accuracy(daal::data_feature_utils::internal::EpsilonVal<algorithmFPType, cpu>::get())
    {
        if(_impurityThreshold < _accuracy)
            _impurityThreshold = _accuracy;
    }

    size_t nFeatures() const { return _data->getNumberOfColumns(); }
    typename DataHelper::NodeType::Base* build(size_t iStart, size_t n, size_t level,
        typename DataHelper::ImpurityData& curImpurity, bool& bUnorderedFeaturesUsed);
    algorithmFPType* featureBuf(size_t iBuf) const { DAAL_ASSERT(iBuf < _nFeatureBufs); return _aFeatureBuf[iBuf].get(); }
    IndexType* featureIndexBuf(size_t iBuf) const { DAAL_ASSERT(iBuf < _nFeatureBufs); return _aFeatureIndexBuf[iBuf].get(); }
    bool terminateCriteria(size_t nSamples, size_t level, algorithmFPType impurityValue) const
    {
        return (nSamples < 2 * _par.minObservationsInLeafNode ||
            impurityValue <= _impurityThreshold ||
            ((_par.maxTreeDepth > 0) && (level >= _par.maxTreeDepth)));
    }
    typename DataHelper::NodeType::Split* makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered,
        typename DataHelper::NodeType::Base* left, typename DataHelper::NodeType::Base* right, algorithmFPType imp) const;

    bool findBestSplit(size_t iStart, size_t n, const typename DataHelper::ImpurityData& curImpurity,
        IndexType& iBestFeature, typename DataHelper::TSplitData& split);
    bool findBestSplitSerial(size_t iStart, size_t n, const typename DataHelper::ImpurityData& curImpurity,
        IndexType& iBestFeature, typename DataHelper::TSplitData& split);
    bool findBestSplitThreaded(size_t iStart, size_t n, const typename DataHelper::ImpurityData& curImpurity,
        IndexType& iBestFeature, typename DataHelper::TSplitData& split);
    bool simpleSplit(size_t iStart, const typename DataHelper::ImpurityData& curImpurity,
        IndexType& iFeatureBest, typename DataHelper::TSplitData& split);
    void addImpurityDecrease(IndexType iFeature, size_t n, const typename DataHelper::ImpurityData& curImpurity,
        const typename DataHelper::TSplitData& split);

    void featureValuesToBuf(size_t iFeature, algorithmFPType* featureVal, IndexType* aIdx, size_t n)
    {
        _helper.getColumnValues(iFeature, aIdx, n, featureVal);
        daal::algorithms::internal::qSort<algorithmFPType, int, cpu>(n, featureVal, aIdx);
    }

    //find features to check in the current split node
    void chooseFeatures()
    {
        const size_t n = nFeatures();
        if(n == _nFeaturesPerNode)
        {
            for(size_t i = 0; i < n; _aFeatureIdx[i] = i, ++i);
            generatePermutation<cpu>(_brng, n, _aFeatureIdx.get());
        }
        else
        {
            RNGs<IndexType, cpu> rng;
            rng.uniformWithoutReplacement(_nFeaturesPerNode, _aFeatureIdx.get(), _brng, 0, n);
        }
    }

    bool computeResults(const decision_forest::internal::Tree& t);

    algorithmFPType computeOOBError(const decision_forest::internal::Tree& t, size_t n, const IndexType* aInd);

    algorithmFPType computeOOBErrorPerm(const decision_forest::internal::Tree& t,
        size_t n, const IndexType* aInd, const IndexType* aPerm, size_t iPermutedFeature);

protected:
    BaseRNGs<cpu> _brng;
    const NumericTable *_data;
    const NumericTable *_resp;
    const Parameter& _par;
    const size_t _nSamples;
    const size_t _nFeaturesPerNode;
    const size_t _nFeatureBufs; //number of buffers to get feature values (to process features independently in parallel)
    mutable TArray<IndexType, cpu> _aSample;
    mutable TArray<algorithmFPTypeArray, cpu> _aFeatureBuf;
    mutable TArray<IndexTypeArray, cpu> _aFeatureIndexBuf;

    TArray<IndexType, cpu> _aFeatureIdx; //indices of features to be used for the soplit at the current level
    DataHelper _helper;
    const FeatureHelper<cpu>& _featHelper;
    algorithmFPType _accuracy;
    algorithmFPType _impurityThreshold;
    ThreadCtxType& _threadCtx;
};

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
decision_forest::internal::Tree* TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::run()
{
    _aSample.reset(_nSamples);
    _aFeatureBuf.reset(_nFeatureBufs);
    _aFeatureIndexBuf.reset(_nFeatureBufs);
    _aFeatureIdx.reset(_nFeaturesPerNode);

    if(!_aSample.get() || !_helper.reset(_nSamples) || !_aFeatureBuf.get() || !_aFeatureIndexBuf.get() || !_aFeatureIdx.get())
        return nullptr;

    //allocate temporary bufs
    for(size_t i = 0; i < _nFeatureBufs; ++i)
    {
        _aFeatureBuf[i].reset(_nSamples);
        if(!_aFeatureBuf[i].get())
            return nullptr;
        _aFeatureIndexBuf[i].reset(_nSamples);
        if(!_aFeatureIndexBuf[i].get())
            return nullptr;
    }

    RNGs<int, cpu> rng;
    rng.uniform(_nSamples, _aSample.get(), _brng, 0, _data->getNumberOfRows());
    daal::algorithms::internal::qSort<int, cpu>(_nSamples, _aSample.get());

    //init responses buffer, keep _aSample values in it
    _helper.init(_data, _resp, _aSample.get());

    //use _aSample as an array of response indices stored by helper from now on
    for(size_t i = 0; i < _aSample.size(); _aSample[i] = i, ++i);

    typename DataHelper::ImpurityData initialImpurity;
    _helper.calcImpurity(_aSample.get(), _nSamples, initialImpurity);
    bool bUnorderedFeaturesUsed = false;
    typename DataHelper::NodeType::Base* nd = build(0, _nSamples, 0, initialImpurity, bUnorderedFeaturesUsed);
    if(!nd)
        return nullptr;
    decision_forest::internal::Tree* pTree = new decision_forest::internal::TreeImpl<typename DataHelper::NodeType>(nd, bUnorderedFeaturesUsed);
    _threadCtx.nTrees++;
    if((_par.resultsToCompute & computeOutOfBagError) || (_par.varImportance > MDI))
    {
        if(!computeResults(*pTree))
        {
            delete pTree;
            pTree = nullptr;
        }
    }
    return pTree;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
typename DataHelper::NodeType::Split* TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::makeSplit(size_t iFeature,
    algorithmFPType featureValue, bool bUnordered,
    typename DataHelper::NodeType::Base* left, typename DataHelper::NodeType::Base* right, algorithmFPType imp) const
{
    typename DataHelper::NodeType::Split* pNode = DataHelper::TreeType::Allocator::allocSplit();
    pNode->set(iFeature, featureValue, bUnordered);
    pNode->kid[0] = left;
    pNode->kid[1] = right;
    pNode->impurity = imp;
    return pNode;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
typename DataHelper::NodeType::Base* TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::build(size_t iStart, size_t n,
    size_t level, typename DataHelper::ImpurityData& curImpurity, bool& bUnorderedFeaturesUsed)
{
    if(terminateCriteria(n, level, curImpurity.value()))
        return _helper.makeLeaf(_aSample.get() + iStart, n, curImpurity);

    typename DataHelper::TSplitData split;
    IndexType iFeature;
    if(findBestSplit(iStart, n, curImpurity, iFeature, split))
    {
        typename DataHelper::NodeType::Base* left = build(iStart, split.nLeft, level + 1, split.left, bUnorderedFeaturesUsed);
        typename DataHelper::NodeType::Base* right = build(iStart + split.nLeft, n - split.nLeft, level + 1, split.right, bUnorderedFeaturesUsed);
        typename DataHelper::NodeType::Base* res = nullptr;
        if(!left || !right || !(res = makeSplit(iFeature, split.featureValue, split.featureUnordered, left, right, curImpurity.var)))
        {
            if(left)
                decision_forest::internal::deleteNode<typename DataHelper::NodeType, typename DataHelper::TreeType::Allocator>(left);
            if(right)
                decision_forest::internal::deleteNode<typename DataHelper::NodeType, typename DataHelper::TreeType::Allocator>(right);
            return nullptr;
        }
        bUnorderedFeaturesUsed |= split.featureUnordered;
        DAAL_ASSERT(n == res->count);
        DAAL_ASSERT(split.nLeft == left->count);
        DAAL_ASSERT(n - split.nLeft == right->count);
        if(_par.varImportance == training::MDI)
            addImpurityDecrease(iFeature, n, curImpurity, split);
        return res;
    }
    return _helper.makeLeaf(_aSample.get() + iStart, n, curImpurity);
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::simpleSplit(size_t iStart,
    const typename DataHelper::ImpurityData& curImpurity, IndexType& iFeatureBest, typename DataHelper::TSplitData& split)
{
    RNGs<IndexType, cpu> rng;
    algorithmFPType featBuf[2];
    IndexType* aIdx = _aSample.get() + iStart;
    for(size_t i = 0; i < _nFeaturesPerNode; ++i)
    {
        IndexType iFeature;
        rng.uniform(1, &iFeature, _brng, 0, _data->getNumberOfColumns());
        featureValuesToBuf(iFeature, featBuf, aIdx, 2);
        if(featBuf[1] - featBuf[0] < _accuracy) //all values of the feature are the same
            continue;
        _helper.simpleSplit(featBuf, aIdx, split);
        iFeatureBest = iFeature;
        return true;
    }
    return false;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::findBestSplit(size_t iStart, size_t n,
    const typename DataHelper::ImpurityData& curImpurity, IndexType& iFeatureBest, typename DataHelper::TSplitData& split)
{
    if(n == 2)
    {
        DAAL_ASSERT(_par.minObservationsInLeafNode == 1);
#ifdef DEBUG_CHECK_IMPURITY
        _helper.checkImpurity(_aSample.get() + iStart, n, curImpurity);
#endif
        return simpleSplit(iStart, curImpurity, iFeatureBest, split);
    }
    if(_nFeatureBufs == 1)
        return findBestSplitSerial(iStart, n, curImpurity, iFeatureBest, split);
    return findBestSplitThreaded(iStart, n, curImpurity, iFeatureBest, split);
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::findBestSplitSerial(size_t iStart, size_t n,
    const typename DataHelper::ImpurityData& curImpurity, IndexType& iFeatureBest, typename DataHelper::TSplitData& bestSplit)
{
    chooseFeatures();

    algorithmFPType* featBuf = featureBuf(0) + iStart; //single thread
    IndexType* bestSplitIdx = featureIndexBuf(0) + iStart;
    IndexType* aIdx = _aSample.get() + iStart;
    IndexType iBestSplit = -1;

    //find best split for each feature
    {
        typename DataHelper::TSplitData split;
        for(size_t i = 0; i < _nFeaturesPerNode; ++i)
        {
            const auto iFeature = _aFeatureIdx[i];
            featureValuesToBuf(iFeature, featBuf, aIdx, n);
            if(featBuf[n - 1] - featBuf[0] < _accuracy) //all values of the feature are the same
                continue;
            split.featureUnordered = _featHelper.isUnordered(iFeature);
#ifdef DEBUG_CHECK_IMPURITY
            _helper.checkImpurity(aIdx, n, curImpurity);
#endif
            if(!_helper.findBestSplitForFeature(featBuf, aIdx, n, _par.minObservationsInLeafNode, _accuracy, curImpurity, split))
                continue;
            if(iBestSplit < 0 || split.totalImpurity(n) < bestSplit.totalImpurity(n))
            {
                iBestSplit = i;
                split.copyTo(bestSplit);
                DAAL_ASSERT(bestSplit.iStart < n);
                DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= n);
                if(i + 1 < _nFeaturesPerNode || split.featureUnordered)
                    daal::services::daal_memcpy_s(bestSplitIdx, sizeof(IndexType)*n, aIdx, sizeof(IndexType)*n);
            }
        }
    }
    if(iBestSplit < 0)
        return false;

    if(bestSplit.featureUnordered)
    {
        if(bestSplit.iStart)
        {
            DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= n);
            daal::services::daal_memcpy_s(aIdx, sizeof(IndexType)*bestSplit.nLeft, bestSplitIdx + bestSplit.iStart, sizeof(IndexType)*bestSplit.nLeft);
            aIdx += bestSplit.nLeft;
            daal::services::daal_memcpy_s(aIdx, sizeof(IndexType)*bestSplit.iStart, bestSplitIdx, sizeof(IndexType)*bestSplit.iStart);
            aIdx += bestSplit.iStart;
            bestSplitIdx += bestSplit.iStart + bestSplit.nLeft;
            n -= bestSplit.iStart + bestSplit.nLeft;
            if(n)
                daal::services::daal_memcpy_s(aIdx, sizeof(IndexType)*n, bestSplitIdx, sizeof(IndexType)*n);
        }
        else
        {
            daal::services::daal_memcpy_s(aIdx, sizeof(IndexType)*n, bestSplitIdx, sizeof(IndexType)*n);
        }
    }
    else if(iBestSplit + 1 < _nFeaturesPerNode)
        daal::services::daal_memcpy_s(aIdx, sizeof(IndexType)*n, bestSplitIdx, sizeof(IndexType)*n);
    iFeatureBest = _aFeatureIdx[iBestSplit];
    return true;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::findBestSplitThreaded(size_t iStart, size_t n,
    const typename DataHelper::ImpurityData& curImpurity, IndexType& iFeatureBest, typename DataHelper::TSplitData& split)
{
    chooseFeatures();
    TArray<typename DataHelper::TSplitData, cpu> aFeatureSplit(_nFeaturesPerNode);

    //TODO, see previous implementation in svn
    return false;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::addImpurityDecrease(IndexType iFeature, size_t n,
    const typename DataHelper::ImpurityData& curImpurity,
    const typename DataHelper::TSplitData& split)
{
    DAAL_ASSERT(_threadCtx.varImp);
    algorithmFPType val = curImpurity.value() -
        (algorithmFPType(split.nLeft)*split.left.value() + algorithmFPType(n - split.nLeft)*split.right.value()) / algorithmFPType(n);
    if((val < daal::data_feature_utils::internal::EpsilonVal<algorithmFPType, cpu>::get()) &&
        (val > -daal::data_feature_utils::internal::EpsilonVal<algorithmFPType, cpu>::get()))
        val = 0;
    _threadCtx.varImp[iFeature] += val;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::computeResults(const decision_forest::internal::Tree& t)
{
    const size_t nOOB = _helper.getNumOOBIndices();
    TArray<IndexType, cpu> oobIndices(nOOB);
    if(!oobIndices.get())
        return false;
    _helper.getOOBIndices(oobIndices.get());
    const bool bMDA(_par.varImportance == training::MDA_Raw || _par.varImportance == training::MDA_Scaled);
    if(_par.resultsToCompute & computeOutOfBagError || bMDA)
    {
        const algorithmFPType oobError = computeOOBError(t, nOOB, oobIndices.get());
        if(bMDA)
        {
            TArray<IndexType, cpu> permutation(nOOB);
            if(!permutation.get())
                return false;
            for(size_t i = 0; i < nOOB; permutation[i] = i, ++i);

            for(size_t i = 0, n = nFeatures(); i < n; ++i)
            {
                generatePermutation(_brng, nOOB, permutation.get());
                const algorithmFPType permOOBError = computeOOBErrorPerm(t, nOOB, oobIndices.get(), permutation.get(), i);
                const algorithmFPType diff = (permOOBError - oobError);
                _threadCtx.varImp[i] += diff;
                if(_threadCtx.varImpVariance)
                    _threadCtx.varImpVariance[i] += diff*diff;
            }
        }
    }
    return true;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
algorithmFPType predictionError(const DataHelper& h, const decision_forest::internal::Tree& t,
    const algorithmFPType* x, const NumericTable* resp, size_t iRow)
{
    ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable*>(resp), iRow, 1);
    return h.predictionError(h.predict(t, x), *y.get());
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
algorithmFPType TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::computeOOBErrorPerm(const decision_forest::internal::Tree& t,
    size_t n, const IndexType* aInd, const IndexType* aPerm, size_t iPermutedFeature)
{
    DAAL_ASSERT(n);

    const auto dim = nFeatures();
    const auto sz = dim*sizeof(algorithmFPType);

    //compute prediction error on each OOB row and get its mean using online formulae (Welford)
    //TODO: can be threader_for() block
    TArray<algorithmFPType, cpu> buf(dim);
    ReadRows<algorithmFPType, cpu> x(const_cast<NumericTable*>(_data), aInd[0], 1);
    daal::services::daal_memcpy_s(buf.get(), sz, x.get(), sz);
    ReadRows<algorithmFPType, cpu> p(const_cast<NumericTable*>(_data), aInd[aPerm[0]], 1);
    buf[iPermutedFeature] = p.get()[iPermutedFeature];
    algorithmFPType mean = predictionError<algorithmFPType, DataHelper, cpu>(_helper, t, buf.get(), _resp, aInd[0]);

    for(size_t i = 1; i < n; ++i)
    {
        daal::services::daal_memcpy_s(buf.get(), sz, x.set(const_cast<NumericTable*>(_data), aInd[i], 1), sz);
        buf[iPermutedFeature] = p.set(const_cast<NumericTable*>(_data), aInd[aPerm[i]], 1)[iPermutedFeature];
        algorithmFPType val = predictionError<algorithmFPType, DataHelper, cpu>(_helper, t, buf.get(), _resp, aInd[i]);
        mean += (val - mean) / algorithmFPType(i + 1);
    }
    return mean;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
algorithmFPType TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::computeOOBError(const decision_forest::internal::Tree& t,
    size_t n, const IndexType* aInd)
{
    DAAL_ASSERT(n);
    //compute prediction error on each OOB row and get its mean online formulae (Welford)
    //TODO: can be threader_for() block
    ReadRows<algorithmFPType, cpu> x(const_cast<NumericTable*>(_data), aInd[0], 1);
    algorithmFPType mean = _helper.predictionError(t, x.get(), _resp, aInd[0], _threadCtx.oobBuf);
    for(size_t i = 1; i < n; ++i)
    {
        algorithmFPType val = _helper.predictionError(t, x.set(const_cast<NumericTable*>(_data), aInd[i], 1),
            _resp, aInd[i], _threadCtx.oobBuf);
        mean += (val - mean) / algorithmFPType(i + 1);
    }
    return mean;
}

} /* namespace internal */
} /* namespace training */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif

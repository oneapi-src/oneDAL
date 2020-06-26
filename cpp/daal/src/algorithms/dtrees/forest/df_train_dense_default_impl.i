/* file: df_train_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/threading/threading.h"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/engines/engine_types_internal.h"
#include "src/services/service_defines.h"
#include "src/algorithms/distributions/uniform/uniform_kernel.h"

using namespace daal::algorithms::dtrees::training::internal;

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
// Service class, it uses to keep information about nodes
//////////////////////////////////////////////////////////////////////////////////////////
template <typename WorkItem>
class WorkQueue
{
public:
    WorkQueue(services::Status & s)
        : _capacity(1024),
          _capacityMinus1(_capacity - 1),
          _first(0),
          _last(_capacityMinus1),
          _size(0),
          _data(new WorkItem[_capacity]) { DAAL_CHECK_COND_ERROR(_data, s, services::ErrorMemoryAllocationFailed) }

          WorkQueue(const WorkQueue &) = delete;

    ~WorkQueue()
    {
        delete[] _data;
        _data = nullptr;
    }

    size_t size() const { return _size; }

    bool empty() const { return (_size == 0); }

    WorkItem & front()
    {
        DAAL_ASSERT(!empty());

        return _data[_first];
    }

    void pop()
    {
        DAAL_ASSERT(!empty());

        ++_first;
        _first *= (_first != _capacity);
        --_size;
    }

    services::Status push(const WorkItem & value)
    {
        if (_size == _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }
        DAAL_ASSERT(_size < _capacity);
        DAAL_ASSERT(((_capacityMinus1 + 1) & _capacityMinus1) == 0);

        if (_size && _data[_last].improvement > value.improvement)
            sort(value);
        else
            _data[_last = (_last + 1) & _capacityMinus1] = value;
        ++_size;
        return services::Status();
    }

private:
    services::Status grow()
    {
        const size_t newCapacity = _capacity * 2;
        DAAL_ASSERT(_size < newCapacity);
        WorkItem * const newData = new WorkItem[newCapacity];
        DAAL_CHECK_MALLOC(newData)
        size_t srcIdx = _first;
        for (size_t i = 0; i < _size; ++i)
        {
            newData[i].moveFrom(_data[srcIdx]);
            ++srcIdx;
            srcIdx *= (srcIdx != _capacity);
        }
        delete[] _data;
        _data           = newData;
        _capacity       = newCapacity;
        _capacityMinus1 = _capacity - 1;
        _first          = 0;
        _last           = _size != 0 ? _size - 1 : _capacityMinus1;
        return services::Status();
    }

    void sort(const WorkItem & value)
    {
        size_t tail = _last;
        size_t head = _first ? _first - 1 : _capacityMinus1;
        while (tail != head)
        {
            if (_data[tail].improvement < value.improvement) break;
            _data[(tail + 1) & _capacityMinus1].moveFrom(_data[tail]);
            tail = tail ? tail - 1 : _capacityMinus1;
        }
        _data[(tail + 1) & _capacityMinus1] = value;
        _last                               = (_last + 1) & _capacityMinus1;
    }

    size_t _capacity;
    size_t _capacityMinus1;
    size_t _first;
    size_t _last;
    size_t _size;
    WorkItem * _data;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains numeric tables to be calculated as result
//////////////////////////////////////////////////////////////////////////////////////////
struct ResultData
{
public:
    ResultData(const Parameter & par, NumericTable * _varImp, NumericTable * _oobError, NumericTable * _oobErrorPerObs)
    {
        if (par.varImportance != decision_forest::training::none) varImp = _varImp;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagError) oobError = _oobError;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation) oobErrorPerObs = _oobErrorPerObs;
    }
    NumericTable * varImp         = nullptr; //if needed then allocated outside kernel
    NumericTable * oobError       = nullptr; //if needed then allocated outside kernel
    NumericTable * oobErrorPerObs = nullptr; //if needed then allocated outside kernel
    NumericTablePtr oobIndices;              //if needed then allocated in kernel
    engines::EnginePtr updatedEngine;        // engine updated after simulations
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains workset required for tree calculation in one thread
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class TreeThreadCtxBase
{
public:
    TreeThreadCtxBase(algorithmFPType * _varImp = nullptr) : varImp(_varImp), varImpVariance(nullptr), nTrees(0), oobBuf(nullptr) {}
    ~TreeThreadCtxBase()
    {
        if (varImpVariance) service_free<algorithmFPType, cpu>(varImpVariance);
        if (oobBuf) service_free<byte, cpu>(oobBuf);
    }
    void finalizeVarImp(training::VariableImportanceMode mode, size_t nVars);

protected:
    bool init(const decision_forest::training::Parameter & par, const NumericTable * x)
    {
        if (par.varImportance == training::MDA_Scaled)
        {
            varImpVariance = service_calloc<algorithmFPType, cpu>(x->getNumberOfColumns());
            DAAL_CHECK_STATUS_VAR(varImpVariance);
        }
        return true;
    }

    void reduceTo(training::VariableImportanceMode mode, TreeThreadCtxBase & other, size_t nVars, size_t nSamples) const
    {
        // Reduces tls variable importance results
        if (varImp)
        {
            if (mode == training::MDI)
            {
                for (size_t i = 0; i < nVars; ++i) other.varImp[i] += varImp[i];
            }
            else
            {
                const algorithmFPType div  = algorithmFPType(1) / algorithmFPType(other.nTrees + nTrees);
                const algorithmFPType n1   = algorithmFPType(nTrees) * div;
                const algorithmFPType n2   = algorithmFPType(other.nTrees) * div;
                const algorithmFPType div2 = algorithmFPType(nTrees * other.nTrees) * div;
                for (size_t i = 0; i < nVars; ++i)
                {
                    const algorithmFPType newM = varImp[i] * n1 + other.varImp[i] * n2;
                    if (varImpVariance)
                    {
                        const algorithmFPType deltaM = varImp[i] - other.varImp[i];
                        const algorithmFPType v1     = varImpVariance[i];
                        const algorithmFPType v2     = other.varImpVariance[i];
                        const algorithmFPType v      = v1 + v2 + deltaM * deltaM * div2;
                        other.varImpVariance[i]      = v;
                    }
                    other.varImp[i] = newM;
                }
            }
        }
        other.nTrees += nTrees;
    }

public:
    algorithmFPType * varImp;
    algorithmFPType * varImpVariance;
    size_t nTrees;
    byte * oobBuf;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Finalizes calculation of variable importance results
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
void TreeThreadCtxBase<algorithmFPType, cpu>::finalizeVarImp(training::VariableImportanceMode mode, size_t nVars)
{
    if (mode == training::MDA_Scaled)
    {
        //average over all trees and scale by its variance
        if (nTrees > 1)
        {
            const algorithmFPType div = 1. / algorithmFPType(nTrees);
            for (size_t i = 0; i < nVars; ++i)
            {
                varImpVariance[i] *= div;
                if (isPositive<algorithmFPType, cpu>(varImpVariance[i]))
                    varImp[i] /= daal::internal::Math<algorithmFPType, cpu>::sSqrt(varImpVariance[i] * div);
            }
        }
        else
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < nVars; ++i) varImp[i] = 0;
        }
    }
    else if (mode == training::MDI)
    {
        const algorithmFPType div = 1. / algorithmFPType(nTrees);
        //average over all trees
        for (size_t i = 0; i < nVars; ++i) varImp[i] *= div;
    }

#if ENABLE_VAR_IMP_NORMALIZATION
    algorithmFPType sum = 0;
    for (size_t i = 0; i < nVars; ++i) sum += varImp[i];
    //normalize by division to the sum of all values
    if (!isPositive<algorithmFPType, cpu>(sum))
    {
        algorithmFPType maxVal = 0;
        for (size_t i = 0; i < nVars; ++i)
        {
            const algorithmFPType val = daal::internal::Math<algorithmFPType, cpu>::sFabs(varImp[i]);
            maxVal                    = daal::internal::Math<algorithmFPType, cpu>::sMax(maxVal, val);
        }
        if (!isPositive<algorithmFPType, cpu>(maxVal)) return;
        const algorithmFPType div = 1. / maxVal;
        for (size_t i = 0; i < nVars; varImp[i++] *= div)
            ;
    }
    else
    {
        sum = 1. / sum;
        for (size_t i = 0; i < nVars; varImp[i++] *= sum)
            ;
    }
#endif
}

template <typename algorithmFPType, CpuType cpu, class Ctx>
Ctx * createTlsContext(const NumericTable * x, const Parameter & par, size_t nClasses)
{
    const size_t szVarImp = (par.varImportance == decision_forest::training::none ? 0 : x->getNumberOfColumns() * sizeof(algorithmFPType));
    size_t szCtx          = sizeof(Ctx);
    if (szCtx % sizeof(algorithmFPType)) szCtx = sizeof(algorithmFPType) * (1 + (szCtx / sizeof(algorithmFPType)));
    const size_t sz = szCtx + szVarImp;
    byte * ptr      = service_scalable_calloc<byte, cpu>(sz);
    Ctx * ctx       = new (ptr) Ctx();
    if (ctx)
    {
        if (szVarImp) ctx->varImp = (algorithmFPType *)(ptr + szCtx);
        if (!ctx->init(par, x, nClasses))
        {
            //allocation of extra data hasn't succeeded
            ctx->~Ctx();
            service_scalable_free<byte, cpu>(ptr);
            return nullptr;
        }
    }
    return ctx;
}

template <CpuType cpu>
services::Status selectParallelizationTechnique(const Parameter & par, engines::internal::ParallelizationTechnique & technique)
{
    auto engineImpl = dynamic_cast<engines::internal::BatchBaseImpl *>(par.engine.get());

    engines::internal::ParallelizationTechnique techniques[] = { engines::internal::family, engines::internal::leapfrog,
                                                                 engines::internal::skipahead };

    for (auto & t : techniques)
    {
        if (engineImpl->hasSupport(t))
        {
            technique = t;
            return services::Status();
        }
    }
    return services::Status(ErrorEngineNotSupported);
}

//////////////////////////////////////////////////////////////////////////////////////////
// compute() implementation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu, typename ModelType, typename TaskType>
services::Status computeImpl(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, ModelType & md, ResultData & res,
                             const Parameter & par, size_t nClasses)
{
    DAAL_CHECK(md.resize(par.nTrees), ErrorMemoryAllocationFailed);
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK(featTypes.init(*x), ErrorMemoryAllocationFailed);
    dtrees::internal::IndexedFeatures indexedFeatures;
    services::Status s;
    if (!par.memorySavingMode)
    {
        s = indexedFeatures.init<algorithmFPType, cpu>(*x, &featTypes);
        DAAL_CHECK_STATUS_VAR(s);
    }

    const auto nFeatures = x->getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> varImpBD(res.varImp, 0, 1);
    if (res.varImp) DAAL_CHECK_BLOCK_STATUS(varImpBD);

    //set of auxiliary data used by single thread or a target for reduction by multiple threads
    typedef typename TaskType::ThreadCtxType Ctx;
    Ctx mainCtx(varImpBD.get());
    DAAL_CHECK(mainCtx.init(par, x, nClasses), ErrorMemoryAllocationFailed);

    if (mainCtx.varImp)
        //initialize its data
        daal::services::internal::service_memset<algorithmFPType, cpu>(mainCtx.varImp, 0, nFeatures);

    //use tls in case of multiple threads
    const bool bThreaded = (threader_get_max_threads_number() > 1) && (par.nTrees > 1);
    daal::tls<Ctx *> tlsCtx([&]() -> Ctx * {
        //in case of single thread no need to allocate
        return (bThreaded ? createTlsContext<algorithmFPType, cpu, Ctx>(x, par, nClasses) : &mainCtx);
    });
    daal::tls<TaskType *> tlsTask([&]() -> TaskType * {
        //in case of single thread no need to allocate
        Ctx * ctx = tlsCtx.local();
        return ctx ? new TaskType(pHostApp, x, y, par, featTypes, par.memorySavingMode ? nullptr : &indexedFeatures, *ctx, nClasses) : nullptr;
    });

    engines::internal::ParallelizationTechnique technique = engines::internal::family;
    selectParallelizationTechnique<cpu>(par, technique);
    engines::internal::Params<cpu> params(par.nTrees);
    for (size_t i = 0; i < par.nTrees; i++)
    {
        params.nSkip[i] = i * par.nTrees * x->getNumberOfRows() * (par.featuresPerNode + 1);
    }
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par.nTrees, sizeof(engines::EnginePtr));
    TArray<engines::EnginePtr, cpu> engines(par.nTrees);
    engines::internal::EnginesCollection<cpu> enginesCollection(par.engine, technique, params, engines, &s);
    DAAL_CHECK_STATUS_VAR(s);

    services::internal::TArray<size_t, cpu> numElems(par.nTrees);

    daal::SafeStatus safeStat;
    daal::threader_for(par.nTrees, par.nTrees, [&](size_t i) {
        if (!safeStat.ok()) return;
        TaskType * task = tlsTask.local();
        DAAL_CHECK_MALLOC_THR(task);
        dtrees::internal::Tree * pTree = nullptr;
        numElems[i]                    = 0;
        auto engineImpl                = dynamic_cast<engines::internal::BatchBaseImpl *>(engines[i].get());
        DAAL_CHECK_THR(engineImpl, ErrorEngineNotSupported);
        services::Status s = task->run(engineImpl, pTree, numElems[i]);
        DAAL_CHECK_STATUS_THR(s);
        if (pTree)
        {
            md.add((typename ModelType::TreeType &)*pTree, nClasses);
        }
    });
    s                = safeStat.detach();
    const auto nRows = x->getNumberOfRows();
    tlsCtx.reduce([&](Ctx * ctx) -> void {
        if (ctx && bThreaded)
        {
            ctx->reduceTo(par.varImportance, mainCtx, nFeatures, nRows);
            ctx->~Ctx();
            service_scalable_free<byte, cpu>((byte *)ctx);
        }
    });
    tlsTask.parallel_reduce([&](TaskType * task) -> void {
        delete task;
        task = nullptr;
    });
    DAAL_CHECK_STATUS_VAR(s);
    DAAL_CHECK_MALLOC(md.size() == par.nTrees);

    res.updatedEngine = enginesCollection.getUpdatedEngine(par.engine, engines, numElems);

    //finalize results computation
    //variable importance
    if (varImpBD.get()) mainCtx.finalizeVarImp(par.varImportance, nFeatures);

    //OOB error
    if (par.resultsToCompute & (computeOutOfBagError | computeOutOfBagErrorPerObservation))
    {
        WriteOnlyRows<algorithmFPType, cpu> oobErr(res.oobError, 0, 1);
        if (par.resultsToCompute & computeOutOfBagError) DAAL_CHECK_BLOCK_STATUS(oobErr);

        WriteOnlyRows<algorithmFPType, cpu> oobErrPerObs(res.oobErrorPerObs, 0, nRows);
        if (par.resultsToCompute & computeOutOfBagErrorPerObservation) DAAL_CHECK_BLOCK_STATUS(oobErrPerObs);

        s = mainCtx.finalizeOOBError(y, oobErr.get(), oobErrPerObs.get());
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
    typedef TreeThreadCtxBase<algorithmFPType, cpu> ThreadCtxType;
    services::Status run(engines::internal::BatchBaseImpl * engineImpl, dtrees::internal::Tree *& pTree, size_t & numElems);

protected:
    typedef dtrees::internal::TVector<algorithmFPType, cpu> algorithmFPTypeArray;
    typedef dtrees::internal::TVector<IndexType, cpu> IndexTypeArray;
    TrainBatchTaskBase(HostAppIface * hostApp, const NumericTable * x, const NumericTable * y, const Parameter & par,
                       const dtrees::internal::FeatureTypes & featTypes, const dtrees::internal::IndexedFeatures * indexedFeatures,
                       ThreadCtxType & threadCtx, size_t nClasses)
        : _hostApp(hostApp, 0), //set granularity later
          _data(x),
          _resp(y),
          _par(par),
          _nClasses(nClasses),
          _nSamples(par.observationsPerTreeFraction * x->getNumberOfRows()),
          _nFeaturesPerNode(par.featuresPerNode),
          _helper(indexedFeatures, nClasses),
          _impurityThreshold(_par.impurityThreshold),
          _nFeatureBufs(1), //for sequential processing
          _featHelper(featTypes),
          _threadCtx(threadCtx),
          _accuracy(daal::services::internal::EpsilonVal<algorithmFPType>::get()),
          _minSamplesSplit(2),
          _minWeightLeaf(0.),
          _minImpurityDecrease(-daal::services::internal::EpsilonVal<algorithmFPType>::get() * x->getNumberOfRows()),
          _remainingSplitNodes(2)
    {
        if (_impurityThreshold < _accuracy) _impurityThreshold = _accuracy;

        const daal::algorithms::decision_forest::training::interface2::Parameter * algParameter =
            dynamic_cast<const daal::algorithms::decision_forest::training::interface2::Parameter *>(&par);
        if (algParameter != NULL)
        {
            _minSamplesSplit     = 2 > par.minObservationsInSplitNode ? 2 : par.minObservationsInSplitNode;
            _minSamplesSplit     = _minSamplesSplit > 2 * par.minObservationsInLeafNode ? _minSamplesSplit : 2 * par.minObservationsInLeafNode;
            _minWeightLeaf       = par.minWeightFractionInLeafNode * x->getNumberOfRows(); // no sample_weight
            _minImpurityDecrease = par.minImpurityDecreaseInSplitNode * x->getNumberOfRows()
                                   - daal::services::internal::EpsilonVal<algorithmFPType>::get() * x->getNumberOfRows();
            _remainingSplitNodes = 2;
        }
    }

    size_t nFeatures() const { return _data->getNumberOfColumns(); }
    typename DataHelper::NodeType::Base * build(services::Status & s, size_t iStart, size_t n, size_t level,
                                                typename DataHelper::ImpurityData & curImpurity, bool & bUnorderedFeaturesUsed, size_t nClasses);
    typename DataHelper::NodeType::Base * buildBF(services::Status & s, size_t iStart, size_t n, size_t level,
                                                  typename DataHelper::ImpurityData & curImpurity, bool & bUnorderedFeaturesUsed, size_t nClasses);

    algorithmFPType * featureBuf(size_t iBuf) const
    {
        DAAL_ASSERT(iBuf < _nFeatureBufs);
        return _aFeatureBuf[iBuf].get();
    }
    IndexType * featureIndexBuf(size_t iBuf) const
    {
        DAAL_ASSERT(iBuf < _nFeatureBufs);
        return _aFeatureIndexBuf[iBuf].get();
    }
    bool terminateCriteria(size_t nSamples, size_t level, typename DataHelper::ImpurityData & imp) const
    {
        const daal::algorithms::decision_forest::training::interface2::Parameter * algParameter =
            dynamic_cast<const daal::algorithms::decision_forest::training::interface2::Parameter *>(&_par);
        if (algParameter != NULL)
        {
            return ((nSamples < 2 * _par.minObservationsInLeafNode) || (nSamples < _minSamplesSplit) || (nSamples < 2 * _minWeightLeaf)
                    || _helper.terminateCriteria(imp, _impurityThreshold, nSamples) || ((_par.maxTreeDepth > 0) && (level >= _par.maxTreeDepth)));
        }
        else
            return (nSamples < 2 * _par.minObservationsInLeafNode || _helper.terminateCriteria(imp, _impurityThreshold, nSamples)
                    || ((_par.maxTreeDepth > 0) && (level >= _par.maxTreeDepth)));
    }
    ThreadCtxType & threadCtx() { return _threadCtx; }
    typename DataHelper::NodeType::Split * makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered,
                                                     typename DataHelper::NodeType::Base * left, typename DataHelper::NodeType::Base * right,
                                                     algorithmFPType imp);
    typename DataHelper::NodeType::Leaf * makeLeaf(const IndexType * idx, size_t n, typename DataHelper::ImpurityData & imp, size_t makeLeaf);

    bool findBestSplit(size_t iStart, size_t n, const typename DataHelper::ImpurityData & curImpurity, IndexType & iBestFeature,
                       typename DataHelper::TSplitData & split);
    bool findBestSplitSerial(size_t iStart, size_t n, const typename DataHelper::ImpurityData & curImpurity, IndexType & iBestFeature,
                             typename DataHelper::TSplitData & split);
    bool findBestSplitThreaded(size_t iStart, size_t n, const typename DataHelper::ImpurityData & curImpurity, IndexType & iBestFeature,
                               typename DataHelper::TSplitData & split);
    bool simpleSplit(size_t iStart, const typename DataHelper::ImpurityData & curImpurity, IndexType & iFeatureBest,
                     typename DataHelper::TSplitData & split);
    void addImpurityDecrease(IndexType iFeature, size_t n, const typename DataHelper::ImpurityData & curImpurity,
                             const typename DataHelper::TSplitData & split);

    void featureValuesToBuf(size_t iFeature, algorithmFPType * featureVal, IndexType * aIdx, size_t n)
    {
        _helper.getColumnValues(iFeature, aIdx, n, featureVal);
        daal::algorithms::internal::qSort<algorithmFPType, int, cpu>(n, featureVal, aIdx);
    }

    //find features to check in the current split node
    void chooseFeatures()
    {
        const size_t n = nFeatures();
        if (n == _nFeaturesPerNode)
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < n; ++i) _aFeatureIdx[i] = i;
        }
        else
        {
            *_numElems += n;
            RNGs<IndexType, cpu> rng;
            rng.uniformWithoutReplacement(_nFeaturesPerNode, _aFeatureIdx.get(), _aFeatureIdx.get() + _nFeaturesPerNode, _engineImpl->getState(), 0,
                                          n);
        }
    }

    services::Status computeResults(const dtrees::internal::Tree & t);

    algorithmFPType computeOOBError(const dtrees::internal::Tree & t, size_t n, const IndexType * aInd);

    algorithmFPType computeOOBErrorPerm(const dtrees::internal::Tree & t, size_t n, const IndexType * aInd, const IndexType * aPerm,
                                        size_t iPermutedFeature);

    void setupHostApp()
    {
        const size_t minPart = 4 * _helper.size();        //corresponds to the 4 topmost levels
        const size_t minSize = 24000 / _nFeaturesPerNode; //at least that many, corresponds to the tree 1000 obs/10 features/8 levels
        _hostApp.setup(minPart < minSize ? minSize : minPart);
    }

protected:
    TArray<IndexType, cpu> _aFeatureIdx; //indices of features to be used for the soplit at the current level
    DataHelper _helper;
    services::internal::HostAppHelper _hostApp;
    typename DataHelper::TreeType _tree;
    mutable TVector<IndexType, cpu> _aSample;
    mutable TArray<algorithmFPTypeArray, cpu> _aFeatureBuf;
    mutable TArray<IndexTypeArray, cpu> _aFeatureIndexBuf;
    engines::internal::BatchBaseImpl * _engineImpl;
    const NumericTable * _data;
    const NumericTable * _resp;
    const Parameter & _par;
    const size_t _nSamples;
    const size_t _nFeaturesPerNode;
    const size_t _nFeatureBufs; //number of buffers to get feature values (to process features independently in parallel)

    const FeatureTypes & _featHelper;
    algorithmFPType _accuracy;
    algorithmFPType _impurityThreshold;
    ThreadCtxType & _threadCtx;
    size_t _nClasses;
    size_t * _numElems;
    size_t _minSamplesSplit;
    double _minWeightLeaf;
    double _minImpurityDecrease;
    size_t _remainingSplitNodes;
};

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
services::Status TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::run(engines::internal::BatchBaseImpl * engineImpl,
                                                                           dtrees::internal::Tree *& pTree, size_t & numElems)
{
    _numElems   = &numElems;
    _engineImpl = engineImpl;
    pTree       = nullptr;
    _tree.destroy();
    _aSample.reset(_nSamples);
    _aFeatureBuf.reset(_nFeatureBufs);
    _aFeatureIndexBuf.reset(_nFeatureBufs);
    _aFeatureIdx.reset(_nFeaturesPerNode * 2); // _nFeaturesPerNode elements are used by algorithm, others are used internally by generator

    DAAL_CHECK_MALLOC(_aSample.get() && _helper.reset(_nSamples) && _aFeatureBuf.get() && _aFeatureIndexBuf.get() && _aFeatureIdx.get());

    //allocate temporary bufs

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < _nFeatureBufs; ++i)
    {
        _aFeatureBuf[i].reset(_nSamples);
        DAAL_CHECK_MALLOC(_aFeatureBuf[i].get());
        _aFeatureIndexBuf[i].reset(_nSamples);
        DAAL_CHECK_MALLOC(_aFeatureIndexBuf[i].get());
    }

    if (_par.bootstrap)
    {
        *_numElems += _nSamples;
        RNGs<int, cpu> rng;
        rng.uniform(_nSamples, _aSample.get(), _engineImpl->getState(), 0, _data->getNumberOfRows());
        daal::algorithms::internal::qSort<int, cpu>(_nSamples, _aSample.get());
    }
    else
    {
        auto aSample = _aSample.get();
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nSamples; ++i) aSample[i] = i;
    }
    //init responses buffer, keep _aSample values in it
    DAAL_CHECK_MALLOC(_helper.init(_data, _resp, _aSample.get()));

    //use _aSample as an array of response indices stored by helper from now on
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < _aSample.size(); ++i) _aSample[i] = i;

    setupHostApp();

    typename DataHelper::ImpurityData initialImpurity;
    _helper.calcImpurity(_aSample.get(), _nSamples, initialImpurity);
    bool bUnorderedFeaturesUsed = false;
    services::Status s;
    typename DataHelper::NodeType::Base * nd = buildBF(s, 0, _nSamples, 0, initialImpurity, bUnorderedFeaturesUsed, _nClasses);
    if (nd)
    {
        //to prevent memory leak in case of general allocator
        _tree.reset(nd, bUnorderedFeaturesUsed);
        _threadCtx.nTrees++;
    }

    if (s && ((_par.resultsToCompute & (computeOutOfBagError | computeOutOfBagErrorPerObservation)) || (_par.varImportance > MDI)))
        s = computeResults(_tree);
    if (s) pTree = &_tree;
    return s;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
typename DataHelper::NodeType::Split * TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::makeSplit(size_t iFeature, algorithmFPType featureValue,
                                                                                                       bool bUnordered,
                                                                                                       typename DataHelper::NodeType::Base * left,
                                                                                                       typename DataHelper::NodeType::Base * right,
                                                                                                       algorithmFPType imp)
{
    typename DataHelper::NodeType::Split * pNode = _tree.allocator().allocSplit();
    pNode->set(iFeature, featureValue, bUnordered);
    pNode->kid[0]   = left;
    pNode->kid[1]   = right;
    pNode->impurity = imp;
    return pNode;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
typename DataHelper::NodeType::Leaf * TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::makeLeaf(const IndexType * idx, size_t n,
                                                                                                     typename DataHelper::ImpurityData & imp,
                                                                                                     size_t nClasses)
{
    typename DataHelper::NodeType::Leaf * pNode = _tree.allocator().allocLeaf(_nClasses);
    _helper.setLeafData(*pNode, idx, n, imp);
    return pNode;
}

// Deep-first
template <typename algorithmFPType, typename DataHelper, CpuType cpu>
typename DataHelper::NodeType::Base * TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::build(services::Status & s, size_t iStart, size_t n,
                                                                                                  size_t level,
                                                                                                  typename DataHelper::ImpurityData & curImpurity,
                                                                                                  bool & bUnorderedFeaturesUsed, size_t nClasses)
{
    if (_hostApp.isCancelled(s, n)) return nullptr;

    if (terminateCriteria(n, level, curImpurity)) return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);

    typename DataHelper::TSplitData split;
    IndexType iFeature;
    if (findBestSplit(iStart, n, curImpurity, iFeature, split))
    {
        const size_t nLeft   = split.nLeft;
        const double imp     = curImpurity.var;
        const double impLeft = split.left.var;

        // check impurity decrease
        if (imp * n - impLeft * nLeft - (n - nLeft) * (imp - impLeft) < _minImpurityDecrease)
            return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);
        if (_par.varImportance == training::MDI) addImpurityDecrease(iFeature, n, curImpurity, split);
        typename DataHelper::NodeType::Base * left = build(s, iStart, split.nLeft, level + 1, split.left, bUnorderedFeaturesUsed, nClasses);
        _helper.convertLeftImpToRight(n, curImpurity, split);
        typename DataHelper::NodeType::Base * right =
            s.ok() ? build(s, iStart + nLeft, split.nLeft, level + 1, split.left, bUnorderedFeaturesUsed, nClasses) : nullptr;
        typename DataHelper::NodeType::Base * res = nullptr;
        if (!left || !right || !(res = makeSplit(iFeature, split.featureValue, split.featureUnordered, left, right, curImpurity.var)))
        {
            if (left) dtrees::internal::deleteNode<typename DataHelper::NodeType, typename DataHelper::TreeType::Allocator>(left, _tree.allocator());
            if (right)
                dtrees::internal::deleteNode<typename DataHelper::NodeType, typename DataHelper::TreeType::Allocator>(right, _tree.allocator());
            return nullptr;
        }
        bUnorderedFeaturesUsed |= bool(split.featureUnordered);
        res->count = n;
        DAAL_ASSERT(nLeft == left->count);
        DAAL_ASSERT(split.nLeft == right->count);
        return res;
    }
    return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);
}

// Best-first
template <typename algorithmFPType, typename DataHelper, CpuType cpu>
typename DataHelper::NodeType::Base * TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::buildBF(services::Status & s, size_t iStart, size_t n,
                                                                                                    size_t level,
                                                                                                    typename DataHelper::ImpurityData & curImpurity,
                                                                                                    bool & bUnorderedFeaturesUsed, size_t nClasses)
{
    struct WorkItem
    {
        bool featureUnordered;
        size_t start;
        size_t n;
        size_t nLeft;
        size_t level;
        double improvement;
        typename DataHelper::ImpurityData impurityLeft;
        typename DataHelper::ImpurityData impurityRight;
        typename DataHelper::NodeType::Split * node;

        WorkItem() {}

        WorkItem(bool featureUnordered, size_t start, size_t n, size_t level)
            : featureUnordered(featureUnordered), start(start), n(n), nLeft(0), level(level), improvement(0.0), node(nullptr)
        {}

        void moveFrom(WorkItem & src)
        {
            DAAL_ASSERT(this != &src);

            featureUnordered = src.featureUnordered;
            start            = src.start;
            n                = src.n;
            nLeft            = src.nLeft;
            level            = src.level;
            improvement      = src.improvement;
            impurityLeft     = src.impurityLeft;
            impurityRight    = src.impurityRight;
            node             = src.node;
        }
    };

    if (_hostApp.isCancelled(s, n)) return nullptr;
    WorkQueue<WorkItem> workQueue(s);
    if (!s.ok()) return nullptr;
    --_remainingSplitNodes; //_remainingSplitNodes = maxLeafNodes - 1

    // Create base node
    typename DataHelper::TSplitData split;
    IndexType iFeature;
    WorkItem base(bUnorderedFeaturesUsed, iStart, n, level);

    if (terminateCriteria(n, level, curImpurity)) return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);
    if (findBestSplit(iStart, n, curImpurity, iFeature, split))
    {
        const size_t nLeft   = split.nLeft;
        const double imp     = curImpurity.var;
        const double impLeft = split.left.var;

        // check impurity decrease
        double improvement = imp * n - impLeft * nLeft - (n - nLeft) * (imp - impLeft);
        if (improvement < _minImpurityDecrease) return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);
        if (_par.varImportance == training::MDI) addImpurityDecrease(iFeature, n, curImpurity, split);

        base.nLeft        = split.nLeft;
        base.improvement  = improvement;
        base.impurityLeft = split.left;
        _helper.convertLeftImpToRight(n, curImpurity, split);
        base.impurityRight = split.left;
        if (!(base.node = makeSplit(iFeature, split.featureValue, split.featureUnordered, nullptr, nullptr, curImpurity.var))) return nullptr;
        base.featureUnordered |= bool(split.featureUnordered);
        base.node->count = n;

        s = workQueue.push(base);
        if (!s.ok()) return nullptr;
    }
    else
        return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);

    while (!workQueue.empty())
    {
        WorkItem & src = workQueue.front();
        workQueue.pop();
        //_remainingSplitNodes = _remainingSplitNodes ? _remainingSplitNodes - 1 : 0;

        // Create leftChild
        typename DataHelper::TSplitData splitLeft;
        IndexType iFeatureLeft;

        if (!_remainingSplitNodes || terminateCriteria(src.nLeft, src.level + 1, src.impurityLeft))
            src.node->kid[0] = makeLeaf(_aSample.get() + src.start, src.nLeft, src.impurityLeft, nClasses);
        else if (findBestSplit(src.start, src.nLeft, src.impurityLeft, iFeatureLeft, splitLeft))
        {
            const size_t nLeft   = splitLeft.nLeft;
            const double imp     = src.impurityLeft.var;
            const double impLeft = splitLeft.left.var;

            // check impurity decrease
            double improvement = imp * src.nLeft - impLeft * nLeft - (src.nLeft - nLeft) * (imp - impLeft);
            if (improvement < _minImpurityDecrease)
                src.node->kid[0] = makeLeaf(_aSample.get() + src.start, src.nLeft, src.impurityLeft, nClasses);
            else
            {
                if (_par.varImportance == training::MDI) addImpurityDecrease(iFeatureLeft, src.nLeft, src.impurityLeft, splitLeft);
                WorkItem leftChild(src.featureUnordered, src.start, src.nLeft, src.level + 1);

                leftChild.nLeft        = splitLeft.nLeft;
                leftChild.improvement  = improvement;
                leftChild.impurityLeft = splitLeft.left;
                _helper.convertLeftImpToRight(src.nLeft, src.impurityLeft, splitLeft);
                leftChild.impurityRight = splitLeft.left;

                if (!(leftChild.node =
                          makeSplit(iFeatureLeft, splitLeft.featureValue, splitLeft.featureUnordered, nullptr, nullptr, src.impurityLeft.var)))
                    return nullptr;

                src.node->kid[0] = leftChild.node;
                leftChild.featureUnordered |= bool(splitLeft.featureUnordered);
                leftChild.node->count = src.nLeft;

                s = workQueue.push(leftChild);
                if (!s.ok()) return nullptr;
            }
        }
        else
            src.node->kid[0] = makeLeaf(_aSample.get() + src.start, src.nLeft, src.impurityLeft, nClasses);

        // Create rightChild
        typename DataHelper::TSplitData splitRight;
        IndexType iFeatureRight;

        if (!_remainingSplitNodes || terminateCriteria(src.n - src.nLeft, src.level + 1, src.impurityRight))
            src.node->kid[1] = makeLeaf(_aSample.get() + src.start + src.nLeft, src.n - src.nLeft, src.impurityRight, nClasses);
        else if (findBestSplit(src.start + src.nLeft, src.n - src.nLeft, src.impurityRight, iFeatureRight, splitRight))
        {
            const size_t nLeft   = splitRight.nLeft;
            const double imp     = src.impurityRight.var;
            const double impLeft = splitRight.left.var;

            // check impurity decrease
            double improvement = imp * (src.n - src.nLeft) - impLeft * nLeft - (src.n - src.nLeft - nLeft) * (imp - impLeft);
            if (improvement < _minImpurityDecrease)
                src.node->kid[1] = makeLeaf(_aSample.get() + src.start + src.nLeft, src.n - src.nLeft, src.impurityRight, nClasses);
            else
            {
                if (_par.varImportance == training::MDI) addImpurityDecrease(iFeatureRight, src.n - src.nLeft, src.impurityRight, splitRight);
                WorkItem rightChild(src.featureUnordered, src.start + src.nLeft, src.n - src.nLeft, src.level + 1);

                rightChild.nLeft        = splitRight.nLeft;
                rightChild.improvement  = improvement;
                rightChild.impurityLeft = splitRight.left;
                _helper.convertLeftImpToRight(src.n - src.nLeft, src.impurityRight, splitRight);
                rightChild.impurityRight = splitRight.left;

                if (!(rightChild.node =
                          makeSplit(iFeatureRight, splitRight.featureValue, splitRight.featureUnordered, nullptr, nullptr, src.impurityRight.var)))
                    return nullptr;

                src.node->kid[1] = rightChild.node;
                rightChild.featureUnordered |= bool(splitRight.featureUnordered);
                rightChild.node->count = src.n - src.nLeft;

                s = workQueue.push(rightChild);
                if (!s.ok()) return nullptr;
            }
        }
        else
            src.node->kid[1] = makeLeaf(_aSample.get() + src.start + src.nLeft, src.n - src.nLeft, src.impurityRight, nClasses);
    }
    return base.node;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::simpleSplit(size_t iStart, const typename DataHelper::ImpurityData & curImpurity,
                                                                       IndexType & iFeatureBest, typename DataHelper::TSplitData & split)
{
    RNGs<IndexType, cpu> rng;
    algorithmFPType featBuf[2];
    IndexType * aIdx = _aSample.get() + iStart;
    for (size_t i = 0; i < _nFeaturesPerNode; ++i)
    {
        IndexType iFeature;
        *_numElems += 1;
        rng.uniform(1, &iFeature, _engineImpl->getState(), 0, _data->getNumberOfColumns());
        featureValuesToBuf(iFeature, featBuf, aIdx, 2);
        if (featBuf[1] - featBuf[0] <= _accuracy) //all values of the feature are the same
            continue;
        _helper.simpleSplit(featBuf, aIdx, split);
        split.featureUnordered = _featHelper.isUnordered(iFeature);
        split.impurityDecrease = curImpurity.var;
        iFeatureBest           = iFeature;
        return true;
    }
    return false;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::findBestSplit(size_t iStart, size_t n,
                                                                         const typename DataHelper::ImpurityData & curImpurity,
                                                                         IndexType & iFeatureBest, typename DataHelper::TSplitData & split)
{
    if (n == 2)
    {
        DAAL_ASSERT(_par.minObservationsInLeafNode == 1);
#ifdef DEBUG_CHECK_IMPURITY
        _helper.checkImpurity(_aSample.get() + iStart, n, curImpurity);
#endif
        return simpleSplit(iStart, curImpurity, iFeatureBest, split);
    }
    if (_nFeatureBufs == 1) return findBestSplitSerial(iStart, n, curImpurity, iFeatureBest, split);
    return findBestSplitThreaded(iStart, n, curImpurity, iFeatureBest, split);
}

//find best split and put it to featureIndexBuf
template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::findBestSplitSerial(size_t iStart, size_t n,
                                                                               const typename DataHelper::ImpurityData & curImpurity,
                                                                               IndexType & iBestFeature, typename DataHelper::TSplitData & bestSplit)
{
    chooseFeatures();
    const float qMax             = 0.02; //min fracture of observations to be handled as indexed feature values
    IndexType * bestSplitIdx     = featureIndexBuf(0) + iStart;
    IndexType * aIdx             = _aSample.get() + iStart;
    int iBestSplit               = -1;
    int idxFeatureValueBestSplit = -1; //when sorted feature is used
    typename DataHelper::TSplitData split;
    const float fact = float(n);
    for (size_t i = 0; i < _nFeaturesPerNode; ++i)
    {
        const auto iFeature            = _aFeatureIdx[i];
        const bool bUseIndexedFeatures = (!_par.memorySavingMode) && (fact > qMax * float(_helper.indexedFeatures().numIndices(iFeature)));

        if (bUseIndexedFeatures)
        {
            if (!_helper.hasDiffFeatureValues(iFeature, aIdx, n)) continue; //all values of the feature are the same
            split.featureUnordered = _featHelper.isUnordered(iFeature);
            //index of best feature value in the array of sorted feature values
            const int idxFeatureValue = _helper.findBestSplitForFeatureSorted(featureBuf(0), iFeature, aIdx, n, _par.minObservationsInLeafNode,
                                                                              curImpurity, split, _minWeightLeaf);
            if (idxFeatureValue < 0) continue;
            iBestSplit = i;
            split.copyTo(bestSplit);
            idxFeatureValueBestSplit = idxFeatureValue;
        }
        else
        {
            algorithmFPType * featBuf = featureBuf(0) + iStart; //single thread
            featureValuesToBuf(iFeature, featBuf, aIdx, n);
            if (featBuf[n - 1] - featBuf[0] <= _accuracy) //all values of the feature are the same
                continue;
#ifdef DEBUG_CHECK_IMPURITY
            _helper.checkImpurity(aIdx, n, curImpurity);
#endif
            split.featureUnordered = _featHelper.isUnordered(iFeature);
            if (!_helper.findBestSplitForFeature(featBuf, aIdx, n, _par.minObservationsInLeafNode, _accuracy, curImpurity, split, _minWeightLeaf))
                continue;
            idxFeatureValueBestSplit = -1;
            iBestSplit               = i;
            split.copyTo(bestSplit);
            DAAL_ASSERT(bestSplit.iStart < n);
            DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= n);
            if (i + 1 < _nFeaturesPerNode || split.featureUnordered) services::internal::tmemcpy<IndexType, cpu>(bestSplitIdx, aIdx, n);
#ifdef DEBUG_CHECK_IMPURITY
            _helper.checkImpurity(aIdx, bestSplit.nLeft, bestSplit.left);
#endif
        }
    }
    if (iBestSplit < 0) return false; //not found

    iBestFeature    = _aFeatureIdx[iBestSplit];
    bool bCopyToIdx = true;
    if (idxFeatureValueBestSplit >= 0)
    {
        //sorted feature was used
        //calculate impurity and get split to bestSplitIdx
        _helper.finalizeBestSplit(aIdx, n, iBestFeature, idxFeatureValueBestSplit, bestSplit, bestSplitIdx);
    }
    else if (bestSplit.featureUnordered)
    {
        if (bestSplit.iStart)
        {
            DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= n);
            services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx + bestSplit.iStart, bestSplit.nLeft);
            aIdx += bestSplit.nLeft;
            services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, bestSplit.iStart);
            aIdx += bestSplit.iStart;
            bestSplitIdx += bestSplit.iStart + bestSplit.nLeft;
            if (n > (bestSplit.iStart + bestSplit.nLeft))
                services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, n - bestSplit.iStart - bestSplit.nLeft);
            bCopyToIdx = false; //done
        }
    }
    else
        bCopyToIdx = (iBestSplit + 1 < _nFeaturesPerNode); //if iBestSplit is the last considered feature
                                                           //then aIdx already contains the best split, no need to copy
    if (bCopyToIdx) services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, n);
    return true;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::findBestSplitThreaded(size_t iStart, size_t n,
                                                                                 const typename DataHelper::ImpurityData & curImpurity,
                                                                                 IndexType & iFeatureBest, typename DataHelper::TSplitData & split)
{
    chooseFeatures();
    TArray<typename DataHelper::TSplitData, cpu> aFeatureSplit(_nFeaturesPerNode);
    //TODO, if parallel for features
    return false;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::addImpurityDecrease(IndexType iFeature, size_t n,
                                                                               const typename DataHelper::ImpurityData & curImpurity,
                                                                               const typename DataHelper::TSplitData & split)
{
    DAAL_ASSERT(_threadCtx.varImp);
    if (!isZero<algorithmFPType, cpu>(split.impurityDecrease)) _threadCtx.varImp[iFeature] += split.impurityDecrease;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
services::Status TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::computeResults(const dtrees::internal::Tree & t)
{
    const size_t nOOB = _helper.getNumOOBIndices();
    if (!nOOB) return services::Status();
    TArray<IndexType, cpu> oobIndices(nOOB);
    DAAL_CHECK_MALLOC(oobIndices.get());
    _helper.getOOBIndices(oobIndices.get());
    const bool bMDA(_par.varImportance == training::MDA_Raw || _par.varImportance == training::MDA_Scaled);
    if (_par.resultsToCompute & (computeOutOfBagError | computeOutOfBagErrorPerObservation) || bMDA)
    {
        const algorithmFPType oobError = computeOOBError(t, nOOB, oobIndices.get());
        if (bMDA)
        {
            TArray<IndexType, cpu> permutation(nOOB);
            DAAL_CHECK_MALLOC(permutation.get());
            for (size_t i = 0; i < nOOB; permutation[i] = i, ++i)
                ;
            const size_t nTrees        = _threadCtx.nTrees;
            const algorithmFPType div1 = algorithmFPType(1) / algorithmFPType(nTrees);
            for (size_t i = 0, n = nFeatures(); i < n; ++i)
            {
                shuffle<cpu>(_engineImpl->getState(), nOOB, permutation.get());
                const algorithmFPType permOOBError = computeOOBErrorPerm(t, nOOB, oobIndices.get(), permutation.get(), i);
                const algorithmFPType diff         = (permOOBError - oobError);
                //_threadCtx.varImp[i] is a mean of diff among all the trees
                const algorithmFPType delta = diff - _threadCtx.varImp[i]; //old mean
                _threadCtx.varImp[i] += div1 * delta;
                if (_threadCtx.varImpVariance) _threadCtx.varImpVariance[i] += delta * (diff - _threadCtx.varImp[i]); //new mean
            }
        }
    }
    return services::Status();
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
algorithmFPType predictionError(const DataHelper & h, const dtrees::internal::Tree & t, const algorithmFPType * x, const NumericTable * resp,
                                size_t iRow)
{
    ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable *>(resp), iRow, 1);
    return h.predictionError(h.predict(t, x), *y.get());
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
algorithmFPType TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::computeOOBErrorPerm(const dtrees::internal::Tree & t, size_t n,
                                                                                          const IndexType * aInd, const IndexType * aPerm,
                                                                                          size_t iPermutedFeature)
{
    DAAL_ASSERT(n);

    const auto dim = nFeatures();

    //compute prediction error on each OOB row and get its mean using online formulae (Welford)
    //TODO: can be threader_for() block
    TArray<algorithmFPType, cpu> buf(dim);
    ReadRows<algorithmFPType, cpu> x(const_cast<NumericTable *>(_data), aInd[0], 1);
    services::internal::tmemcpy<algorithmFPType, cpu>(buf.get(), x.get(), dim);
    ReadRows<algorithmFPType, cpu> p(const_cast<NumericTable *>(_data), aInd[aPerm[0]], 1);
    buf[iPermutedFeature] = p.get()[iPermutedFeature];
    algorithmFPType mean  = predictionError<algorithmFPType, DataHelper, cpu>(_helper, t, buf.get(), _resp, aInd[0]);

    for (size_t i = 1; i < n; ++i)
    {
        services::internal::tmemcpy<algorithmFPType, cpu>(buf.get(), x.set(const_cast<NumericTable *>(_data), aInd[i], 1), dim);
        buf[iPermutedFeature] = p.set(const_cast<NumericTable *>(_data), aInd[aPerm[i]], 1)[iPermutedFeature];
        algorithmFPType val   = predictionError<algorithmFPType, DataHelper, cpu>(_helper, t, buf.get(), _resp, aInd[i]);
        mean += (val - mean) / algorithmFPType(i + 1);
    }
    return mean;
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
algorithmFPType TrainBatchTaskBase<algorithmFPType, DataHelper, cpu>::computeOOBError(const dtrees::internal::Tree & t, size_t n,
                                                                                      const IndexType * aInd)
{
    DAAL_ASSERT(n);
    //compute prediction error on each OOB row and get its mean online formulae (Welford)
    //TODO: can be threader_for() block
    ReadRows<algorithmFPType, cpu> x(const_cast<NumericTable *>(_data), aInd[0], 1);
    algorithmFPType mean = _helper.predictionError(t, x.get(), _resp, aInd[0], _threadCtx.oobBuf);
    for (size_t i = 1; i < n; ++i)
    {
        algorithmFPType val = _helper.predictionError(t, x.set(const_cast<NumericTable *>(_data), aInd[i], 1), _resp, aInd[i], _threadCtx.oobBuf);
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

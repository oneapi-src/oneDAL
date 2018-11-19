/* file: gbt_train_tree_builder.i */
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

#ifndef __GBT_TRAIN_TREE_BUILDER_I__
#define __GBT_TRAIN_TREE_BUILDER_I__

#include "dtrees_model_impl.h"
#include "dtrees_train_data_helper.i"
#include "dtrees_predict_dense_default_impl.i"
#include "gbt_train_aux.i"

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

template<typename algorithmFPType, CpuType cpu>
class TreeBuilder : public TreeBuilderBase
{
public:
    typedef TrainBatchTaskBaseXBoost<algorithmFPType, cpu> CommonCtx;
    using MemHelperType = MemHelperBase<algorithmFPType, cpu>;
    typedef typename CommonCtx::DataHelperType DataHelperType;

    typedef gh<algorithmFPType, cpu> ghType;
    typedef ghSum<algorithmFPType, cpu> ghSumType;
    typedef SplitJob<algorithmFPType, cpu> SplitJobType;
    typedef gbt::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    typedef SplitData<algorithmFPType, ImpurityType> SplitDataType;

    struct SplitTask : public SplitJobType
    {
        typedef TreeBuilder<algorithmFPType, cpu> Task;
        typedef SplitJobType super;

        SplitTask(const SplitTask& o) : super(o), _task(o._task){}
        SplitTask(Task& task, size_t _iStart, size_t _n, size_t _level, const ImpurityType& _imp, NodeType::Base*& _res) :
            super(_iStart, _n, _level, _imp, _res), _task(task){}
        Task& _task;
        void operator()()
        {
            _task._ctx._nParallelNodes.inc();
            _task.buildSplit(*this);
            _task._ctx._nParallelNodes.dec();
        }
    };

    class BestSplit
    {
    public:
        BestSplit(SplitDataType& split, Mutex* mt) :
            _split(split), _mt(mt), _iIndexedFeatureSplitValue(-1), _iFeature(-1){}
        void safeGetData(algorithmFPType& impDec, int& iFeature)
        {
            if(_mt)
            {
                _mt->lock();
                impDec = impurityDecrease();
                iFeature = _iFeature;
                _mt->unlock();
            }
            else
            {
                impDec = impurityDecrease();
                iFeature = _iFeature;
            }
        }
        void update(const SplitDataType& split, int iIndexedFeatureSplitValue, int iFeature)
        {
            if(_mt)
            {
                _mt->lock();
                updateImpl(split, iIndexedFeatureSplitValue, iFeature);
                _mt->unlock();
            }
            else
                updateImpl(split, iIndexedFeatureSplitValue, iFeature);
        }

        void update(const SplitDataType& split, int iFeature, IndexType* bestSplitIdx, const IndexType* aIdx, size_t n)
        {
            if(_mt)
            {
                _mt->lock();
                if(updateImpl(split, -1, iFeature))
                    services::internal::tmemcpy<IndexType, cpu>(bestSplitIdx, aIdx, n);
                _mt->unlock();
            }
            else
            {
                if(updateImpl(split, -1, iFeature))
                    services::internal::tmemcpy<IndexType, cpu>(bestSplitIdx, aIdx, n);
            }
        }

        int iIndexedFeatureSplitValue() const { return _iIndexedFeatureSplitValue; }
        int iFeature() const { return _iFeature; }
        bool isThreadedMode() const { return _mt != nullptr; }

    private:
        algorithmFPType impurityDecrease() const { return _split.impurityDecrease; }
        bool updateImpl(const SplitDataType& split, int iIndexedFeatureSplitValue, int iFeature)
        {
            if(split.impurityDecrease < impurityDecrease())
                return false;

            if(split.impurityDecrease == impurityDecrease())
            {
                if(_iFeature < (int)iFeature) //deterministic way, let the split be the same as in sequential case
                    return false;
            }
            _iFeature = (int)iFeature;
            split.copyTo(_split);
            _iIndexedFeatureSplitValue = iIndexedFeatureSplitValue;
            return true;
        }

    private:
        SplitDataType& _split;
        Mutex* _mt;
        volatile int _iIndexedFeatureSplitValue;
        volatile int _iFeature;
    };

    TreeBuilder(CommonCtx& ctx) : _ctx(ctx){}
    ~TreeBuilder()
    {
        delete _memHelper;
        delete _taskGroup;
    }

    bool isInitialized() const { return !!_aBestSplitIdxBuf.get(); }
    virtual services::Status run(gbt::internal::GbtDecisionTree*& pRes, HomogenNumericTable<double>*& pTblImp,
        HomogenNumericTable<int>*& pTblSmplCnt, size_t iTree) DAAL_C11_OVERRIDE;
    virtual services::Status init() DAAL_C11_OVERRIDE
    {
        _aBestSplitIdxBuf.reset(_ctx.nSamples());
        _aSample.reset(_ctx.nSamples());
        DAAL_CHECK_MALLOC(_aBestSplitIdxBuf.get() && _aSample.get());
        DAAL_CHECK_MALLOC(initMemHelper());
        if(_ctx.isParallelNodes() && !_taskGroup)
            DAAL_CHECK_MALLOC((_taskGroup = new daal::task_group()));
        return services::Status();
    }
    daal::task_group* taskGroup() { return _taskGroup; }
    static TreeBuilder<algorithmFPType, cpu>* create(CommonCtx& ctx);

protected:
    bool initMemHelper();
    //find features to check in the current split node
    const IndexType* chooseFeatures()
    {
        if(_ctx.nFeatures() == _ctx.nFeaturesPerNode())
            return nullptr;
        IndexType* featureSample = _memHelper->getFeatureSampleBuf();
        _ctx.chooseFeatures(featureSample);
        return featureSample;
    }
    void buildNode(size_t iStart, size_t n, size_t level, const ImpurityType& imp, NodeType::Base*& res);
    IndexType* bestSplitIdxBuf() const { return _aBestSplitIdxBuf.get(); }
    NodeType::Base* buildRoot(size_t iTree)
    {
        _iTree = iTree;
        const size_t nSamples = _ctx.nSamples();
        auto aSample = _aSample.get();
        if(_ctx.isBagging() && !_ctx.isIndirect())
        {
            //aSample contains indices of rows
            const auto aSampleToF = _ctx.aSampleToF();
            for(size_t i = 0; i < nSamples; ++i)
                aSample[i] = aSampleToF[i];
        }
        else
        {
            for(size_t i = 0; i < nSamples; ++i)
                aSample[i] = i;
        }

        ImpurityType imp;
        getInitialImpurity(imp);
        typename NodeType::Base* res = buildLeaf(0, nSamples, 0, imp);
        if(res)
            return res;
        SplitJobType job(0, nSamples, 0, imp, res);
        buildSplit(job);
        if(taskGroup())
            taskGroup()->wait();
        return res;
    }

    void getInitialImpurity(ImpurityType& val)
    {
        const size_t nSamples = _ctx.nSamples();
        const ghType* pgh = _ctx.grad(this->_iTree);
        auto& G = val.g;
        auto& H = val.h;
        G = H = 0;
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nSamples; ++i)
        {
            G += pgh[i].g;
            H += pgh[i].h;
        }
    }

    void calcImpurityIndirect(const IndexType* aIdx, size_t n, ImpurityType& imp) const
    {
        DAAL_ASSERT(n);
        const ghType* pgh = _ctx.grad(this->_iTree);
        imp = pgh[aIdx[0]];
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 1; i < n; ++i)
        {
            imp.g += pgh[aIdx[i]].g;
            imp.h += pgh[aIdx[i]].h;
        }
    }

    void calcImpurity(size_t iStart, size_t n, ImpurityType& imp) const
    {
        if(_ctx.isIndirect())
        {
            this->calcImpurityIndirect(_aSample.get() + iStart, n, imp);
        }
        else
        {
            DAAL_ASSERT(n);
            const ghType* pgh = _ctx.grad(this->_iTree) + iStart;
            imp = pgh[0];
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 1; i < n; ++i)
            {
                imp.g += pgh[i].g;
                imp.h += pgh[i].h;
            }
        }
    }

    NodeType::Base* buildLeaf(size_t iStart, size_t n, size_t level, const ImpurityType& imp)
    {
        return _ctx.terminateCriteria(n, level, imp) ? makeLeaf(_aSample.get() + iStart, n, imp) : nullptr;
    }

    typename NodeType::Leaf* makeLeaf(const IndexType* idx, size_t n, const ImpurityType& imp)
    {
        typename NodeType::Leaf* pNode = nullptr;
        if(_ctx.isThreaded())
        {
            _mtAlloc.lock();
            pNode = _tree.allocator().allocLeaf();
            _mtAlloc.unlock();
        }
        else
            pNode = _tree.allocator().allocLeaf();
        pNode->response = _ctx.computeLeafWeightUpdateF(idx, n, imp, _iTree);
        pNode->count = n;
        pNode->impurity = imp.value(_ctx.par().lambda);
        return pNode;
    }
    typename NodeType::Split* makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered);
    void buildSplit(SplitJobType& job);
    bool findBestSplit(SplitJobType& job, SplitDataType& split, IndexType& iFeature);
    void findBestSplitImpl(const SplitJobType& job, SplitDataType& split, IndexType& iFeature, int& idxFeatureValueBestSplit, bool& bCopyToIdx);
    virtual void findSplitOneFeature(const IndexType* featureSample, size_t iFeatureInSample, const SplitJobType& job, BestSplit& bestSplit) = 0;
    bool simpleSplit(SplitJobType& job, SplitDataType& split, IndexType& iFeature);
    virtual void finalizeBestSplit(size_t n, size_t iStart, SplitDataType& bestSplit, IndexType iFeature, size_t idxFeatureValueBestSplit) = 0;

protected:
    CommonCtx& _ctx;
    size_t _iTree = 0;
    TreeType _tree;
    daal::Mutex _mtAlloc;
    typedef dtrees::internal::TVector<IndexType, cpu> IndexTypeArray;
    mutable IndexTypeArray _aBestSplitIdxBuf;
    mutable IndexTypeArray _aSample;
    MemHelperType* _memHelper = nullptr;
    daal::task_group* _taskGroup = nullptr;
};

template<typename algorithmFPType, CpuType cpu>
class TreeBuilderIndexed : public TreeBuilder<algorithmFPType, cpu>
{
public:
    typedef TreeBuilder<algorithmFPType, cpu> super;
    TreeBuilderIndexed(typename super::CommonCtx& ctx) : super(ctx){}
    virtual void findSplitOneFeature(const IndexType* featureSample, size_t iFeatureInSample,
        const typename super::SplitJobType& job, typename super::BestSplit& bestSplit) DAAL_C11_OVERRIDE;

protected:
    int findBestSplitImp(IndexType iFeature, const IndexType* aIdx,
        const typename super::SplitJobType& job, typename super::SplitDataType& split, bool bUpdateWhenTie) const;
    void computeGHSums(IndexType iFeature, const IndexType* aIdx,
        const typename super::SplitJobType& job, typename super::ghSumType* aGHSum, algorithmFPType& gTotal, algorithmFPType& hTotal) const;
    virtual void finalizeBestSplit(size_t n, size_t iStart, typename super::SplitDataType& bestSplit,
        IndexType iFeature, size_t idxFeatureValueBestSplit) DAAL_C11_OVERRIDE;
    static int doPartitionDirect(size_t n, const IndexType* aIdx,
        const typename IndexedFeatures::IndexType* indexedFeature, bool featureUnordered,
        IndexType idxFeatureValueBestSplit,
        IndexType* bestSplitIdx,
        typename super::ghType* pGH,
        size_t nLeft);

    int doPartition(size_t n, size_t iStart, typename super::SplitDataType& split, IndexType iFeature, size_t idxFeatureValueBestSplit)
    {
        if(_ctx.isIndirect())
            return dtrees::training::internal::doPartitionIdx<IndexType, typename IndexedFeatures::IndexType, size_t, cpu>(n,
            _aSample.get() + iStart, _ctx.aSampleToF(),
            _ctx.dataHelper().indexedFeatures().data(iFeature), split.featureUnordered,
            idxFeatureValueBestSplit,
            this->bestSplitIdxBuf() + iStart + split.nLeft,
            this->bestSplitIdxBuf() + iStart,
            split.nLeft);

        return doPartitionDirect(n, _aSample.get() + iStart,
            _ctx.dataHelper().indexedFeatures().data(iFeature), split.featureUnordered,
            idxFeatureValueBestSplit,
            this->bestSplitIdxBuf() + iStart,
            _ctx.grad(this->_iTree) + iStart,
            split.nLeft);
    }
    using super::_ctx;
    using super::_aSample;
    using super::_memHelper;
};

template<typename algorithmFPType, CpuType cpu>
class TreeBuilderSorted : public TreeBuilder<algorithmFPType, cpu>
{
public:
    typedef TreeBuilder<algorithmFPType, cpu> super;
    TreeBuilderSorted(typename super::CommonCtx& ctx) : super(ctx){}
    virtual void findSplitOneFeature(const IndexType* featureSample, size_t iFeatureInSample,
        const typename super::SplitJobType& job, typename super::BestSplit& bestSplit) DAAL_C11_OVERRIDE;

protected:
    bool findBestSplitFeatSorted(const algorithmFPType* featureVal, const IndexType* aIdx,
        const typename super::SplitJobType& job, typename super::SplitDataType& split, bool bUpdateWhenTie) const
    {
        return split.featureUnordered ? findBestSplitCategorical(featureVal, aIdx, job, split, bUpdateWhenTie) :
            findBestSplitOrdered(featureVal, aIdx, job, split, bUpdateWhenTie);
    }

    bool findBestSplitOrdered(const algorithmFPType* featureVal,
        const IndexType* aIdx, const typename super::SplitJobType& job,
        typename super::SplitDataType& split, bool bUpdateWhenTie) const;

    bool findBestSplitCategorical(const algorithmFPType* featureVal,
        const IndexType* aIdx, const typename super::SplitJobType& job,
        typename super::SplitDataType& split, bool bUpdateWhenTie) const;

    virtual void finalizeBestSplit(size_t n, size_t iStart, typename super::SplitDataType& bestSplit,
        IndexType iFeature, size_t idxFeatureValueBestSplit) DAAL_C11_OVERRIDE
    {}
    using super::_ctx;
    using super::_memHelper;
    using super::_aSample;
};

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilder<algorithmFPType, cpu>::initMemHelper()
{
    auto featuresSampleBufSize = 0; //do not allocate if not required
    const auto nFeat = _ctx.nFeatures();
    if(nFeat != _ctx.nFeaturesPerNode())
    {
        if(_ctx.nFeaturesPerNode()*_ctx.nFeaturesPerNode() < 2 * nFeat)
            featuresSampleBufSize = 2 * _ctx.nFeaturesPerNode();
        else
            featuresSampleBufSize = nFeat;
    }
    if(_ctx.isThreaded())
        _memHelper = new MemHelperThr<algorithmFPType, cpu>(featuresSampleBufSize);
    else
        _memHelper = new MemHelperSeq<algorithmFPType, cpu>(featuresSampleBufSize,
        _ctx.par().memorySavingMode ? 0 : _ctx.dataHelper().indexedFeatures().maxNumIndices(),
        _ctx.nSamples()); //TODO
    return _memHelper && _memHelper->init();
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilder<algorithmFPType, cpu>::buildNode(
    size_t iStart, size_t n, size_t level, const ImpurityType& imp, NodeType::Base*&res)
{
    if(taskGroup())
    {
        SplitTask job(*this, iStart, n, level, imp, res);
        taskGroup()->run(job);
    }
    else
    {
        SplitJobType job(iStart, n, level, imp, res);
        buildSplit(job);
    }
}

template <typename algorithmFPType, CpuType cpu>
typename TreeBuilder<algorithmFPType, cpu>::NodeType::Split*
TreeBuilder<algorithmFPType, cpu>::makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered)
{
    typename NodeType::Split* pNode = nullptr;
    if(_ctx.isThreaded())
    {
        _mtAlloc.lock();
        pNode = _tree.allocator().allocSplit();
        _mtAlloc.unlock();
    }
    else
        pNode = _tree.allocator().allocSplit();
    pNode->set(iFeature, featureValue, bUnordered);
    return pNode;
}

template <typename algorithmFPType, CpuType cpu>
services::Status TreeBuilder<algorithmFPType, cpu>::run(gbt::internal::GbtDecisionTree*& pRes,
    HomogenNumericTable<double>*& pTblImp, HomogenNumericTable<int>*& pTblSmplCnt, size_t iTree)
{
    _tree.destroy();
    typename NodeType::Base* nd = buildRoot(iTree);
    DAAL_CHECK_MALLOC(nd);
    _tree.reset(nd, false); //bUnorderedFeaturesUsed - TODO?
    gbt::internal::ModelImpl::treeToTable(_tree, &pRes, &pTblImp, &pTblSmplCnt);
    if(_ctx.isBagging() && _tree.top())
        _ctx.updateOOB(iTree, _tree);
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilder<algorithmFPType, cpu>::buildSplit(SplitJobType& job)
{
    SplitDataType split;
    IndexType iFeature;
    if(findBestSplit(job, split, iFeature))
    {
        typename NodeType::Split* res = makeSplit(iFeature, split.featureValue, split.featureUnordered);
        if(res)
        {
            job.res = res;

            res->kid[0] = buildLeaf(job.iStart, split.nLeft, job.level + 1, split.left);
            ImpurityType impRight; //statistics for the right part of the split
            //actually it is equal to job.imp - split.left, but 'imp' contains roundoff errors.
            //calculate right part directly to avoid propagation of the errors
            calcImpurity(job.iStart + split.nLeft, job.n - split.nLeft, impRight);
            res->kid[1] = buildLeaf(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, impRight);

            res->count = job.n;
            res->impurity = job.imp.value(_ctx.par().lambda);

            if(res->kid[0])
            {
                if(res->kid[1])
                    return; //all done
                SplitJobType right(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, impRight, res->kid[1]);
                buildSplit(right); //by this thread, no new job
            }
            else if(res->kid[1])
            {
                SplitJobType left(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                buildSplit(left); //by this thread, no new job
            }
            else
            {
                //one kid can be a new job, the left one, if there are available threads
                if(_ctx.numAvailableThreads())
                    buildNode(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                else
                {
                    SplitJobType left(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                    buildSplit(left); //by this thread, no new job
                }
                //and another kid is processed in the same thread
                SplitJobType right(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, impRight, res->kid[1]);
                buildSplit(right); //by this thread, no new job
            }
            return;
        }
    }
    job.res = makeLeaf(_aSample.get() + job.iStart, job.n, job.imp);
}

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilder<algorithmFPType, cpu>::findBestSplit(SplitJobType& job,
    SplitDataType& bestSplit, IndexType& iFeature)
{
    if(job.n == 2)
    {
        DAAL_ASSERT(_ctx.par().minObservationsInLeafNode == 1);
        return simpleSplit(job, bestSplit, iFeature);
    }

    int idxFeatureValueBestSplit = -1; //when sorted feature is used
    bool bCopyToIdx = true;
    findBestSplitImpl(job, bestSplit, iFeature, idxFeatureValueBestSplit, bCopyToIdx);
    if(iFeature < 0)
        return false;
    IndexType* bestSplitIdx = bestSplitIdxBuf() + job.iStart;
    IndexType* aIdx = _aSample.get() + job.iStart;
    if(idxFeatureValueBestSplit >= 0)
    {
        //indexed feature was used
        //calculate impurity (??) and get split to bestSplitIdx
        finalizeBestSplit(job.n, job.iStart, bestSplit, iFeature, idxFeatureValueBestSplit);
        bCopyToIdx = true;
    }
    else if(bestSplit.featureUnordered)
    {
        if(bestSplit.iStart)
        {
            DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= job.n);
            services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx + bestSplit.iStart, bestSplit.nLeft);
            aIdx += bestSplit.nLeft;
            services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, bestSplit.iStart);
            aIdx += bestSplit.iStart;
            bestSplitIdx += bestSplit.iStart + bestSplit.nLeft;
            if(job.n > (bestSplit.iStart + bestSplit.nLeft))
                services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, job.n - bestSplit.iStart - bestSplit.nLeft);
            bCopyToIdx = false;//done
        }
    }
    if(bCopyToIdx)
        services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, job.n);
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilder<algorithmFPType, cpu>::simpleSplit(SplitJobType& job, SplitDataType& split, IndexType& iFeature)
{
    if(_ctx.isIndirect())
    {
        algorithmFPType featBuf[2];
        IndexType* aIdx = _aSample.get() + job.iStart;
        for(size_t i = 0; i < _ctx.nFeatures(); ++i)
        {
            _ctx.featureValuesToBuf(i, featBuf, aIdx, 2);
            if(featBuf[1] - featBuf[0] <= _ctx.accuracy()) //all values of the feature are the same
                continue;
            split.featureValue = featBuf[0];
            split.nLeft = 1;
            split.iStart = 0;
            const auto pgh = _ctx.grad(_iTree);
            split.left.reset(pgh[*aIdx].g, pgh[*aIdx].h);
            split.impurityDecrease = job.imp.value(_ctx.par().lambda);
            iFeature = IndexType(i);
            return true;
        }
    }
    else
    {
        IndexType* aIdx = _aSample.get() + job.iStart;
        for(size_t i = 0; i < _ctx.nFeatures(); ++i)
        {
            const IndexedFeatures::IndexType* indexedFeature = _ctx.dataHelper().indexedFeatures().data(i);
            const IndexType iRow1 = aIdx[0];
            const IndexType iRow2 = aIdx[1];
            if(indexedFeature[iRow1] == indexedFeature[iRow2])
                continue;
            split.featureValue = _ctx.dataHelper().getValue(i, iRow1);
            split.nLeft = 1;
            split.iStart = 0;
            const auto pgh = _ctx.grad(this->_iTree) + job.iStart;
            split.left.reset(pgh[0].g, pgh[0].h);
            split.impurityDecrease = job.imp.value(_ctx.par().lambda);
            iFeature = IndexType(i);
            return true;
        }
    }
    return false;
}

//partition given set of indices into the left and right parts
//corresponding to the split feature value (cut value)
//given as the index in the sorted feature values array
//returns index of the row in the dataset corresponding to the split feature value (cut value)
template <typename algorithmFPType, CpuType cpu>
int TreeBuilderIndexed<algorithmFPType, cpu>::doPartitionDirect(size_t n, const IndexType* aIdx,
    const IndexedFeatures::IndexType* indexedFeature, bool featureUnordered,
    IndexType idxFeatureValueBestSplit,
    IndexType* bestSplitIdx,
    typename super::ghType* pGH,
    size_t nLeft)
{
    size_t iLeft = 0;
    size_t iRight = 0;
    int iRowSplitVal = -1;

    IndexType* bestSplitIdxRight = bestSplitIdx + nLeft;
    dtrees::internal::TVector<typename super::ghType, cpu, ScalableAllocator<cpu>> aTmp(n);
    typename super::ghType* pGHLeft = aTmp.get();
    typename super::ghType* pGHRight = pGHLeft + nLeft;
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(size_t i = 0; i < n; ++i)
    {
        const IndexType iRow = aIdx[i];
        const IndexedFeatures::IndexType idx = indexedFeature[iRow];

        if((featureUnordered && (idx != idxFeatureValueBestSplit)) || ((!featureUnordered) && (idx > idxFeatureValueBestSplit)))
        {
            bestSplitIdxRight[iRight] = iRow;
            pGHRight[iRight++] = pGH[i];
        }
        else
        {
            if(idx == idxFeatureValueBestSplit)
                iRowSplitVal = iRow;
            bestSplitIdx[iLeft] = iRow;
            pGHLeft[iLeft++] = pGH[i];
        }
    }
    DAAL_ASSERT(iRight == n - nLeft);
    DAAL_ASSERT(iLeft == nLeft);
    services::internal::tmemcpy<typename super::ghType, cpu>(pGH, aTmp.get(), n);
    return iRowSplitVal;
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilderIndexed<algorithmFPType, cpu>::finalizeBestSplit(size_t n, size_t iStart,
    typename super::SplitDataType& split, IndexType iFeature, size_t idxFeatureValueBestSplit)
{
    DAAL_ASSERT(split.nLeft > 0);
    const int iRowSplitVal = doPartition(n, iStart, split, iFeature, idxFeatureValueBestSplit);
    DAAL_ASSERT(iRowSplitVal >= 0);
    split.iStart = 0;
    if(_ctx.dataHelper().indexedFeatures().isBinned(iFeature))
        split.featureValue = (algorithmFPType)_ctx.dataHelper().indexedFeatures().binRightBorder(iFeature, idxFeatureValueBestSplit);
    else
        split.featureValue = _ctx.dataHelper().getValue(iFeature, iRowSplitVal);
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilderIndexed<algorithmFPType, cpu>::findSplitOneFeature(
    const IndexType* featureSample, size_t iFeatureInSample, const typename super::SplitJobType& job, typename super::BestSplit& bestSplit)
{
    const IndexType iFeature = featureSample ? featureSample[iFeatureInSample] : (IndexType)iFeatureInSample;
    const IndexType* aIdx = _aSample.get() + job.iStart;
    if(!_ctx.dataHelper().hasDiffFeatureValues(iFeature, aIdx, job.n))
        return;//all values of the feature are the same
    //use best split estimation when searching on iFeature
    algorithmFPType bestImpDec;
    int iBestFeat;
    bestSplit.safeGetData(bestImpDec, iBestFeat);
    typename super::SplitDataType split(bestImpDec, _ctx.featTypes().isUnordered(iFeature));
    //index of best feature value in the array of sorted feature values
    const int idxFeatureValue = findBestSplitImp(iFeature, aIdx, job, split, iBestFeat < 0 || iBestFeat > iFeature);
    if(idxFeatureValue < 0)
        return;
    bestSplit.update(split, idxFeatureValue, iFeature);
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilderSorted<algorithmFPType, cpu>::findSplitOneFeature(
    const IndexType* featureSample, size_t iFeatureInSample, const typename super::SplitJobType& job, typename super::BestSplit& bestSplit)
{
    const IndexType iFeature = featureSample ? featureSample[iFeatureInSample] : (IndexType)iFeatureInSample;
    IndexType* aIdx = _aSample.get() + job.iStart;
    const bool bThreaded = bestSplit.isThreadedMode();
    IndexType* bestSplitIdx = this->bestSplitIdxBuf() + job.iStart;
    auto aFeatBuf = _memHelper->getFeatureValueBuf(job.n); //TODO?
    typename super::MemHelperType::IndexTypeVector* aFeatIdxBuf = nullptr;
    if(bThreaded)
    {
        //get a local index, since it is used by parallel threads
        aFeatIdxBuf = _memHelper->getSortedFeatureIdxBuf(job.n);
        services::internal::tmemcpy<IndexType, cpu>(aFeatIdxBuf->get(), aIdx, job.n);
        aIdx = aFeatIdxBuf->get();
    }
    algorithmFPType* featBuf = aFeatBuf->get();
    _ctx.featureValuesToBuf(iFeature, featBuf, aIdx, job.n);
    if(featBuf[job.n - 1] - featBuf[0] <= _ctx.accuracy()) //all values of the feature are the same
    {
        _memHelper->releaseFeatureValueBuf(aFeatBuf);
        if(aFeatIdxBuf)
            _memHelper->releaseSortedFeatureIdxBuf(aFeatIdxBuf);
        return;
    }
    //use best split estimation when searching on iFeature
    algorithmFPType bestImpDec;
    int iBestFeat;
    bestSplit.safeGetData(bestImpDec, iBestFeat);
    typename super::SplitDataType split(bestImpDec, _ctx.featTypes().isUnordered(iFeature));
    bool bFound = findBestSplitFeatSorted(featBuf, aIdx, job, split, iBestFeat < 0 || iBestFeat > iFeature);
    _memHelper->releaseFeatureValueBuf(aFeatBuf);
    if(bFound)
    {
        DAAL_ASSERT(split.iStart < job.n);
        DAAL_ASSERT(split.iStart + split.nLeft <= job.n);
        if(split.featureUnordered || bThreaded ||
            (featureSample ? (iFeature != featureSample[_ctx.nFeaturesPerNode() - 1]) : (iFeature + 1 < _ctx.nFeaturesPerNode()))) //not a last feature
            bestSplit.update(split, iFeature, bestSplitIdx, aIdx, job.n);
        else
            bestSplit.update(split, -1, iFeature);
    }
    if(aFeatIdxBuf)
        _memHelper->releaseSortedFeatureIdxBuf(aFeatIdxBuf);
}

template <typename algorithmFPType, CpuType cpu>
void TreeBuilder<algorithmFPType, cpu>::findBestSplitImpl(const SplitJobType& job,
    SplitDataType& split, IndexType& iFeature, int& idxFeatureValueBestSplit, bool& bCopyToIdx)
{
    const IndexType* featureSample = chooseFeatures();
    iFeature = -1;
    bCopyToIdx = true;
    if(_ctx.isParallelFeatures() && _ctx.numAvailableThreads())
    {
        daal::Mutex mtBestSplit;
        BestSplit bestSplit(split, &mtBestSplit);
        daal::threader_for(_ctx.nFeaturesPerNode(), _ctx.nFeaturesPerNode(), [&](size_t i)
        {
            findSplitOneFeature(featureSample, i, job, bestSplit);
        });
        idxFeatureValueBestSplit = bestSplit.iIndexedFeatureSplitValue();
        iFeature = bestSplit.iFeature();
    }
    else
    {
        BestSplit bestSplit(split, nullptr);
        for(size_t i = 0; i < _ctx.nFeaturesPerNode(); ++i)
        {
            findSplitOneFeature(featureSample, i, job, bestSplit);
        }
        idxFeatureValueBestSplit = bestSplit.iIndexedFeatureSplitValue();
        iFeature = bestSplit.iFeature();
        if((iFeature >= 0) && (idxFeatureValueBestSplit < 0) && !split.featureUnordered)
        {
            //in sequential mode, if iBestSplit is the last considered feature then aIdx already contains the best split, no need to copy
            if(featureSample ? (iFeature == featureSample[_ctx.nFeaturesPerNode() - 1]) : (iFeature + 1 == _ctx.nFeaturesPerNode())) //last feature
                bCopyToIdx = false;
        }
    }
    if(featureSample)
        _memHelper->releaseFeatureSampleBuf(const_cast<IndexType*>(featureSample));

    if(iFeature < 0)
        return; //not found
    //now calculate full impurity decrease
    split.impurityDecrease -= job.imp.value(_ctx.par().lambda);
    if(split.impurityDecrease < _ctx.par().minSplitLoss)
        iFeature = -1; //not found
}

template<typename algorithmFPType, CpuType cpu>
void TreeBuilderIndexed<algorithmFPType, cpu>::computeGHSums(IndexType iFeature, const IndexType* aIdx,
    const typename super::SplitJobType& job, typename super::ghSumType* aGHSum, algorithmFPType& gTotal, algorithmFPType& hTotal) const
{
    const size_t n = job.n;
    const IndexedFeatures::IndexType* indexedFeature = _ctx.dataHelper().indexedFeatures().data(iFeature);
    if(!_ctx.isIndirect())
    {
        const typename super::ghType* pgh = _ctx.grad(this->_iTree) + job.iStart;
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < n; ++i)
        {
            const IndexType iRow = aIdx[i];
            const typename IndexedFeatures::IndexType idx = indexedFeature[iRow];
            auto& sum = aGHSum[idx];
            sum.n++;
            sum.g += pgh[i].g;
            sum.h += pgh[i].h;
            gTotal += pgh[i].g;
            hTotal += pgh[i].h;
        }
        return;
    }
    const typename super::ghType* pgh = _ctx.grad(this->_iTree);
    if(_ctx.aSampleToF())
    {
        const IndexType* aSampleToF = _ctx.aSampleToF();
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const IndexType iRow = aSampleToF[iSample];
            const typename IndexedFeatures::IndexType idx = indexedFeature[iRow];
            auto& sum = aGHSum[idx];
            sum.n++;
            sum.g += pgh[iSample].g;
            sum.h += pgh[iSample].h;
            gTotal += pgh[iSample].g;
            hTotal += pgh[iSample].h;
        }
    }
    else
    {
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const typename IndexedFeatures::IndexType idx = indexedFeature[iSample];
            auto& sum = aGHSum[idx];
            sum.n++;
            sum.g += pgh[iSample].g;
            sum.h += pgh[iSample].h;
            gTotal += pgh[iSample].g;
            hTotal += pgh[iSample].h;
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
int TreeBuilderIndexed<algorithmFPType, cpu>::findBestSplitImp(IndexType iFeature, const IndexType* aIdx,
    const typename super::SplitJobType& job, typename super::SplitDataType& split, bool bUpdateWhenTie) const
{
    const size_t nDiffFeatMax = _ctx.dataHelper().indexedFeatures().numIndices(iFeature);
    auto aGHSum = _memHelper->getGHSumBuf(nDiffFeatMax); //sums of gradients per each value of the indexed feature
    DAAL_ASSERT(aGHSum); //TODO: return status
    if(!aGHSum)
        return -1;

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(size_t i = 0; i < nDiffFeatMax; ++i)
    {
        aGHSum[i].n = 0;
        aGHSum[i].g = algorithmFPType(0);
        aGHSum[i].h = algorithmFPType(0);
    }

    algorithmFPType gTotal = 0; //total sum of g in the set being split
    algorithmFPType hTotal = 0; //total sum of h in the set being split
    computeGHSums(iFeature, aIdx, job, aGHSum, gTotal, hTotal);

    const size_t n = job.n;
    //make a copy since it can be corrected below. TODO: propagate this corrected value to the argument?
    typename super::ImpurityType imp(gTotal, hTotal);
    //index of best feature value in the array of sorted feature values
    int idxFeatureBestSplit = -1;
    //below we calculate only part of the impurity decrease dependent on split itself
    algorithmFPType bestImpDecrease = split.impurityDecrease;
    if(!split.featureUnordered)
    {
        size_t nLeft = 0;
        typename super::ImpurityType left;
        for(size_t i = 0; i < nDiffFeatMax; ++i)
        {
            if(!aGHSum[i].n)
                continue;
            nLeft += aGHSum[i].n;
            if((n - nLeft) < _ctx.par().minObservationsInLeafNode)
                break;
            left.add((const typename super::ghType&)aGHSum[i]);
            if(nLeft < _ctx.par().minObservationsInLeafNode)
                continue;

            typename super::ImpurityType right(imp, left);
            //the part of the impurity decrease dependent on split itself
            const algorithmFPType impDecrease = left.value(_ctx.par().lambda) + right.value(_ctx.par().lambda);
            if((impDecrease > bestImpDecrease) || (bUpdateWhenTie && (impDecrease == bestImpDecrease)))
            {
                split.left = left;
                split.nLeft = nLeft;
                idxFeatureBestSplit = i;
                bestImpDecrease = impDecrease;
            }
        }
        if(idxFeatureBestSplit >= 0)
            split.impurityDecrease = bestImpDecrease;
    }
    else
    {
        for(size_t i = 0; i < nDiffFeatMax; ++i)
        {
            if((aGHSum[i].n < _ctx.par().minObservationsInLeafNode) || ((n - aGHSum[i].n) < _ctx.par().minObservationsInLeafNode))
                continue;
            const typename super::ImpurityType& left = aGHSum[i];
            typename super::ImpurityType right(imp, left);
            //the part of the impurity decrease dependent on split itself
            const algorithmFPType impDecrease = left.value(_ctx.par().lambda) + right.value(_ctx.par().lambda);
            if(impDecrease > bestImpDecrease)
            {
                idxFeatureBestSplit = i;
                bestImpDecrease = impDecrease;
            }
        }
        if(idxFeatureBestSplit >= 0)
        {
            split.impurityDecrease = bestImpDecrease;
            split.nLeft = aGHSum[idxFeatureBestSplit].n;
            split.left = (const typename super::ghType&)aGHSum[idxFeatureBestSplit];
        }
    }
    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilderSorted<algorithmFPType, cpu>::findBestSplitOrdered(const algorithmFPType* featureVal,
    const IndexType* aIdx, const typename super::SplitJobType& job, typename super::SplitDataType& split, bool bUpdateWhenTie) const
{
    typename super::ImpurityType left(_ctx.grad(this->_iTree)[*aIdx]);
    algorithmFPType bestImpurityDecrease = split.impurityDecrease;
    IndexType iBest = -1;
    const size_t n = job.n;
    const auto nMinSplitPart = _ctx.par().minObservationsInLeafNode;
    const algorithmFPType last = featureVal[n - nMinSplitPart];
    for(size_t i = 1; i < (n - nMinSplitPart + 1); ++i)
    {
        const bool bSameFeaturePrev(featureVal[i] <= featureVal[i - 1] + _ctx.accuracy());
        if(!(bSameFeaturePrev || i < nMinSplitPart))
        {
            //can make a split
            //nLeft == i, nRight == n - i
            typename super::ImpurityType right(job.imp, left);
            const algorithmFPType v = left.value(_ctx.par().lambda) + right.value(_ctx.par().lambda);
            if((v > bestImpurityDecrease) || (bUpdateWhenTie && (v == bestImpurityDecrease)))
            {
                bestImpurityDecrease = v;
                split.left = left;
                iBest = i;
            }
        }

        //update impurity and continue
        left.add(_ctx.grad(this->_iTree)[aIdx[i]]);
    }
    if(iBest < 0)
        return false;

    split.impurityDecrease = bestImpurityDecrease;
    split.nLeft = iBest;
    split.iStart = 0;
    split.featureValue = featureVal[iBest - 1];
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool TreeBuilderSorted<algorithmFPType, cpu>::findBestSplitCategorical(const algorithmFPType* featureVal,
    const IndexType* aIdx, const typename super::SplitJobType& job, typename super::SplitDataType& split, bool bUpdateWhenTie) const
{
    const size_t n = job.n;
    const auto nMinSplitPart = _ctx.par().minObservationsInLeafNode;
    DAAL_ASSERT(n >= 2 * nMinSplitPart);
    algorithmFPType bestImpurityDecrease = split.impurityDecrease;
    typename super::ImpurityType left;
    bool bFound = false;
    size_t nDiffFeatureValues = 0;
    for(size_t i = 0; i < n - nMinSplitPart;)
    {
        ++nDiffFeatureValues;
        size_t count = 1;
        const algorithmFPType first = featureVal[i];
        const size_t iStart = i;
        for(++i; (i < n) && (featureVal[i] == first); ++count, ++i);
        if((count < nMinSplitPart) || ((n - count) < nMinSplitPart))
            continue;

        if((i == n) && (nDiffFeatureValues == 2) && bFound)
            break; //only 2 feature values, one possible split, already found

        this->calcImpurityIndirect(aIdx + iStart, count, left);
        typename super::ImpurityType right(job.imp, left);
        const algorithmFPType v = left.value(_ctx.par().lambda) + right.value(_ctx.par().lambda);
        if(v > bestImpurityDecrease || (bUpdateWhenTie && (v == bestImpurityDecrease)))
        {
            bestImpurityDecrease = v;
            split.left = left;
            split.nLeft = count;
            split.iStart = iStart;
            split.featureValue = first;
            bFound = true;
        }
    }
    if(bFound)
        split.impurityDecrease = bestImpurityDecrease;
    return bFound;
}

template <typename algorithmFPType, CpuType cpu>
TreeBuilder<algorithmFPType, cpu>* TreeBuilder<algorithmFPType, cpu>::create(CommonCtx& ctx)
{
    if(ctx.par().memorySavingMode)
        return new TreeBuilderSorted<algorithmFPType, cpu>(ctx);
    return new TreeBuilderIndexed<algorithmFPType, cpu>(ctx);
}

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

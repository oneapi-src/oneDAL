/* file: gbt_classification_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for gradient boosted trees classification
//  (defaultDense) method.
//--
*/

#ifndef __GBT_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __GBT_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "gbt_classification_train_kernel.h"
#include "gbt_classification_model_impl.h"
#include "gbt_train_dense_default_impl.i"
#include "gbt_train_tree_builder.i"
#include "service_error_handling.h"
#include "service_algo_utils.h"

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::training::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace training
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// Logistic loss function, L(y,f) = -[y*ln(sigmoid(f)) + (1 - y)*ln(1-sigmoid(f))]
// where sigmoid(f) = 1/(1 + exp(-f)
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class LogisticLoss : public LossFunction<algorithmFPType, cpu>
{
public:
    virtual void getGradients(size_t n, const algorithmFPType* y, const algorithmFPType* f,
        const IndexType* sampleInd,
        algorithmFPType* gh) DAAL_C11_OVERRIDE
    {
        TVector<algorithmFPType, cpu, ScalableAllocator<cpu>> aExp(n);
        auto exp = aExp.get();
        const algorithmFPType expThreshold = daal::internal::Math<algorithmFPType, cpu>::vExpThreshold();
        if(sampleInd)
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < n; ++i)
            {
                exp[i] = -f[sampleInd[i]];
                /* make all values less than threshold as threshold value
                to fix slow work on vExp on large negative inputs */
                if(exp[i] < expThreshold)
                    exp[i] = expThreshold;
            }
        }
        else
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < n; ++i)
            {
                exp[i] = -f[i];
                /* make all values less than threshold as threshold value
                to fix slow work on vExp on large negative inputs */
                if(exp[i] < expThreshold)
                    exp[i] = expThreshold;
            }
        }
        daal::internal::Math<algorithmFPType, cpu>::vExp(n, exp, exp);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < n; ++i)
        {
            const auto sigm = algorithmFPType(1.0) / (algorithmFPType(1.0) + exp[i]);
            gh[2 * i] = sigm - y[i]; //gradient
            gh[2 * i + 1] = sigm * (algorithmFPType(1.0) - sigm); //hessian
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Cross entropy loss function, L(y,f)=-sum(I(y=k)*ln(pk)) where pk = exp(fk)/sum(exp(f))
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class CrossEntropyLoss : public LossFunction<algorithmFPType, cpu>
{
public:
    CrossEntropyLoss(size_t numClasses) : _nClasses(numClasses){}
    virtual void getGradients(size_t n, const algorithmFPType* y, const algorithmFPType* f,
        const IndexType* sampleInd,
        algorithmFPType* gh) DAAL_C11_OVERRIDE
    {
        static const size_t s_cMaxClassesBufSize = 12;
        const bool bUseTLS(_nClasses > s_cMaxClassesBufSize);
        daal::TlsMem<algorithmFPType, cpu> lsData(_nClasses);
        daal::threader_for(n, n, [&](size_t i)
        {
            algorithmFPType buf[s_cMaxClassesBufSize];
            algorithmFPType* p = bUseTLS ? lsData.local() : buf;
            const size_t iSample = (sampleInd ? sampleInd[i] : i);
            getSoftmax(f + _nClasses*iSample, p);
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t k = 0; k < _nClasses; ++k)
            {
                const algorithmFPType pk = p[k];
                const algorithmFPType h = algorithmFPType(2.) * pk * (algorithmFPType(1.) - pk);
                algorithmFPType* gh_ik = gh + 2*(k*n + i);
                gh_ik[1] = h;
                if(size_t(y[i]) == k)
                    gh_ik[0] = (pk - algorithmFPType(1.));
                else
                    gh_ik[0] = pk;
            }
        });
    }

protected:
    void getSoftmax(const algorithmFPType* arg, algorithmFPType* res) const
    {
        const algorithmFPType expThreshold = daal::internal::Math<algorithmFPType, cpu>::vExpThreshold();
        algorithmFPType maxArg = arg[0];
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 1; i < _nClasses; ++i)
        {
            if(maxArg < arg[i])
                maxArg = arg[i];
        }
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < _nClasses; ++i)
        {
            res[i] = arg[i] - maxArg;
            /* make all values less than threshold as threshold value
            to fix slow work on vExp on large negative inputs */
            if(res[i] < expThreshold)
                res[i] = expThreshold;
        }
        daal::internal::Math<algorithmFPType, cpu>::vExp(_nClasses, res, res);
        algorithmFPType sum(0.);
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < _nClasses; ++i)
            sum += res[i];

        sum = algorithmFPType(1.) / sum;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < _nClasses; ++i)
            res[i] *= sum;
    }

protected:
    size_t _nClasses;
};


//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchTask for classification
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::classification::training::Method method, CpuType cpu>
class TrainBatchTask : public TrainBatchTaskBaseXBoost<algorithmFPType, cpu>
{
    typedef TrainBatchTaskBaseXBoost<algorithmFPType, cpu> super;
    typedef TreeBuilder<algorithmFPType, cpu> TreeBuilderType;
    typedef ls<TreeBuilderType*> lsType;

public:
    TrainBatchTask(HostAppIface* pHostApp, const NumericTable *x, const NumericTable *y,
        const gbt::training::Parameter& par,
        const dtrees::internal::FeatureTypes& featTypes,
        const dtrees::internal::IndexedFeatures* indexedFeatures,
        engines::internal::BatchBaseImpl& engine, size_t nClasses) :
        super(pHostApp, x, y, par, featTypes, indexedFeatures, engine, nClasses),
        _builder(nullptr), _ls(nullptr)
    {
    }

    ~TrainBatchTask()
    {
        delete _builder;
        if(_ls)
        {
            _ls->reduce([](TreeBuilderType* ptr)-> void
            {
                if(ptr)
                    delete ptr;
            });
            delete _ls;
        }
    }

    bool done() { return false; }
    virtual services::Status init() DAAL_C11_OVERRIDE
    {
        auto s = super::init();
        if(!s)
            return s;
        if(this->isParallelTrees())
        {
            _ls = new lsType([=]()->TreeBuilderType* { return TreeBuilderType::create(*this); });
            DAAL_CHECK_MALLOC(_ls);
            return s;
        }
        _builder = TreeBuilderType::create(*this);
        DAAL_CHECK_MALLOC(_builder);
        return _builder->init();
    }

protected:
    virtual void initLossFunc() DAAL_C11_OVERRIDE
    {
        switch(static_cast<const gbt::classification::training::Parameter&>(this->_par).loss)
        {
        case crossEntropy:
            if(this->_nClasses == 2)
                this->_loss = new LogisticLoss<algorithmFPType, cpu>();
            else
                this->_loss = new CrossEntropyLoss<algorithmFPType, cpu>(this->_nClasses);
            break;
        default:
            DAAL_ASSERT(false);
        }
    }

    virtual services::Status buildTrees(gbt::internal::GbtDecisionTree** aTbl, HomogenNumericTable<double>** aTblImp, HomogenNumericTable<int>** aTblSmplCnt) DAAL_C11_OVERRIDE
    {
        if(this->isParallelTrees())
        {
            this->_nParallelNodes.set(this->_nTrees); //highest level parallelization first
            daal::SafeStatus safeStat;
            daal::threader_for(this->_nTrees, this->_nTrees, [&](size_t i)
            {
                if(safeStat)
                    safeStat |= buildTreeThreadLocal(aTbl[i], aTblImp[i], aTblSmplCnt[i], i);
                else
                    return;
                this->_nParallelNodes.dec();//allow lower levels of parallelization
            });
            return safeStat.detach();
        }

        services::Status s;
        for(size_t i = 0; s.ok() && (i < this->_nTrees) && !daal::algorithms::internal::isCancelled(s, this->_hostApp); ++i)
        {
            DAAL_ASSERT(this->_nParallelNodes.get() == 0);
            this->_nParallelNodes.inc();
            s |= _builder->run(aTbl[i], aTblImp[i], aTblSmplCnt[i], i);
            this->_nParallelNodes.dec();
            DAAL_ASSERT(this->_nParallelNodes.get() == 0);
        }
        return s;
    }

    services::Status buildTreeThreadLocal(gbt::internal::GbtDecisionTree*& tbl, HomogenNumericTable<double>*& pTblImp,
        HomogenNumericTable<int>*& pTblSmplCnt, size_t iTree)
    {
        auto pBuilder = _ls->local();
        DAAL_CHECK_MALLOC(pBuilder);
        services::Status s;
        if((pBuilder->isInitialized() || (s = pBuilder->init()).ok()) && !algorithms::internal::isCancelled(s, this->_hostApp))
            s = pBuilder->run(tbl, pTblImp, pTblSmplCnt, iTree);
        _ls->release(pBuilder);
        if(s)
            algorithms::internal::isCancelled(s, this->_hostApp);
        return s;
    }

protected:
    TreeBuilder<algorithmFPType, cpu>* _builder;
    lsType* _ls;
};

//////////////////////////////////////////////////////////////////////////////////////////
// ClassificationTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::classification::training::Method method, CpuType cpu>
services::Status ClassificationTrainBatchKernel<algorithmFPType, method, cpu>::compute(
    HostAppIface* pHost, const NumericTable *x, const NumericTable *y, gbt::classification::Model& m, Result& res, const Parameter& par,
    engines::internal::BatchBaseImpl& engine)
{
    return computeImpl<algorithmFPType, cpu,
        TrainBatchTask<algorithmFPType, method, cpu> >
        (pHost, x, y, *static_cast<daal::algorithms::gbt::classification::internal::ModelImpl*>(&m), par, engine, par.nClasses);
}

} /* namespace internal */
} /* namespace training */
} /* namespace classification */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif

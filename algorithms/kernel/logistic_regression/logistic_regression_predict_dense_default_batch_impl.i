/* file: logistic_regression_predict_dense_default_batch_impl.i */
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
//  Common functions for logistic regression classification predictions calculation
//--
*/

#ifndef __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "logistic_regression_predict_kernel.h"
#include "threading.h"
#include "daal_defines.h"
#include "logistic_regression_model_impl.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_data_utils.h"
#include "service_algo_utils.h"
#include "service_environment.h"
#include "service_blas.h"
#include "objective_function/cross_entropy_loss/cross_entropy_loss_dense_default_batch_kernel.h"
#include "objective_function/logistic_loss/logistic_loss_dense_default_batch_kernel.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace internal
{

namespace ll = daal::algorithms::optimization_solver::logistic_loss;

//////////////////////////////////////////////////////////////////////////////////////////
// PredictBinaryClassificationTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictBinaryClassificationTask
{
public:
    PredictBinaryClassificationTask(const NumericTable *x, NumericTable *y,
        NumericTable *prob, NumericTable *logProb) :
        _data(x), _res(y), _prob(prob), _logProb(logProb){}
    services::Status run(const NumericTable& beta, services::HostAppIface* pHostApp)
    {
        NumericTable* pRaw = _prob ? _prob : (_logProb ? _logProb : _res);
        const auto nRowsTotal = pRaw->getNumberOfRows();
        WriteOnlyRows<algorithmFPType, cpu> rawBD(pRaw, 0, nRowsTotal);
        DAAL_CHECK_BLOCK_STATUS(rawBD);
        algorithmFPType* aRaw = rawBD.get();
        //compute raw values
        auto s = predictRaw(pHostApp, beta, aRaw);
        if(!s)
            return s;
        if(_prob || _logProb)
        {
            //before transforming raw values to sigmoid and logarithm, predict labels
            if(_res)
            {
                WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, nRowsTotal);
                DAAL_CHECK_BLOCK_STATUS(resBD);
                predictLabels(aRaw, resBD.get(), nRowsTotal);
            }
            ll::internal::LogLossKernel<algorithmFPType, ll::defaultDense, cpu>::sigmoid(aRaw, aRaw, nRowsTotal);
            if(_logProb)
            {
                if(_prob)
                {
                    WriteOnlyRows<algorithmFPType, cpu> logBD(_logProb, 0, nRowsTotal);
                    DAAL_CHECK_BLOCK_STATUS(logBD);
                    daal::internal::Math<algorithmFPType, cpu>::vLog(nRowsTotal, aRaw, logBD.get());
                }
                else
                    daal::internal::Math<algorithmFPType, cpu>::vLog(nRowsTotal, aRaw, aRaw);
            }
            return s;
        }
        //only labels are required
        DAAL_ASSERT(pRaw == _res);
        predictLabels(aRaw, aRaw, nRowsTotal);
        return s;
    }

protected:
    services::Status predictRaw(services::HostAppIface* pHostApp, const NumericTable& beta, algorithmFPType* pRes);
    void predictLabels(const algorithmFPType* pRaw, algorithmFPType* pRes, size_t nRows)
    {
        const algorithmFPType label[2] = { algorithmFPType(1.), algorithmFPType(0.) };
        //pRaw contains raw values of sigmoid argument
        for(size_t iRow = 0; iRow < nRows; ++iRow)
        {
            //probablity is a sigmoid(f) hence sign(f) can be checked
            pRes[iRow] = label[services::internal::SignBit<algorithmFPType, cpu>::get(pRaw[iRow])];
        }
    }

protected:
    const NumericTable* _data;
    NumericTable* _res;
    NumericTable* _prob;
    NumericTable* _logProb;
};

template <typename algorithmFPType, CpuType cpu>
services::Status PredictBinaryClassificationTask<algorithmFPType, cpu>::predictRaw(services::HostAppIface* pHostApp,
    const NumericTable& beta, algorithmFPType* pRes)
{
    const size_t nRowsTotal = _data->getNumberOfRows();
    const size_t nCols = _data->getNumberOfColumns();
    const size_t nYPerRow = 1;
    const size_t nRowsInBlockDefault = 500;

    const size_t nRowsInBlock = services::internal::getNumElementsFitInMemory(services::internal::getL1CacheSize()*0.8,
        (nCols + nYPerRow)*sizeof(algorithmFPType), nRowsInBlockDefault);
    const size_t nDataBlocks = nRowsTotal / nRowsInBlock + !!(nRowsTotal%nRowsInBlock);

    ReadRows<algorithmFPType, cpu> betaBD(const_cast<NumericTable&>(beta), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(betaBD);

    SafeStatus safeStat;
    HostAppHelper host(pHostApp, 1000);
    daal::threader_for(nDataBlocks, nDataBlocks, [&](size_t iBlock)
    {
        services::Status s;
        if(host.isCancelled(s, 1))
        {
            safeStat.add(s);
            return;
        }
        const size_t iStartRow = iBlock*nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == nDataBlocks - 1) ? nRowsTotal - iBlock * nRowsInBlock : nRowsInBlock;
        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        algorithmFPType* res = pRes + iStartRow;
        ll::internal::LogLossKernel<algorithmFPType, ll::defaultDense, cpu>::applyBeta(xBD.get(), betaBD.get(), res, nRowsToProcess, nCols, true);
    });
    return safeStat.detach();
}

//////////////////////////////////////////////////////////////////////////////////////////
// PredictMulticlassTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictMulticlassTask
{
public:
    PredictMulticlassTask(const NumericTable *x, NumericTable *y,
        NumericTable *prob, NumericTable *logProb) :
        _data(x), _res(y), _prob(prob), _logProb(logProb){}
    services::Status run(const NumericTable& beta, services::HostAppIface* pHostApp);

protected:
    void predictRaw(const algorithmFPType* x, const algorithmFPType* beta, algorithmFPType* rawRes,
        size_t nRows, size_t nClasses, size_t nCols);

protected:
    const NumericTable* _data;
    NumericTable* _res;
    NumericTable* _prob;
    NumericTable* _logProb;
};

template <typename algorithmFPType, CpuType cpu>
struct TlsData
{
    DAAL_NEW_DELETE();
    TlsData(size_t n, const NumericTable* ntX) : x(const_cast<NumericTable*>(ntX))
    {
        raw = services::internal::service_scalable_malloc<algorithmFPType, cpu>(n);
    }

    ~TlsData()
    {
        if(raw)
            services::internal::service_scalable_free<algorithmFPType, cpu>(raw);
    }

    ReadRows<algorithmFPType, cpu> x;
    WriteOnlyRows<algorithmFPType, cpu> tmp;
    algorithmFPType* raw = nullptr;
};

template <typename algorithmFPType, CpuType cpu>
services::Status PredictMulticlassTask<algorithmFPType, cpu>::run(const NumericTable& beta, services::HostAppIface* pHostApp)
{
    const size_t nRowsTotal = _data->getNumberOfRows();
    const size_t nCols = _data->getNumberOfColumns();
    const size_t nClasses = beta.getNumberOfRows();
    const size_t nBetaPerClass = beta.getNumberOfColumns();
    DAAL_ASSERT(nBetaPerClass == nCols + 1);
    const size_t nYPerRow = nClasses;
    const size_t nRowsInBlockDefault = 500;

    WriteOnlyRows<algorithmFPType, cpu> resBD;
    if(_res)
    {
        resBD.set(_res, 0, nRowsTotal);
        DAAL_CHECK_BLOCK_STATUS(resBD);
    }
    const size_t nRowsInBlock = services::internal::getNumElementsFitInMemory(services::internal::getL1CacheSize()*0.8,
        (nCols + nYPerRow)*sizeof(algorithmFPType), nRowsInBlockDefault);
    const size_t nDataBlocks = nRowsTotal / nRowsInBlock + !!(nRowsTotal%nRowsInBlock);

    ReadRows<algorithmFPType, cpu> betaBD(const_cast<NumericTable&>(beta), 0, nClasses);
    DAAL_CHECK_BLOCK_STATUS(betaBD);

    using TlsDataCpu = TlsData<algorithmFPType, cpu>;
    daal::tls<TlsDataCpu*> tlsData([=]()-> TlsDataCpu*
    {
        return new TlsDataCpu(nRowsInBlock*nClasses, _data);
    });

    SafeStatus safeStat;
    HostAppHelper host(pHostApp, 1000);
    daal::threader_for(nDataBlocks, nDataBlocks, [&](size_t iBlock)
    {
        services::Status s;
        if(host.isCancelled(s, 1))
        {
            safeStat.add(s);
            return;
        }
        const size_t iStartRow = iBlock*nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == nDataBlocks - 1) ? nRowsTotal - iBlock * nRowsInBlock : nRowsInBlock;
        TlsDataCpu* pLocal = tlsData.local();
        DAAL_CHECK_MALLOC_THR(pLocal);
        const algorithmFPType* pXBlock = pLocal->x.next(iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(pLocal->x);
        algorithmFPType* pRawValues = pLocal->raw;
        predictRaw(pXBlock, betaBD.get(), pRawValues, nRowsToProcess, nClasses, nCols);
        if(_res)
        {
            algorithmFPType* res = resBD.get() + iStartRow;
            for(size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
                res[iRow] = algorithmFPType(services::internal::getMaxElementIndex<algorithmFPType, cpu>(pRawValues + iRow*nClasses, nClasses));
        }
        namespace cel = daal::algorithms::optimization_solver::cross_entropy_loss;
        //softmax
        if(_prob && !_logProb)
        {
            //compute softmax and write results directly to _prob
            pLocal->tmp.set(_prob, iStartRow, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS_THR(pLocal->tmp);
            cel::internal::CrossEntropyLossKernel<algorithmFPType, cel::defaultDense, cpu>::softmax(pRawValues, pLocal->tmp.get(), nRowsToProcess, nClasses);
        }
        else if(_prob || _logProb)
        {
            //compute softmax in pRawValues
            cel::internal::CrossEntropyLossKernel<algorithmFPType, cel::defaultDense, cpu>::softmax(pRawValues, pRawValues, nRowsToProcess, nClasses);
            if(_prob)
            {
                pLocal->tmp.set(_prob, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(pLocal->tmp);
                services::internal::tmemcpy<algorithmFPType, cpu>(pLocal->tmp.get(), pRawValues, nRowsToProcess*nClasses);
            }
            if(_logProb)
            {
                pLocal->tmp.set(_logProb, iStartRow, nRowsToProcess);
                DAAL_CHECK_BLOCK_STATUS_THR(pLocal->tmp);
                daal::internal::Math<algorithmFPType, cpu>::vLog(nRowsToProcess*nClasses, pRawValues, pLocal->tmp.get());
            }
        }
    });
    tlsData.reduce([](TlsDataCpu* ptr){ delete ptr; });
    return safeStat.detach();
}

template<typename algorithmFPType, CpuType cpu>
void PredictMulticlassTask<algorithmFPType, cpu>::predictRaw(const algorithmFPType* x, const algorithmFPType* beta,
    algorithmFPType* rawRes, size_t nRows, size_t nClasses, size_t nCols)
{
    namespace cel = daal::algorithms::optimization_solver::cross_entropy_loss;
    cel::internal::CrossEntropyLossKernel<algorithmFPType, cel::defaultDense, cpu>::applyBeta(x, beta, rawRes, nRows, nClasses, nCols, true);
}

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface* pHostApp, const NumericTable *x,
    const logistic_regression::Model *m, size_t nClasses,
    NumericTable* pRes, NumericTable* pProbab, NumericTable* pLogProbab)
{
    const daal::algorithms::logistic_regression::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::logistic_regression::internal::ModelImpl*>(m);
    if(nClasses == 2)
    {
        PredictBinaryClassificationTask<algorithmFPType, cpu> task(x, pRes, pProbab, pLogProbab);
        return task.run(*pModel->getBeta(), pHostApp);
    }
    PredictMulticlassTask<algorithmFPType, cpu> task(x, pRes, pProbab, pLogProbab);
    return task.run(*pModel->getBeta(), pHostApp);
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace logistic_regression */
} /* namespace algorithms */
} /* namespace daal */

#endif

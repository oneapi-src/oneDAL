/* file: logistic_regression_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for logistic regression classification
//  (defaultDense) method.
//--
*/

#ifndef __LOGISTIC_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __LOGISTIC_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "logistic_regression_train_kernel.h"
#include "logistic_regression_model_impl.h"
#include "service_error_handling.h"
#include "service_algo_utils.h"
#include "algorithms/optimization_solver/objective_function/logistic_loss_batch.h"
#include "algorithms/optimization_solver/objective_function/cross_entropy_loss_batch.h"
#include "service_numeric_table.h"
#include "service_math.h"

using namespace daal::algorithms::logistic_regression::training::internal;
using namespace daal::algorithms::optimization_solver;
using namespace daal;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, logistic_regression::training::Method method, CpuType cpu>
services::Status TrainBatchKernel<algorithmFPType, method, cpu>::compute(
    const HostAppIfacePtr& pHost, const NumericTablePtr& x, const NumericTablePtr& y,
    logistic_regression::Model& m, Result& res, const Parameter& par)
{
    const size_t p = x->getNumberOfColumns() + 1;
    DAAL_ASSERT(p == m.getNumberOfBetas());
    services::SharedPtr<optimization_solver::iterative_solver::Batch > pSolver = par.optimizationSolver->clone();
    pSolver->setHostApp(pHost);
    if(par.nClasses == 2)
    {
        services::SharedPtr<logistic_loss::Batch<algorithmFPType>> objFunc(logistic_loss::Batch<algorithmFPType>::create(x->getNumberOfRows()));
        objFunc->input.set(logistic_loss::data, x);
        objFunc->input.set(logistic_loss::dependentVariables, y);
        objFunc->parameter().interceptFlag = par.interceptFlag;
        objFunc->parameter().penaltyL1 = par.penaltyL1;
        objFunc->parameter().penaltyL2 = par.penaltyL2;
        pSolver->getParameter()->function = objFunc;
    }
    else
    {
        services::SharedPtr<cross_entropy_loss::Batch<algorithmFPType>> objFunc(cross_entropy_loss::Batch<algorithmFPType>::create(par.nClasses, x->getNumberOfRows()));
        objFunc->input.set(cross_entropy_loss::data, x);
        objFunc->input.set(cross_entropy_loss::dependentVariables, y);
        objFunc->parameter().interceptFlag = par.interceptFlag;
        objFunc->parameter().penaltyL1 = par.penaltyL1;
        objFunc->parameter().penaltyL2 = par.penaltyL2;
        pSolver->getParameter()->function = objFunc;
    }

    const size_t nBetaRows = m.getBeta()->getNumberOfRows();
    const size_t nBetaTotal = p*nBetaRows;
    services::Status s;
    auto pArg = data_management::HomogenNumericTable<algorithmFPType>::create(1, nBetaTotal, NumericTable::doAllocate, 0, &s);
    if(!s)
        return s;

    //initialization
    {
        const size_t nRows = y->getNumberOfRows();
        daal::internal::ReadRows<algorithmFPType, cpu> yrows(*y, 0, nRows);
        daal::internal::WriteRows<algorithmFPType, cpu> argRows(*pArg, 0, nBetaTotal);
        DAAL_CHECK_BLOCK_STATUS(yrows);
        DAAL_CHECK_BLOCK_STATUS(argRows);
        const algorithmFPType* py = yrows.get();
        algorithmFPType* pb = argRows.get();
        if(par.nClasses == 2)
        {
            size_t count = 0;
            for(size_t i = 0; i < nRows; ++i)
                count += (py[i] != 0);
            algorithmFPType initialVal = 1;
            if(count && (count != nRows))
            {
                auto val = algorithmFPType(count) / (algorithmFPType(nRows) - algorithmFPType(count)); //mean/(1-mean)
                initialVal = daal::internal::Math<algorithmFPType, cpu>::sLog(val);
            }
            pb[0] = initialVal;
        }
        else
        {
            const algorithmFPType initialVal = 1e-3;
            for(size_t i = 0; i < par.nClasses; ++i)
                pb[i*p + 0] = initialVal;
        }
    }
    //initialize solver arguments
    pSolver->getInput()->set(optimization_solver::iterative_solver::inputArgument, pArg);
    DAAL_CHECK_STATUS(s, pSolver->compute());
    daal::internal::ReadRows<int, cpu> nIterationsRows(*pSolver->getResult()->get(optimization_solver::iterative_solver::nIterations),0,1);
    const int *pnIterations = nIterationsRows.get();
    NumericTablePtr nIterationsOut(data_management::HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, pnIterations[0], &s));
    DAAL_CHECK_STATUS_VAR(s);
    par.optimizationSolver->getResult()->set(optimization_solver::iterative_solver::nIterations,nIterationsOut);

    //write data to model
    daal::internal::ReadRows<algorithmFPType, cpu> ar(*pSolver->getResult()->get(optimization_solver::iterative_solver::minimum), 0, nBetaTotal);
    daal::internal::WriteRows<algorithmFPType, cpu> br(*m.getBeta(), 0, nBetaRows);
    DAAL_CHECK_BLOCK_STATUS(ar);
    DAAL_CHECK_BLOCK_STATUS(br);
    const algorithmFPType *a = ar.get();
    algorithmFPType *pBeta = br.get();
    for(size_t j = 0; j < nBetaTotal; j++)
        pBeta[j] = a[j];

    if(!par.interceptFlag)
    {
        for(size_t j = 0; j < nBetaRows; ++j)
            pBeta[p*j + 0] = 0;
    }
    return s;
}

} /* namespace internal */
} /* namespace training */
} /* namespace logistic_regression */
} /* namespace algorithms */
} /* namespace daal */

#endif

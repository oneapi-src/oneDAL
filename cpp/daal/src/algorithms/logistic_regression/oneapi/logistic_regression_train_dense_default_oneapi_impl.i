/* file: logistic_regression_train_dense_default_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of auxiliary functions for logistic regression classification
//  (defaultDense) method.
//--
*/

#ifndef __LOGISTIC_REGRESSION_TRAIN_DENSE_DEFAULT_ONEAPI_IMPL_I__
#define __LOGISTIC_REGRESSION_TRAIN_DENSE_DEFAULT_ONEAPI_IMPL_I__

#include "algorithms/optimization_solver/objective_function/logistic_loss_batch.h"
#include "algorithms/optimization_solver/objective_function/cross_entropy_loss_batch.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"

#include "src/services/service_data_utils.h"

#include "src/externals/service_profiler.h"

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
using namespace daal::algorithms::logistic_regression::training::internal;
using namespace daal::algorithms::optimization_solver;
using namespace daal::data_management;
using namespace daal::services::internal::sycl;

template <typename algorithmFPType, Method method>
services::Status TrainBatchKernelOneAPI<algorithmFPType, method>::compute(const services::HostAppIfacePtr & pHost, const NumericTablePtr & x,
                                                                          const NumericTablePtr & y, logistic_regression::Model & m, Result & res,
                                                                          const Parameter & par)
{
    services::Status status;

    const size_t p = x->getNumberOfColumns();
    const size_t n = x->getNumberOfRows();

    constexpr size_t maxInt32Value = static_cast<size_t>(daal::services::internal::MaxVal<int32_t>::get());

    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, p, 1);
    const size_t nBeta = p + 1;

    DAAL_ASSERT(nBeta == m.getNumberOfBetas());
    const size_t nClasses    = par.nClasses;
    const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

    DAAL_CHECK(nClasses <= maxInt32Value, services::ErrorIncorrectNumberOfClasses);
    DAAL_CHECK(p <= maxInt32Value, services::ErrorIncorrectNumberOfFeatures);

    auto & ctx = services::internal::getDefaultContext();

    services::SharedPtr<optimization_solver::iterative_solver::Batch> pSolver = par.optimizationSolver->clone();
    DAAL_ASSERT(pSolver == true);
    pSolver->setHostApp(pHost);
    if (nClasses == 2)
    {
        services::SharedPtr<logistic_loss::Batch<algorithmFPType> > objFunc(logistic_loss::Batch<algorithmFPType>::create(n));

        DAAL_ASSERT(objFunc == true);
        objFunc->input.set(logistic_loss::data, x);
        objFunc->input.set(logistic_loss::dependentVariables, y);
        objFunc->parameter().interceptFlag = par.interceptFlag;
        objFunc->parameter().penaltyL1     = par.penaltyL1;
        objFunc->parameter().penaltyL2     = par.penaltyL2;
        pSolver->getParameter()->function  = objFunc;
    }
    else
    {
        DAAL_CHECK(nClasses > 2, services::ErrorIncorrectParameter);

        services::SharedPtr<cross_entropy_loss::Batch<algorithmFPType> > objFunc(cross_entropy_loss::Batch<algorithmFPType>::create(nClasses, n));

        DAAL_ASSERT(objFunc == true);
        objFunc->input.set(cross_entropy_loss::data, x);
        objFunc->input.set(cross_entropy_loss::dependentVariables, y);
        objFunc->parameter().interceptFlag = par.interceptFlag;
        objFunc->parameter().penaltyL1     = par.penaltyL1;
        objFunc->parameter().penaltyL2     = par.penaltyL2;
        pSolver->getParameter()->function  = objFunc;
    }

    const size_t nBetaRows = m.getBeta()->getNumberOfRows();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBeta, nBetaRows);
    const size_t nBetaTotal = nBeta * nBetaRows;

    UniversalBuffer argumentU                                = ctx.allocate(idType, nBetaTotal, status);
    services::internal::Buffer<algorithmFPType> argumentBuff = argumentU.get<algorithmFPType>();

    auto argumentSNT = data_management::internal::SyclHomogenNumericTable<algorithmFPType>::create(argumentBuff, 1, nBetaTotal, &status);
    DAAL_CHECK_STATUS_VAR(status);

    ctx.fill(argumentU, 0.0, status);
    DAAL_CHECK_STATUS_VAR(status);

    //initialization
    if (nClasses != 2)
    {
        const algorithmFPType initialVal = algorithmFPType(1e-3);
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::setColElem(0, initialVal, argumentBuff, nClasses, nBeta));
    }

    //initialize solver arguments
    pSolver->getInput()->set(optimization_solver::iterative_solver::inputArgument, argumentSNT);

    DAAL_CHECK_STATUS(status, pSolver->computeNoThrow());

    {
        NumericTablePtr nIterationsNT = pSolver->getResult()->get(optimization_solver::iterative_solver::nIterations);

        BlockDescriptor<int> nIterationsBlock;
        DAAL_CHECK_STATUS(status, nIterationsNT->getBlockOfRows(0, 1, ReadWriteMode::readOnly, nIterationsBlock));
        const int * pnIterations = nIterationsBlock.getBlockPtr();

        NumericTablePtr nIterationsOut = data_management::HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, pnIterations[0], &status);
        DAAL_CHECK_STATUS_VAR(status);

        par.optimizationSolver->getResult()->set(optimization_solver::iterative_solver::nIterations, nIterationsOut);
        DAAL_CHECK_STATUS(status, nIterationsNT->releaseBlockOfRows(nIterationsBlock));
    }

    data_management::NumericTablePtr minimumSNT = pSolver->getResult()->get(optimization_solver::iterative_solver::minimum);
    BlockDescriptor<algorithmFPType> minimumBlock;
    DAAL_CHECK_STATUS(status, minimumSNT->getBlockOfRows(0, nBetaTotal, ReadWriteMode::readOnly, minimumBlock));

    services::internal::Buffer<algorithmFPType> minimumBuff = minimumBlock.getBuffer();

    data_management::NumericTablePtr betaNT = m.getBeta();
    {
        BlockDescriptor<algorithmFPType> dataRows;

        DAAL_CHECK_STATUS(status, betaNT->getBlockOfRows(0, nBetaRows, ReadWriteMode::writeOnly, dataRows));

        services::internal::Buffer<algorithmFPType> betaBuff = dataRows.getBuffer();

        DAAL_ASSERT(betaBuff.size() == nBetaTotal);
        DAAL_ASSERT(minimumBuff.size() == nBetaTotal);
        ctx.copy(betaBuff, 0, minimumBuff, 0, nBetaTotal, status);
        DAAL_CHECK_STATUS_VAR(status);

        if (!par.interceptFlag)
        {
            DAAL_CHECK(nBeta <= maxInt32Value, services::ErrorIncorrectNumberOfBetas);
            DAAL_CHECK(nBetaRows <= maxInt32Value, services::ErrorIncorrectNumberOfRows);
            DAAL_CHECK_STATUS(status, HelperObjectiveFunction::setColElem(0, algorithmFPType(0), betaBuff, nBetaRows, nBeta));
        }

        DAAL_CHECK_STATUS(status, betaNT->releaseBlockOfRows(dataRows));
    }

    DAAL_CHECK_STATUS(status, minimumSNT->releaseBlockOfRows(minimumBlock));

    return status;
}

} // namespace internal
} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

#endif

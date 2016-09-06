/* file: ridge_regression_train_dense_normeq_distr_step2_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of normel equations method for ridge regression
//  coefficients calculation
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_DISTR_STEP2_IMPL_I__
#define __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_DISTR_STEP2_IMPL_I__

#include "ridge_regression_train_kernel.h"
#include "ridge_regression_train_dense_normeq_impl.i"
#include "threading.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace internal
{

template <typename algorithmFpType, CpuType cpu>
void RidgeRegressionTrainDistributedKernel<algorithmFpType, training::normEqDense, cpu>::compute(
    size_t n, daal::algorithms::Model ** partialModels, daal::algorithms::Model * r, const daal::algorithms::Parameter * par)
{
    ridge_regression::Model * const rModel = static_cast<Model *>(r);
    rModel->initialize();
    for (size_t i = 0; i < n; ++i)
    {
        merge(partialModels[i], r, par);
    }
}

template <typename algorithmFpType, CpuType cpu>
void RidgeRegressionTrainDistributedKernel<algorithmFpType, training::normEqDense, cpu>::merge(
    daal::algorithms::Model * a, daal::algorithms::Model * r, const daal::algorithms::Parameter * par)
{
    ModelNormEq * const aa = static_cast<ModelNormEq *>(a);
    ModelNormEq * const rr = static_cast<ModelNormEq *>(r);

    const ridge_regression::Parameter * const parameter = static_cast<const ridge_regression::Parameter *>(par);
    size_t nBetas = aa->getNumberOfBetas();
    if(!parameter->interceptFlag) { nBetas--; }

    const size_t nResponses = aa->getNumberOfResponses();

    NumericTable * rxtxTable, * rxtyTable, * axtxTable, * axtyTable;
    BlockDescriptor<algorithmFpType> rxtxBD, rxtyBD, axtxBD, axtyBD;
    algorithmFpType * rxtx, * rxty, * axtx, * axty;

    getModelPartialSums<algorithmFpType, cpu>(aa, nBetas, nResponses, readOnly,  &axtxTable, axtxBD, &axtx, &axtyTable, axtyBD, &axty);
    getModelPartialSums<algorithmFpType, cpu>(rr, nBetas, nResponses, readWrite, &rxtxTable, rxtxBD, &rxtx, &rxtyTable, rxtyBD, &rxty);

    mergePartialSums(nBetas, nResponses, axtx, axty, rxtx, rxty);

    releaseModelNormEqPartialSums<algorithmFpType, cpu>(rxtxTable, rxtxBD, rxtyTable, rxtyBD);
    releaseModelNormEqPartialSums<algorithmFpType, cpu>(axtxTable, axtxBD, axtyTable, axtyBD);
}

template <typename algorithmFpType, CpuType cpu>
void RidgeRegressionTrainDistributedKernel<algorithmFpType, training::normEqDense, cpu>::mergePartialSums(
            MKL_INT nBetas, MKL_INT nResponses, algorithmFpType * axtx, algorithmFpType * axty, algorithmFpType * rxtx, algorithmFpType * rxty)
{
    const size_t xtxSize = nBetas * nBetas;
    daal::threader_for(xtxSize, xtxSize, [ = ](size_t i)
    {
        rxtx[i] += axtx[i];
    } );

    const size_t xtySize = nResponses * nBetas;
    daal::threader_for(xtySize, xtySize, [ = ](size_t i)
    {
        rxty[i] += axty[i];
    } );
}

template <typename algorithmFpType, CpuType cpu>
void RidgeRegressionTrainDistributedKernel<algorithmFpType, training::normEqDense, cpu>::finalizeCompute(
            ridge_regression::Model *a, ridge_regression::Model *r, const daal::algorithms::Parameter * par)
{
    finalizeModelNormEq<algorithmFpType, cpu>(a, r, par, this->_errors.get());
}

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif

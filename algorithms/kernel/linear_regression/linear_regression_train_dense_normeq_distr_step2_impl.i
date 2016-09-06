/* file: linear_regression_train_dense_normeq_distr_step2_impl.i */
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
//  Implementation of normel equations method for linear regression
//  coefficients calculation
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_DISTR_STEP2_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_DISTR_STEP2_IMPL_I__

#include "linear_regression_train_kernel.h"
#include "linear_regression_train_dense_normeq_impl.i"
#include "threading.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{

template <typename interm, CpuType cpu>
void LinearRegressionTrainDistributedKernel<interm, training::normEqDense, cpu>::compute(
            size_t n, daal::algorithms::Model **partialModels, daal::algorithms::Model *r,
            const daal::algorithms::Parameter *par)
{
    linear_regression::Model *rModel = static_cast<Model *>(r);
    rModel->initialize();
    for (size_t i = 0; i < n; i++)
    {
        merge(partialModels[i], r, par);
    }
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainDistributedKernel<interm, training::normEqDense, cpu>::merge(
            daal::algorithms::Model *a, daal::algorithms::Model *r, const daal::algorithms::Parameter *par)
{
    ModelNormEq *aa = static_cast<ModelNormEq *>(a);
    ModelNormEq *rr = static_cast<ModelNormEq *>(r);

    const linear_regression::Parameter *parameter = static_cast<const linear_regression::Parameter *>(par);
    size_t nBetas = aa->getNumberOfBetas();
    if(!parameter->interceptFlag) { nBetas--; }

    size_t nResponses = aa->getNumberOfResponses();

    NumericTable *rxtxTable, *rxtyTable, *axtxTable, *axtyTable;
    BlockDescriptor<interm> rxtxBD, rxtyBD, axtxBD, axtyBD;
    interm *rxtx, *rxty, *axtx, *axty;

    getModelPartialSums<interm, cpu>(aa, nBetas, nResponses, readOnly,  &axtxTable, axtxBD, &axtx, &axtyTable, axtyBD, &axty);
    getModelPartialSums<interm, cpu>(rr, nBetas, nResponses, readWrite, &rxtxTable, rxtxBD, &rxtx, &rxtyTable, rxtyBD, &rxty);

    mergePartialSums(nBetas, nResponses, axtx, axty, rxtx, rxty);

    releaseModelNormEqPartialSums<interm, cpu>(rxtxTable, rxtxBD, rxtyTable, rxtyBD);
    releaseModelNormEqPartialSums<interm, cpu>(axtxTable, axtxBD, axtyTable, axtyBD);
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainDistributedKernel<interm, training::normEqDense, cpu>::mergePartialSums(
            MKL_INT nBetas, MKL_INT nResponses, interm *axtx, interm *axty, interm *rxtx, interm *rxty)
{
    size_t xtxSize = nBetas * nBetas;
    daal::threader_for( xtxSize, xtxSize, [ = ](size_t i)
    {
        rxtx[i] += axtx[i];
    } );

    size_t xtySize = nResponses * nBetas;
    daal::threader_for( xtySize, xtySize, [ = ](size_t i)
    {
        rxty[i] += axty[i];
    } );
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainDistributedKernel<interm, training::normEqDense, cpu>::finalizeCompute(
            linear_regression::Model *a, linear_regression::Model *r,
            const daal::algorithms::Parameter *par)
{
    finalizeModelNormEq<interm, cpu>(a, r, this->_errors.get());
}

} // namespace internal
}
}
}
} // namespace daal

#endif

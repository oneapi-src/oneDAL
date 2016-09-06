/* file: linear_regression_train_dense_qr_distr_step2_impl.i */
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

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_QR_DISTR_STEP2_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_QR_DISTR_STEP2_IMPL_I__

#include "service_memory.h"
#include "linear_regression_train_kernel.h"
#include "linear_regression_train_dense_qr_impl.i"

using namespace daal::services::internal;

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
void LinearRegressionTrainDistributedKernel<interm, training::qrDense, cpu>::compute(
            size_t n,
            daal::algorithms::Model **partialModels,
            daal::algorithms::Model *r,
            const daal::algorithms::Parameter *par
           )
{
    linear_regression::Model *rModel = static_cast<Model *>(r);
    rModel->initialize();
    for (size_t i = 0; i < n; i++)
    {
        merge(partialModels[i], r, par);
        if(!this->_errors->isEmpty()) { return; }
    }
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainDistributedKernel<interm, training::qrDense, cpu>::merge(
            daal::algorithms::Model *a,
            daal::algorithms::Model *r,
            const daal::algorithms::Parameter *par
           )
{
    ModelQR *aa = static_cast<ModelQR *>(const_cast<daal::algorithms::Model *>(a));
    ModelQR *rr = static_cast<ModelQR *>(r);

    /* Retrieve data associated with input tables */
    MKL_INT nBetas = (MKL_INT)(aa->getNumberOfBetas());
    const linear_regression::Parameter *lrPar = static_cast<const linear_regression::Parameter *>(par);
    if(!lrPar->interceptFlag)
    {
        nBetas--;
    }

    MKL_INT nResponses  = (MKL_INT)(aa->getNumberOfResponses());
    MKL_INT nBetas2 = 2 * nBetas;

    /* Retrieve matrices R and Q'*Y from models */
    NumericTable *r1Table, *qty1Table, *r2Table, *qty2Table;
    BlockDescriptor<interm> r1BD, qty1BD, r2BD, qty2BD;
    interm *qrR1, *qrQTY1, *qrR2, *qrQTY2;
    getModelPartialSums<interm, cpu>(aa, nBetas, nResponses, readOnly,  &r1Table, r1BD, &qrR1, &qty1Table, qty1BD, &qrQTY1);
    getModelPartialSums<interm, cpu>(rr, nBetas, nResponses, readWrite, &r2Table, r2BD, &qrR2, &qty2Table, qty2BD, &qrQTY2);

    /* Allocate memory for intermediate calculations */
    interm *qrR12   = service_malloc<interm, cpu>(nBetas * nBetas2);
    interm *qrQTY12 = service_malloc<interm, cpu>(nResponses  * nBetas2);
    if (!qrR12 || !qrQTY12) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    interm *tau  = service_malloc<interm, cpu>(nBetas);

    MKL_INT lwork;
    computeQRWorkSize<interm, cpu>(&nBetas, &nBetas2, qrR12, tau, &lwork, _errors.get());
    if(!this->_errors->isEmpty()) { return; }

    interm* work = service_malloc<interm, cpu>(lwork);
    if (!work) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Merge models and store results into 'r' daal::algorithms::Model */
    mergeQR<interm, cpu>(&nBetas, &nResponses, qrR1, qrQTY1, qrR2, qrQTY2, qrR12, qrQTY12,
            qrR2, qrQTY2, tau, work, &lwork, _errors.get());
    if(!this->_errors->isEmpty()) { return; }

    /* Release memory */
    service_free<interm, cpu>(qrR12);
    service_free<interm, cpu>(qrQTY12);
    service_free<interm, cpu>(tau);
    service_free<interm, cpu>(work);

    releaseModelQRPartialSums<interm, cpu>(r1Table, r1BD, qty1Table, qty1BD);
    if(!this->_errors->isEmpty()) { return; }

    releaseModelQRPartialSums<interm, cpu>(r2Table, r2BD, qty2Table, qty2BD);
    if(!this->_errors->isEmpty()) { return; }
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainDistributedKernel<interm, training::qrDense, cpu>::finalizeCompute(
            linear_regression::Model *a,
            linear_regression::Model *r,
            const daal::algorithms::Parameter *par
           )
{
    finalizeModelQR<interm, cpu>(a, r, this->_errors.get());
}

} // namespace internal
}
}
}
} // namespace daal

#endif

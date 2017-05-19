/* file: linear_regression_train_dense_qr_distr_step2_impl.i */
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

template <typename algorithmFPType, CpuType cpu>
services::Status LinearRegressionTrainDistributedKernel<algorithmFPType, training::qrDense, cpu>::compute(
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
        if(!this->_errors->isEmpty()) { DAAL_RETURN_STATUS(); }
    }
    DAAL_RETURN_STATUS();
}

template <typename algorithmFPType, CpuType cpu>
void LinearRegressionTrainDistributedKernel<algorithmFPType, training::qrDense, cpu>::merge(
    daal::algorithms::Model *a,
    daal::algorithms::Model *r,
    const daal::algorithms::Parameter *par
)
{
    ModelQR *aa = static_cast<ModelQR *>(const_cast<daal::algorithms::Model *>(a));
    ModelQR *rr = static_cast<ModelQR *>(r);

    /* Retrieve data associated with input tables */
    DAAL_INT nBetas = (DAAL_INT)(aa->getNumberOfBetas());
    const linear_regression::Parameter *lrPar = static_cast<const linear_regression::Parameter *>(par);
    if(!lrPar->interceptFlag)
    {
        nBetas--;
    }

    DAAL_INT nResponses  = (DAAL_INT)(aa->getNumberOfResponses());
    DAAL_INT nBetas2 = 2 * nBetas;

    /* Retrieve matrices R and Q'*Y from models */
    NumericTable *r1Table, *qty1Table, *r2Table, *qty2Table;
    BlockDescriptor<algorithmFPType> r1BD, qty1BD, r2BD, qty2BD;
    algorithmFPType *qrR1, *qrQTY1, *qrR2, *qrQTY2;
    getModelPartialSums<algorithmFPType, cpu>(aa, nBetas, nResponses, readOnly,  &r1Table, r1BD, &qrR1, &qty1Table, qty1BD, &qrQTY1);
    getModelPartialSums<algorithmFPType, cpu>(rr, nBetas, nResponses, readWrite, &r2Table, r2BD, &qrR2, &qty2Table, qty2BD, &qrQTY2);

    /* Allocate memory for intermediate calculations */
    algorithmFPType *qrR12   = service_malloc<algorithmFPType, cpu>(nBetas * nBetas2);
    algorithmFPType *qrQTY12 = service_malloc<algorithmFPType, cpu>(nResponses  * nBetas2);
    if (!qrR12 || !qrQTY12) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    algorithmFPType *tau  = service_malloc<algorithmFPType, cpu>(nBetas);

    DAAL_INT lwork;
    computeQRWorkSize<algorithmFPType, cpu>(&nBetas, &nBetas2, qrR12, tau, &lwork, _errors.get());
    if(!this->_errors->isEmpty()) { return; }

    algorithmFPType *work = service_malloc<algorithmFPType, cpu>(lwork);
    if (!work) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Merge models and store results into 'r' daal::algorithms::Model */
    mergeQR<algorithmFPType, cpu>(&nBetas, &nResponses, qrR1, qrQTY1, qrR2, qrQTY2, qrR12, qrQTY12,
                                  qrR2, qrQTY2, tau, work, &lwork, _errors.get());
    if(!this->_errors->isEmpty()) { return; }

    /* Release memory */
    service_free<algorithmFPType, cpu>(qrR12);
    service_free<algorithmFPType, cpu>(qrQTY12);
    service_free<algorithmFPType, cpu>(tau);
    service_free<algorithmFPType, cpu>(work);

    releaseModelQRPartialSums<algorithmFPType, cpu>(r1Table, r1BD, qty1Table, qty1BD);
    if(!this->_errors->isEmpty()) { return; }

    releaseModelQRPartialSums<algorithmFPType, cpu>(r2Table, r2BD, qty2Table, qty2BD);
    if(!this->_errors->isEmpty()) { return; }
}

template <typename algorithmFPType, CpuType cpu>
services::Status LinearRegressionTrainDistributedKernel<algorithmFPType, training::qrDense, cpu>::finalizeCompute(
    linear_regression::Model *a,
    linear_regression::Model *r,
    const daal::algorithms::Parameter *par
)
{
    finalizeModelQR<algorithmFPType, cpu>(a, r, this->_errors.get());
    DAAL_RETURN_STATUS();
}

} // namespace internal
}
}
}
} // namespace daal

#endif

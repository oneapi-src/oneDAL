/* file: svm_predict_impl.i */
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
//  SVM prediction algorithm implementation
//--
*/

#ifndef __SVM_PREDICT_IMPL_I__
#define __SVM_PREDICT_IMPL_I__

#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_blas.h"
#include "service_memory.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace internal
{

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
struct SVMPredictImpl<defaultDense, algorithmFPType, cpu> : public Kernel
{
    services::Status compute(const NumericTablePtr& xTable, const daal::algorithms::Model *m, NumericTable& r,
                             const daal::algorithms::Parameter *par)
    {
        const size_t nVectors = xTable->getNumberOfRows();
        WriteOnlyColumns<algorithmFPType, cpu> mtR(r, 0, 0, nVectors);
        DAAL_CHECK_BLOCK_STATUS(mtR);
        algorithmFPType *distance = mtR.get();

        Model *model = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));
        Parameter *parameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

        NumericTablePtr svCoeffTable  = model->getClassificationCoefficients();
        const size_t nSV = svCoeffTable->getNumberOfRows();
        if(nSV == 0)
        {
            const algorithmFPType zero(0.0);
            for(size_t i = 0; i < nVectors; i++)
            {
                distance[i] = zero;
            }
            return Status();
        }

        const algorithmFPType bias(model->getBias());
        kernel_function::KernelIfacePtr kernel = parameter->kernel->clone();
        NumericTablePtr svTable = model->getSupportVectors();

        ReadColumns<algorithmFPType, cpu> mtSVCoeff(*svCoeffTable, 0, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSVCoeff);
        const algorithmFPType *svCoeff = mtSVCoeff.get();

        TArray<algorithmFPType, cpu> aBuf(nSV * nVectors);
        DAAL_CHECK(aBuf.get(), ErrorMemoryAllocationFailed);
        algorithmFPType *buf = aBuf.get();

        Status s;
        NumericTablePtr shResNT = HomogenNumericTableCPU<algorithmFPType, cpu>::create(buf, nSV, nVectors, &s);
        DAAL_CHECK_STATUS_VAR(s);

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, shResNT);
        kernel->setResult(shRes);
        kernel->getInput()->set(kernel_function::X, xTable);
        kernel->getInput()->set(kernel_function::Y, svTable);
        kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        s = kernel->computeNoThrow();
        if(!s)
            return Status(services::ErrorSVMPredictKernerFunctionCall).add(s);//this order is expected by test system

        char trans = 'T';
        DAAL_INT m_ = nSV;
        DAAL_INT n_ = nVectors;
        algorithmFPType alpha(1.0);
        DAAL_INT lda = m_;
        DAAL_INT incx(1);
        algorithmFPType beta(1.0);
        DAAL_INT incy(1);

        service_memset<algorithmFPType, cpu>(distance, bias, nVectors);

        Blas<algorithmFPType, cpu>::xgemv(&trans, &m_, &n_, &alpha, buf, &lda, svCoeff, &incx, &beta, distance, &incy);

        return s;
    }
};

} // namespace internal

} // namespace prediction

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif

/* file: svm_predict_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  SVM prediction algorithm implementation
//--
*/

#ifndef __SVM_PREDICT_IMPL_I__
#define __SVM_PREDICT_IMPL_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "externals/service_blas.h"
#include "externals/service_memory.h"

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
    services::Status compute(const NumericTablePtr & xTable, const daal::algorithms::Model * m, NumericTable & r,
                             const daal::algorithms::Parameter * par)
    {
        const size_t nVectors = xTable->getNumberOfRows();
        WriteOnlyColumns<algorithmFPType, cpu> mtR(r, 0, 0, nVectors);
        DAAL_CHECK_BLOCK_STATUS(mtR);
        algorithmFPType * distance = mtR.get();

        Model * model = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));
        kernel_function::KernelIfacePtr kernel;
        {
            svm::interface1::Parameter * parameter = dynamic_cast<svm::interface1::Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
            if (parameter) kernel = parameter->kernel->clone();
        }
        {
            svm::interface2::Parameter * parameter = dynamic_cast<svm::interface2::Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
            if (parameter) kernel = parameter->kernel->clone();
        }
        DAAL_CHECK(kernel, ErrorNullParameterNotSupported);

        NumericTablePtr svCoeffTable = model->getClassificationCoefficients();
        const size_t nSV             = svCoeffTable->getNumberOfRows();
        if (nSV == 0)
        {
            const algorithmFPType zero(0.0);
            service_memset<algorithmFPType, cpu>(distance, zero, nVectors);
            return Status();
        }

        const algorithmFPType bias(model->getBias());
        NumericTablePtr svTable = model->getSupportVectors();

        ReadColumns<algorithmFPType, cpu> mtSVCoeff(*svCoeffTable, 0, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSVCoeff);
        const algorithmFPType * svCoeff = mtSVCoeff.get();

        const size_t nRowsPerBlock = 256;
        const size_t nBlocks       = nVectors / nRowsPerBlock + !!(nVectors % nRowsPerBlock);

        /* TLS data initialization */
        daal::tls<algorithmFPType *> tls_data([&]() { return service_scalable_malloc<algorithmFPType, cpu>(nSV * nRowsPerBlock); });

        auto kfResultPtr = new kernel_function::Result();
        DAAL_CHECK_MALLOC(kfResultPtr);

        kernel_function::ResultPtr shRes(kfResultPtr);
        kernel->setResult(shRes);

        service_memset<algorithmFPType, cpu>(distance, bias, nVectors);

        SafeStatus safeStat;

        daal::threader_for(nBlocks, nBlocks, [&](int iBlock) {
            algorithmFPType * buf = tls_data.local();
            DAAL_CHECK_THR(buf, ErrorMemoryAllocationFailed);

            services::Status s;

            const size_t startRow          = iBlock * nRowsPerBlock;
            const size_t offestRow         = startRow + nRowsPerBlock;
            const size_t endRow            = services::internal::min<cpu, size_t>(offestRow, nVectors);
            const size_t nRowsPerBlockReal = endRow - startRow;

            NumericTablePtr shResNT = HomogenNumericTableCPU<algorithmFPType, cpu>::create(buf, nSV, nRowsPerBlockReal, &s);
            DAAL_CHECK_STATUS_THR(s);

            shRes->set(kernel_function::values, shResNT);

            kernel->getInput()->set(kernel_function::X, xTable);
            kernel->getInput()->set(kernel_function::Y, svTable);
            kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
            DAAL_CHECK_THR(kernel->computeNoThrow(), services::ErrorSVMPredictKernerFunctionCall);

            char trans  = 'T';
            DAAL_INT m_ = nSV;
            DAAL_INT n_ = nRowsPerBlockReal;
            algorithmFPType alpha(1.0);
            DAAL_INT lda = m_;
            DAAL_INT incx(1);
            algorithmFPType beta(1.0);
            DAAL_INT incy(1);

            Blas<algorithmFPType, cpu>::xxgemv(&trans, &m_, &n_, &alpha, buf, &lda, svCoeff, &incx, &beta, distance, &incy);
        }); /* daal::threader_for */

        tls_data.reduce([&](algorithmFPType * buf) { service_scalable_free<algorithmFPType, cpu>(buf); });

        return safeStat.detach();
    }
};

} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

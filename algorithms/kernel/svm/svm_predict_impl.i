/* file: svm_predict_impl.i */
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
//  SVM prediction algorithm implementation
//--
*/

#ifndef __SVM_PREDICT_IMPL_I__
#define __SVM_PREDICT_IMPL_I__

#include "service_memory.h"
#include "service_numeric_table.h"

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

        NumericTablePtr shResNT(new HomogenNumericTableCPU<algorithmFPType, cpu>(buf, nSV, nVectors));
        DAAL_CHECK(shResNT.get(), ErrorMemoryAllocationFailed);

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, shResNT);
        kernel->setResult(shRes);
        kernel->inputBase->set(kernel_function::X, xTable);
        kernel->inputBase->set(kernel_function::Y, svTable);
        kernel->parameterBase->computationMode = kernel_function::matrixMatrix;
        services::Status s = kernel->computeNoThrow();
        if(!s)
            return Status(services::ErrorSVMPredictKernerFunctionCall).add(s);//this order is expected by test system

        for (size_t i = 0; i < nVectors; i++)
        {
            distance[i] = bias;
            for (size_t j = 0; j < nSV; j++)
            {
                distance[i] += buf[i * nSV + j] * svCoeff[j];
            }
        }
        return s;
    }
};

} // namespace internal

} // namespace prediction

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif

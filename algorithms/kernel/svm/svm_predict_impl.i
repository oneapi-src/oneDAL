/* file: svm_predict_impl.i */
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
//  SVM prediction algorithm implementation
//--
*/

#ifndef __SVM_PREDICT_IMPL_I__
#define __SVM_PREDICT_IMPL_I__

#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services::internal;

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

template <typename AlgorithmFPType, CpuType cpu>
struct SVMPredictImpl<defaultDense, AlgorithmFPType, cpu> : public Kernel
{
    void compute(const NumericTablePtr a, const daal::algorithms::Model *m, NumericTablePtr r,
                 const daal::algorithms::Parameter *par)
    {
        AlgorithmFPType zero = 0.0;
        NumericTablePtr xTable = a;
        FeatureMicroTable<AlgorithmFPType, writeOnly, cpu> mtR(r.get());
        size_t nVectors = xTable->getNumberOfRows();

        Model *model = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));
        Parameter *parameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

        services::SharedPtr<kernel_function::KernelIface> kernel = parameter->kernel->clone();
        kernel->getErrors()->setCanThrow(false);

        NumericTablePtr svTable       = model->getSupportVectors();
        NumericTablePtr svCoeffTable  = model->getClassificationCoefficients();
        AlgorithmFPType bias = (AlgorithmFPType)model->getBias();

        FeatureMicroTable<AlgorithmFPType, readOnly, cpu> mtSVCoeff(svCoeffTable.get());

        size_t nSV = mtSVCoeff.getFullNumberOfRows();
        AlgorithmFPType *svCoeff;

        AlgorithmFPType *distance;
        mtR.getBlockOfColumnValues(0, 0, nVectors, &distance);

        if (nSV == 0)
        {
            for (size_t i = 0; i < nVectors; i++)
            {
                distance[i] = zero;
            }
        }
        else
        {
            mtSVCoeff.getBlockOfColumnValues(0, 0, nSV, &svCoeff);

            AlgorithmFPType *buf = (AlgorithmFPType *)daal::services::daal_malloc(nSV * nVectors * sizeof(AlgorithmFPType));
            if (buf == NULL) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

            NumericTablePtr shResNT(new HomogenNumericTableCPU<AlgorithmFPType, cpu>(buf, nSV, nVectors));

            services::SharedPtr<kernel_function::Result> shRes(new kernel_function::Result());
            shRes->set(kernel_function::values, shResNT);

            kernel->setResult(shRes);
            kernel->inputBase->set(kernel_function::X, xTable);
            kernel->inputBase->set(kernel_function::Y, svTable);
            kernel->parameterBase->computationMode = kernel_function::matrixMatrix;
            kernel->computeNoThrow();
            if(kernel->getErrors()->size() != 0)
            {
                mtSVCoeff.release();
                daal::services::daal_free(buf);
                mtR.release();
                this->_errors->add(services::ErrorSVMinnerKernel);
                this->_errors->add(kernel->getErrors()->getErrors());
                return;
            }
            for (size_t i = 0; i < nVectors; i++)
            {
                distance[i] = bias;
                for (size_t j = 0; j < nSV; j++)
                {
                    distance[i] += buf[i * nSV + j] * svCoeff[j];
                }
            }

            mtSVCoeff.release();
            daal::services::daal_free(buf);
        }
        mtR.release();
    }
};

} // namespace internal

} // namespace prediction

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif

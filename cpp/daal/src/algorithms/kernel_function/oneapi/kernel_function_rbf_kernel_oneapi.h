/* file: kernel_function_rbf_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Declaration of template structs that calculate SVM RBF Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_DENSE_RBF_KERNEL_ONEAPI_H__
#define __KERNEL_FUNCTION_DENSE_RBF_KERNEL_ONEAPI_H__

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/kernel_function/kernel_function_rbf.h"
#include "src/algorithms/kernel_function/oneapi/kernel_function_helper_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::services::internal::sycl;

template <Method method, typename algorithmFPType>
class KernelImplRBFOneAPI : public Kernel
{
public:
    services::Status compute(NumericTable * ntLeft, NumericTable * ntRight, NumericTable * result, const ParameterBase * par)
    {
        return services::ErrorMethodNotImplemented;
    }
};

template <typename algorithmFPType>
class KernelImplRBFOneAPI<defaultDense, algorithmFPType> : public Kernel
{
public:
    using Helper = HelperKernel<algorithmFPType>;

    services::Status compute(NumericTable * ntLeft, NumericTable * ntRight, NumericTable * result, const ParameterBase * par)
    {
        ComputationMode computationMode = par->computationMode;
        switch (computationMode)
        {
        case vectorVector: return computeInternalVectorVector(ntLeft, ntRight, result, par);
        case matrixVector: return computeInternalMatrixVector(ntLeft, ntRight, result, par);
        case matrixMatrix: return computeInternalMatrixMatrix(ntLeft, ntRight, result, par);
        default: return services::ErrorIncorrectParameter;
        }
    }

protected:
    services::Status computeInternalVectorVector(NumericTable * vecLeft, NumericTable * vecRight, NumericTable * result, const ParameterBase * par);
    services::Status computeInternalMatrixVector(NumericTable * matLeft, NumericTable * vecRight, NumericTable * result, const ParameterBase * par);
    services::Status computeInternalMatrixMatrix(NumericTable * matLeft, NumericTable * matRight, NumericTable * result, const ParameterBase * par);

private:
    UniversalBuffer _sqrMatLeft;
    UniversalBuffer _sqrMatRight;
};

template <typename algorithmFPType>
class KernelImplRBFOneAPI<fastCSR, algorithmFPType> : public Kernel
{
public:
    using Helper = HelperKernel<algorithmFPType>;

    services::Status compute(NumericTable * ntLeft, NumericTable * ntRight, NumericTable * result, const ParameterBase * par)
    {
        ComputationMode computationMode = par->computationMode;
        switch (computationMode)
        {
        case vectorVector: return computeInternalVectorVector(ntLeft, ntRight, result, par);
        case matrixVector: return computeInternalMatrixVector(ntLeft, ntRight, result, par);
        case matrixMatrix: return computeInternalMatrixMatrix(ntLeft, ntRight, result, par);
        default: return services::ErrorIncorrectParameter;
        }
        return services::Status();
    }

protected:
    services::Status computeInternalVectorVector(NumericTable * vecLeft, NumericTable * vecRight, NumericTable * result, const ParameterBase * par);
    services::Status computeInternalMatrixVector(NumericTable * matLeft, NumericTable * vecRight, NumericTable * result, const ParameterBase * par);
    services::Status computeInternalMatrixMatrix(NumericTable * matLeft, NumericTable * matRight, NumericTable * result, const ParameterBase * par);

private:
    UniversalBuffer _sqrMatLeft;
    UniversalBuffer _sqrMatRight;
};

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif

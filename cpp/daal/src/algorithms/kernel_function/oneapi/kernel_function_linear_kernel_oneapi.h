/* file: kernel_function_linear_kernel_oneapi.h */
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
//  Declaration of template structs that calculate SVM Linear Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_DENSE_LINEAR_KERNEL_ONEAPI_H__
#define __KERNEL_FUNCTION_DENSE_LINEAR_KERNEL_ONEAPI_H__

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/kernel_function/kernel_function_types_linear.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;

template <Method method, typename algorithmFPType>
class KernelImplLinearOneAPI : public Kernel
{
public:
    services::Status compute(NumericTable * ntLeft, NumericTable * ntRight, NumericTable * result, const ParameterBase * par)
    {
        return services::ErrorMethodNotImplemented;
    }
};

template <typename algorithmFPType>
class KernelImplLinearOneAPI<defaultDense, algorithmFPType> : public Kernel
{
public:
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
};

template <typename algorithmFPType>
class KernelImplLinearOneAPI<fastCSR, algorithmFPType> : public Kernel
{
public:
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
};

} // namespace internal
} // namespace linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif

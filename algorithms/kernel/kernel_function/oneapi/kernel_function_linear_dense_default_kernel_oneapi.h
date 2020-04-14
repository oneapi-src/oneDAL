/* file: kernel_function_linear_dense_default_kernel_oneapi.h */
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

#include "algorithms/kernel/kernel_function/oneapi/kernel_function_linear_base_oneapi.h"

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

template <typename algorithmFPType>
class KernelImplLinearOneAPI<defaultDense, algorithmFPType> : public Kernel
{
public:
    services::Status compute(ComputationMode computationMode, NumericTable * a1, NumericTable * a2, NumericTable * r,
                             const daal::algorithms::Parameter * par)
    {
        const ParameterBase * svmPar = static_cast<const ParameterBase *>(par);

        switch (computationMode)
        {
        case vectorVector: return computeInternalVectorVector(a1, a2, r, svmPar); break;
        case matrixVector: return computeInternalMatrixVector(a1, a2, r, svmPar); break;
        case matrixMatrix: return computeInternalMatrixMatrix(a1, a2, r, svmPar); break;
        }

        DAAL_ASSERT(false); //should never come here
        return services::Status();
    }

protected:
    services::Status computeInternalVectorVector(NumericTable * a1, NumericTable * a2, NumericTable * r, const ParameterBase * par);
    services::Status computeInternalMatrixVector(NumericTable * a1, NumericTable * a2, NumericTable * r, const ParameterBase * par);
    services::Status computeInternalMatrixMatrix(NumericTable * a1, NumericTable * a2, NumericTable * r, const ParameterBase * par);
};

} // namespace internal
} // namespace linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif

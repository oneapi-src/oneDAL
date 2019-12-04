/* file: kernel_function_dense_base.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Declaration of template structs that calculate SVM Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_DENSE_BASE_H__
#define __KERNEL_FUNCTION_DENSE_BASE_H__

#include "numeric_table.h"
#include "kernel_function_types_linear.h"
#include "kernel_function_types_rbf.h"
#include "kernel_function_linear.h"
#include "kernel_function_rbf.h"
#include "service_micro_table.h"
#include "kernel.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
struct KernelImplBase : public Kernel
{
    virtual services::Status computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const ParameterBase * par) = 0;
    virtual services::Status computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const ParameterBase * par) = 0;
    virtual services::Status computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const ParameterBase * par) = 0;

    services::Status compute(ComputationMode computationMode, const NumericTable * a1, const NumericTable * a2, NumericTable * r,
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
};

} // namespace internal

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif

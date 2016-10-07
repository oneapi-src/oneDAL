/* file: adagrad_dense_default_kernel.h */
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

//++
//  Declaration of template function that calculate adagrad.
//--


#ifndef __ADAGRAD_DENSE_DEFAULT_KERNEL_H__
#define __ADAGRAD_DENSE_DEFAULT_KERNEL_H__

#include "adagrad_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_rng.h"
#include "service_math.h"
#include "service_micro_table.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
class AdagradKernel: public Kernel
{
public:
    void compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
                 NumericTable *gradientSquareSumResult, NumericTable *gradientSquareSumInput,
                 OptionalArgument *optionalArgument, OptionalArgument *optionalResult, Parameter *parameter);

protected:
    algorithmFPType vectorNorm(const algorithmFPType *vec, size_t nElements)
    {
        algorithmFPType norm = 0;
        for(size_t i = 0; i < nElements; i++)
        {
            norm += vec[i] * vec[i];
        }
        return daal::internal::Math<algorithmFPType,cpu>::sSqrt(norm);
    }
};


} // namespace daal::internal

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

/* file: kernel_function_linear_batch_container.h */
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
//  Implementation of kernel function container.
//--
*/

#include "kernel_function_linear.h"
#include "kernel_function_linear_dense_default_kernel.h"
#include "kernel_function_linear_csr_fast_kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KernelImplLinear, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    NumericTable * a[2];
    a[0] = static_cast<NumericTable *>(input->get(X).get());
    a[1] = static_cast<NumericTable *>(input->get(Y).get());

    NumericTable * r[1];
    r[0] = static_cast<NumericTable *>(result->get(values).get());

    algorithms::Parameter * par            = _par;
    daal::services::Environment::env & env = *_env;

    ComputationMode computationMode = static_cast<ParameterBase *>(par)->computationMode;

    __DAAL_CALL_KERNEL(env, internal::KernelImplLinear, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, computationMode, a[0], a[1], r[0],
                       par);
}

}; // namespace linear

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

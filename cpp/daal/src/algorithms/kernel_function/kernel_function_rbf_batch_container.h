/* file: kernel_function_rbf_batch_container.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "algorithms/kernel_function/kernel_function_rbf.h"
#include "src/algorithms/kernel_function/kernel_function_rbf_dense_default_kernel.h"
#include "src/algorithms/kernel_function/kernel_function_rbf_csr_fast_kernel.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
using namespace daal::data_management;

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KernelImplRBF, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Result * result = static_cast<Result *>(_res);
    Input * input   = static_cast<Input *>(_in);

    NumericTable * a[2];
    a[0] = static_cast<NumericTable *>(input->get(X).get());
    a[1] = static_cast<NumericTable *>(input->get(Y).get());

    NumericTable * r[1];
    r[0] = static_cast<NumericTable *>(result->get(values).get());

    daal::services::Environment::env & env = *_env;
    const ParameterBase * par              = static_cast<const ParameterBase *>(_par);

    KernelParameter kernelPar;
    kernelPar.rowIndexX       = par->rowIndexX;
    kernelPar.rowIndexY       = par->rowIndexY;
    kernelPar.rowIndexResult  = par->rowIndexResult;
    kernelPar.computationMode = par->computationMode;
    kernelPar.sigma           = static_cast<const Parameter *>(par)->sigma;
    kernelPar.kernelType      = KernelType::rbf;

    if (method == fastCSR)
    {
        if (dynamic_cast<CSRNumericTableIface *>(a[0]) == NULL || dynamic_cast<CSRNumericTableIface *>(a[1]) == NULL)
            return services::Status(services::ErrorIncorrectTypeOfInputNumericTable);
    }

    __DAAL_CALL_KERNEL(env, internal::KernelImplRBF, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a[0], a[1], r[0], &kernelPar);
}

} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

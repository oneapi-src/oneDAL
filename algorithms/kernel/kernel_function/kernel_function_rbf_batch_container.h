/* file: kernel_function_rbf_batch_container.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of kernel function container.
//--
*/

#include "kernel_function_rbf.h"
#include "kernel_function_rbf_dense_default_kernel.h"
#include "kernel_function_rbf_csr_fast_kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
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
    Result *result = static_cast<Result *>(_res);
    Input *input = static_cast<Input *>(_in);

    NumericTable *a[2];
    a[0] = static_cast<NumericTable *>(input->get(X).get());
    a[1] = static_cast<NumericTable *>(input->get(Y).get());

    NumericTable *r[1];
    r[0] = static_cast<NumericTable *>(result->get(values).get());

    algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    ComputationMode computationMode = static_cast<ParameterBase *>(par)->computationMode;

    if (method == fastCSR)
    {
        if (dynamic_cast<CSRNumericTableIface *>(a[0]) == NULL ||
            dynamic_cast<CSRNumericTableIface *>(a[1]) == NULL) return services::Status(services::ErrorIncorrectTypeOfInputNumericTable);
    }

    __DAAL_CALL_KERNEL(env, internal::KernelImplRBF, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, computationMode, a[0], a[1],
                       r[0], par);
}

} // namespace rbf

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

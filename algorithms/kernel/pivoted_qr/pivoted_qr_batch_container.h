/* file: pivoted_qr_batch_container.h */
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
//  Implementation of pivoted_qr calculation algorithm container.
//--
*/

#ifndef __PIVOTED_QR_BATCH_CONTAINER_H__
#define __PIVOTED_QR_BATCH_CONTAINER_H__

#include "pivoted_qr_types.h"
#include "pivoted_qr_batch.h"
#include "pivoted_qr_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
{

template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PivotedQRKernel, method, interm);
}

template<typename interm, Method method, CpuType cpu>
BatchContainer<interm, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename interm, Method method, CpuType cpu>
void BatchContainer<interm, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    size_t na = input->size();
    size_t nr = result->size();

    NumericTable *a0 = static_cast<NumericTable *>(input->get(data).get());
    NumericTable **a = &a0;
    NumericTable *r[3];
    r[0] = static_cast<NumericTable *>(result->get(matrixQ).get());
    r[1] = static_cast<NumericTable *>(result->get(matrixR).get());
    r[2] = static_cast<NumericTable *>(result->get(permutationMatrix).get());
    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PivotedQRKernel, __DAAL_KERNEL_ARGUMENTS(method, interm), compute, na, a, nr, r, par);
}

} //namespace pivoted_qr

} //namespace algorithms

} //namespace daal

#endif

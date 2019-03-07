/* file: pivoted_qr_batch_container.h */
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

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PivotedQRKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);
    Parameter *parameter = static_cast<Parameter *>(_par);

    size_t na = input->size();
    size_t nr = result->size();

    NumericTable *dataTable = static_cast<NumericTable *>(input->get(data).get());
    NumericTable *matrixQTable = static_cast<NumericTable *>(result->get(matrixQ).get());
    NumericTable *matrixRTable = static_cast<NumericTable *>(result->get(matrixR).get());
    NumericTable *permutationMatrixTable = static_cast<NumericTable *>(result->get(permutationMatrix).get());
    NumericTable *permutedColumnsTable = parameter->permutedColumns.get();
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PivotedQRKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute,
                       *dataTable, *matrixQTable, *matrixRTable, *permutationMatrixTable, permutedColumnsTable);
}

} //namespace pivoted_qr

} //namespace algorithms

} //namespace daal

#endif

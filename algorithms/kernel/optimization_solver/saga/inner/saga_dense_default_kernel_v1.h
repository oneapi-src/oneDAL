/* file: saga_dense_default_kernel_v1.h */
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

//++
//  Declaration of template function that calculate saga.
//--


#ifndef __SAGA_DENSE_DEFAULT_KERNEL_V1_H__
#define __SAGA_DENSE_DEFAULT_KERNEL_V1_H__

#include "saga_batch.h"
#include "kernel.h"
#include "numeric_table.h"
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
namespace saga
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
class I1SagaKernel: public Kernel
{
public:
    services::Status compute(HostAppIface* pHost, NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
                             NumericTable *gradientsTableInput, NumericTable *gradientsTableResult, interface1::Parameter *parameter, engines::BatchBase &engine);

};

} // namespace daal::internal

} // namespace saga

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

/* file: adagrad_dense_default_kernel.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Declaration of template function that calculate adagrad.
//--


#ifndef __ADAGRAD_DENSE_DEFAULT_KERNEL_H__
#define __ADAGRAD_DENSE_DEFAULT_KERNEL_H__

#include "adagrad_batch.h"
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
namespace adagrad
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
class AdagradKernel: public Kernel
{
public:
    services::Status compute(HostAppIface* pHost, NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
                 NumericTable *gradientSquareSumResult, NumericTable *gradientSquareSumInput,
                 OptionalArgument *optionalArgument, OptionalArgument *optionalResult, Parameter *parameter, engines::BatchBase &engine);
private:
    static services::Status initAccumulatedGrad(algorithmFPType *accumulatedG, size_t nRows, NumericTable *pOptInput);
    static const size_t _blockSize = 512;
    static const size_t _threadStart = 50000;
};


} // namespace daal::internal

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

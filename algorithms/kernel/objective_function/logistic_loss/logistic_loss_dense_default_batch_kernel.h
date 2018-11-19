/* file: logistic_loss_dense_default_batch_kernel.h */
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
//  Declaration of template function that calculate logloss.
//--


#ifndef __LOGISTIC_LOSS_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __LOGISTIC_LOSS_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "logistic_loss_batch.h"
#include "kernel.h"
#include "service_numeric_table.h"
#include "service_blas.h"
#include "numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace internal
{

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template<typename algorithmFPType, Method method, CpuType cpu>
class LogLossKernel : public Kernel
{
public:
    services::Status compute(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
                          NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter);
    static void applyBeta(const algorithmFPType* x, const algorithmFPType* beta, algorithmFPType* xb, size_t nRows, size_t nCols, bool bIntercept);
    static void applyBetaThreaded(const algorithmFPType* x, const algorithmFPType* beta, algorithmFPType* xb, size_t nRows, size_t nCols, bool bIntercept);
    static void sigmoid(const algorithmFPType* f, algorithmFPType* s, size_t n);

protected:
    services::Status doCompute(const algorithmFPType* x, const algorithmFPType* y,
        size_t n, size_t p, NumericTable *betaNT, NumericTable *valueNT,
        NumericTable *hessianNT, NumericTable *gradientNT, Parameter *parameter);
};

} // namespace daal::internal

} // namespace logistic_loss

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

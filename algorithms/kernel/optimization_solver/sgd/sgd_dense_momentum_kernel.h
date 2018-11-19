/* file: sgd_dense_momentum_kernel.h */
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
//  Declaration of template function that calculate sgd.
//--


#ifndef __SGD_DENSE_MOMENTUM_KERNEL_H__
#define __SGD_DENSE_MOMENTUM_KERNEL_H__

#include "sgd_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "iterative_solver_kernel.h"
#include "sgd_dense_kernel.h"
#include "sgd_dense_minibatch_kernel.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_utils.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{

/**
* Statuses of the indices of objective function terms that are used for gradient
*/

template<typename algorithmFPType, CpuType cpu>
class SGDKernel<algorithmFPType, momentum, cpu> : public iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>
{
public:
    services::Status compute(HostAppIface* pHost, NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
                 Parameter<momentum> *parameter, NumericTable *learningRateSequence,
                 NumericTable *batchIndices, OptionalArgument *optionalArgument, OptionalArgument *optionalResult, engines::BatchBase &engine);
    using iterative_solver::internal::IterativeSolverKernel<algorithmFPType, cpu>::vectorNorm;
};

template<typename algorithmFPType, CpuType cpu>
struct SGDmomentumTask
{
    SGDmomentumTask(
        size_t batchSize_,
        size_t nTerms_,
        NumericTable *resultTable,
        NumericTable *batchIndicesTable,
        NumericTable *pastUpdateResult,
        NumericTable *lastIterationResultNT,
        Parameter<momentum> *parameter);

    virtual ~SGDmomentumTask();

    Status init(NumericTable *batchIndicesTable, NumericTable *resultTable, Parameter<momentum> *parameter,
        NumericTable *pastUpdateInput,
        NumericTable *lastIterationInput);

    Status setStartValue(NumericTable *inputArgument, NumericTable *minimum);

    Status makeStep(NumericTable *gradient,
                  NumericTable *minimum,
                  NumericTable *pastUpdate,
                  const algorithmFPType learningRate,
                  const algorithmFPType momentum);

    size_t batchSize;
    size_t nTerms;
    size_t startIteration;
    size_t nProceededIters;

    IndicesStatus   indicesStatus;

    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu>> ntBatchIndices;
    NumericTablePtr minimimWrapper;
    NumericTablePtr pastUpdate;
    NumericTablePtr lastIterationResult;
};

} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

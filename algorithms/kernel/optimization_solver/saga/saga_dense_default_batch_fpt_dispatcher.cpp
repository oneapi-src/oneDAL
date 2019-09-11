/* file: saga_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of saga calculation algorithm container.
//--


#include "saga_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(optimization_solver::saga::BatchContainer, batch, DAAL_FPTYPE, optimization_solver::saga::defaultDense)

namespace optimization_solver
{
namespace saga
{

namespace interface2
{
using BatchType = Batch<DAAL_FPTYPE, optimization_solver::saga::defaultDense>;

template<>
BatchType::Batch(const sum_of_functions::BatchPtr& objectiveFunction)
{
    _par = new algorithms::optimization_solver::saga::Parameter(objectiveFunction);
    initialize();
}

template<>
BatchType::Batch(const BatchType &other) :
    iterative_solver::Batch(other),
    input(other.input)
{
    _par = new algorithms::optimization_solver::saga::Parameter(other.parameter());
    initialize();
}

template<>
services::SharedPtr<BatchType> BatchType::create()
{
    return services::SharedPtr<BatchType>(new BatchType());
}
} // namespace interface2
} // namespace saga
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

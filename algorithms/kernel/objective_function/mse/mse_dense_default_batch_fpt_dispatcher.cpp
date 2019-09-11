/* file: mse_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of mse calculation algorithm container.
//--


#include "mse_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(optimization_solver::mse::interface2::BatchContainer, batch, DAAL_FPTYPE, optimization_solver::mse::defaultDense);

namespace optimization_solver
{
namespace mse
{
namespace interface2
{
using BatchType = Batch<DAAL_FPTYPE, optimization_solver::mse::defaultDense>;

template<>
BatchType::Batch(size_t numberOfTerms) : sum_of_functions::Batch(numberOfTerms, &input, new ParameterType(numberOfTerms))
{
    initialize();
    _par = sumOfFunctionsParameter;
}

template<>
BatchType::Batch(const BatchType &other) :
               sum_of_functions::Batch(other.parameter().numberOfTerms, &input, new ParameterType(other.parameter())), input(other.input)
{
    initialize();
    _par = sumOfFunctionsParameter;
}

} // namespace interface2
} // namespace mse
} // namespace optimization_solver
} // namespace algorithms

} // namespace daal

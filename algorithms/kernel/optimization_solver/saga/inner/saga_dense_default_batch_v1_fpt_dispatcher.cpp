/* file: saga_dense_default_batch_v1_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

//++
//  Implementation of saga calculation algorithm container.
//--

#include "saga_batch_container_v1.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(optimization_solver::saga::interface1::BatchContainer, batch, DAAL_FPTYPE,
                                      optimization_solver::saga::defaultDense)

namespace optimization_solver
{
namespace saga
{
namespace interface1
{
using BatchType = Batch<DAAL_FPTYPE, optimization_solver::saga::defaultDense>;

template <>
BatchType::Batch(const sum_of_functions::interface1::BatchPtr & objectiveFunction)
{
    _par = new algorithms::optimization_solver::saga::interface1::Parameter(objectiveFunction);
    initialize();
}

template <>
BatchType::Batch(const BatchType & other) : iterative_solver::interface1::Batch(other), input(other.input)
{
    _par = new algorithms::optimization_solver::saga::interface1::Parameter(other.parameter());
    initialize();
}

template <>
services::SharedPtr<BatchType> BatchType::create()
{
    return services::SharedPtr<BatchType>(new BatchType());
}
} // namespace interface1

} // namespace saga
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

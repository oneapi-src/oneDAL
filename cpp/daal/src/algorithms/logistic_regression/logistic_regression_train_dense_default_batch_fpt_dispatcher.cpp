/* file: logistic_regression_train_dense_default_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of logistic regression container.
//--
*/

#include "src/algorithms/logistic_regression/logistic_regression_train_container.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER_SYCL(logistic_regression::training::BatchContainer, batch, DAAL_FPTYPE,
                                           logistic_regression::training::defaultDense)

namespace logistic_regression
{
namespace training
{
namespace interface3
{
template <>
Batch<DAAL_FPTYPE, logistic_regression::training::defaultDense>::Batch(size_t nClasses, const SolverPtr & solver)
{
    _par = new ParameterType(nClasses, solver);
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, logistic_regression::training::defaultDense>;
template <>
Batch<DAAL_FPTYPE, logistic_regression::training::defaultDense>::Batch(const BatchType & other)
    : classifier::training::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface3
} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

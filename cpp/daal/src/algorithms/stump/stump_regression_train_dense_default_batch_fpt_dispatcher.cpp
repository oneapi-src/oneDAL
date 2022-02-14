/* file: stump_regression_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Decision Stump algorithm container -- a class that contains
//  Friedman Decision Stump kernels for supported architectures.
//--
*/

#include "src/algorithms/stump/stump_regression_train_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(stump::regression::training::BatchContainer, batch, DAAL_FPTYPE, stump::regression::training::defaultDense)

namespace stump
{
namespace regression
{
namespace training
{
namespace interface1
{
template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch()
{
    _par = new ParameterType();
    initialize();
}

template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(const Batch & other) : algorithms::regression::training::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

template Batch<DAAL_FPTYPE, stump::regression::training::defaultDense>::Batch();
template Batch<DAAL_FPTYPE, stump::regression::training::defaultDense>::Batch(const Batch &);

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace stump

} // namespace algorithms
} // namespace daal

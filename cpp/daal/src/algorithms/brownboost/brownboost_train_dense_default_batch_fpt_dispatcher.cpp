/* file: brownboost_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Brown Boost algorithm container -- a class that contains
//  Freund Brown Boost kernels for supported architectures.
//--
*/

#include "src/algorithms/brownboost/brownboost_train_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(brownboost::training::BatchContainer, batch, DAAL_FPTYPE, brownboost::training::defaultDense)

namespace brownboost
{
namespace training
{
namespace interface2
{
template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch()
{
    _par = new ParameterType();
    initialize();
}

template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(size_t nClasses)
{
    ParameterType * param = new ParameterType();
    param->nClasses       = nClasses;
    _par                  = param;
    initialize();
}

template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(const Batch & other) : classifier::training::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

template Batch<DAAL_FPTYPE, brownboost::training::defaultDense>::Batch();
template Batch<DAAL_FPTYPE, brownboost::training::defaultDense>::Batch(size_t nClasses);
template Batch<DAAL_FPTYPE, brownboost::training::defaultDense>::Batch(const Batch &);

} // namespace interface2
} // namespace training
} // namespace brownboost

} // namespace algorithms
} // namespace daal

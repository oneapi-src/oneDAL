/* file: em_gmm_dense_batch_fpt.cpp */
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
//  Implementation of EM Batch constructor
//--
*/

#include "algorithms/em/em_gmm.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
template <typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(const size_t nComponents)
    : parameter(nComponents, services::SharedPtr<covariance::Batch<algorithmFPType, covariance::defaultDense> >(
                                 new covariance::Batch<algorithmFPType, covariance::defaultDense>()))
{
    initialize();
}

template <typename algorithmFPType, Method method>
void Batch<algorithmFPType, method>::initialize()
{
    Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
    _in                  = &input;
    _par                 = &parameter;
    _result              = ResultPtr(new Result());
}

template class Batch<DAAL_FPTYPE, defaultDense>;

} // namespace em_gmm
} // namespace algorithms
} // namespace daal

/* file: gbt_regression_init_distr_step1_fpt_dispatcher.cpp */
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

/*
//++
//  Implementation of  container for initializing gradient boosted trees
//  regression training algorithm in the distributed processing mode
//--
*/

#include "gbt_regression_init_container.h"

namespace daal
{
namespace algorithms
{

__DAAL_INSTANTIATE_DISPATCH_CONTAINER(gbt::regression::init::interface1::DistributedContainer, distributed, step1Local,  \
                                      DAAL_FPTYPE, gbt::regression::init::defaultDense)

namespace gbt
{
namespace regression
{
namespace init
{
namespace interface1
{

using DistributedType = Distributed<step1Local, DAAL_FPTYPE, gbt::regression::init::defaultDense>;
using ParameterType = gbt::regression::init::Parameter;

template<>
DistributedType::Distributed(size_t _localMaxBins)
{
    _par = new ParameterType(_localMaxBins);
    initialize();
}

template<>
DistributedType::Distributed(const DistributedType &other)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace init
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

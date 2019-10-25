/* file: gbt_regression_init_distr_step1_fpt_dispatcher.cpp */
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

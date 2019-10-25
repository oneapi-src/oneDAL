/* file: gbt_regression_train_dense_default_distr_step5_fpt_dispatcher.cpp */
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
//  Implementation of gradient boosted trees container.
//--
*/

#include "gbt_regression_train_container.h"


namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(gbt::regression::training::DistributedContainer, distributed, step5Local,  \
    DAAL_FPTYPE, gbt::regression::training::defaultDense)

namespace gbt
{
namespace regression
{
namespace training
{
namespace interface1
{

using DistributedType = Distributed<step5Local, DAAL_FPTYPE, gbt::regression::training::defaultDense>;

template <>
DistributedType::Distributed()
{
    ParameterType *par = new ParameterType();
    _par = par;
    initialize();
}

template <>
DistributedType::Distributed(const DistributedType &other) : input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

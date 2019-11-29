/* file: linear_regression_ne_model.cpp */
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
//  Implementation of the class defining the linear regression model
//--
*/

#include "linear_regression_ne_model_impl.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
using namespace daal::data_management;
using namespace daal::services;
__DAAL_REGISTER_SERIALIZATION_CLASS2(ModelNormEq, internal::ModelNormEqImpl, SERIALIZATION_LINEAR_REGRESSION_MODELNORMEQ_ID);

namespace internal
{
/**
* Initializes the linear regression model
*/
Status ModelNormEqInternal::initialize()
{
    Status s;
    DAAL_CHECK_STATUS(s, super::initialize());
    DAAL_CHECK_STATUS(s, this->setToZero(*_xtxTable));
    DAAL_CHECK_STATUS(s, this->setToZero(*_xtyTable));
    return s;
}

/**
 * Returns a Numeric table that contains partial sums X'*X
 * \return Numeric table that contains partial sums X'*X
 */
NumericTablePtr ModelNormEqInternal::getXTXTable()
{
    return _xtxTable;
}

/**
 * Returns a Numeric table that contains partial sums X'*Y
 * \return Numeric table that contains partial sums X'*Y
     */
NumericTablePtr ModelNormEqInternal::getXTYTable()
{
    return _xtyTable;
}

} // namespace internal
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

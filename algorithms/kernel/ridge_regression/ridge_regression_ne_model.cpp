/* file: ridge_regression_ne_model.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of the class defining the ridge regression model
//--
*/

#include "ridge_regression_ne_model_impl.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
using namespace daal::data_management;
using namespace daal::services;
__DAAL_REGISTER_SERIALIZATION_CLASS2(ModelNormEq, internal::ModelNormEqImpl, SERIALIZATION_RIDGE_REGRESSION_MODELNORMEQ_ID);

namespace internal
{
/**
 * Initializes the ridge regression model
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
NumericTablePtr ModelNormEqInternal::getXTXTable() { return _xtxTable; }

/**
 * Returns a Numeric table that contains partial sums X'*Y
 * \return Numeric table that contains partial sums X'*Y
 */
NumericTablePtr ModelNormEqInternal::getXTYTable() { return _xtyTable; }

} // namespace internal
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

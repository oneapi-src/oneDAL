/* file: linear_regression_qr_model.cpp */
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
//  Implementation of the class defining the linear regression model
//--
*/

#include "linear_regression_qr_model_impl.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(ModelQR, internal::ModelQRImpl, SERIALIZATION_LINEAR_REGRESSION_MODELQR_ID);

namespace internal
{
/**
 * Initializes the linear regression model
 */
Status ModelQRInternal::initialize()
{
    Status s;
    DAAL_CHECK_STATUS(s, super::initialize());
    DAAL_CHECK_STATUS(s, this->setToZero(*_rTable));
    DAAL_CHECK_STATUS(s, this->setToZero(*_qtyTable));
    return s;
}

/**
 * Returns a Numeric table that contains the R factor of QR decomposition
 * \return Numeric table that contains the R factor of QR decomposition
 */
NumericTablePtr ModelQRInternal::getRTable() { return _rTable; }

/**
 * Returns a Numeric table that contains Q'*Y, where Q is the factor of QR decomposition for a data block,
 * Y is the respective block of the matrix of responses
 * \return Numeric table that contains partial sums Q'*Y
 */
NumericTablePtr ModelQRInternal::getQTYTable() { return _qtyTable; }

} // namespace internal
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

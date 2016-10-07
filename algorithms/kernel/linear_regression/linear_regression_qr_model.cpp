/* file: linear_regression_qr_model.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "algorithms/linear_regression/linear_regression_qr_model.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace interface1
{

/**
 * Initializes the linear regression model
 */
void ModelQR::initialize()
{
    Model::initialize();
    this->setToZero(_rTable.get());
    this->setToZero(_qtyTable.get());
}

/**
 * Returns a Numeric table that contains the R factor of QR decomposition
 * \return Numeric table that contains the R factor of QR decomposition
 */
NumericTablePtr ModelQR::getRTable() { return _rTable; }

/**
 * Returns a Numeric table that contains Q'*Y, where Q is the factor of QR decomposition for a data block,
 * Y is the respective block of the matrix of responses
 * \return Numeric table that contains partial sums Q'*Y
 */
NumericTablePtr ModelQR::getQTYTable() { return _qtyTable; }


} // namespace interface1
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

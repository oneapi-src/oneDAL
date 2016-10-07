/* file: ridge_regression_ne_model.cpp */
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
//  Implementation of the class defining the ridge regression model
//--
*/

#include "algorithms/ridge_regression/ridge_regression_ne_model.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace interface1
{

/**
* Initializes the ridge regression model
*/
void ModelNormEq::initialize()
{
    Model::initialize();

    this->setToZero(_xtxTable.get());
    this->setToZero(_xtyTable.get());
}

/**
 * Returns a Numeric table that contains partial sums X'*X
 * \return Numeric table that contains partial sums X'*X
 */
NumericTablePtr ModelNormEq::getXTXTable() { return _xtxTable; }

/**
 * Returns a Numeric table that contains partial sums X'*Y
 * \return Numeric table that contains partial sums X'*Y
     */
NumericTablePtr ModelNormEq::getXTYTable() { return _xtyTable; }

} // namespace interface1
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

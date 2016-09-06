/* file: ridge_reg_ne_model.h */
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
// */

#ifndef __RIDGE_REG_NE_MODEL_
#define __RIDGE_REG_NE_MODEL_

#include "algorithms/ridge_regression/ridge_regression_ne_model.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{

/**
 * Constructs the ridge regression model for the normal equations method
 * \param[in] featnum Number of features in the training data set
 * \param[in] nrhs    Number of responses in the training data
 * \param[in] par     Parameters of ridge regression model-based training
 * \param[in] dummy   Dummy variable for the templated constructor
 */
template <typename modelFPType>
DAAL_EXPORT ModelNormEq::ModelNormEq(size_t featnum, size_t nrhs, const ridge_regression::Parameter &par, modelFPType dummy):
    Model(featnum, nrhs, par, dummy)
{
    size_t dimWithoutBeta = _coefdim;
    if(!_interceptFlag)
    {
        dimWithoutBeta--;
    };

    _xtxTable = data_management::NumericTablePtr(
        new data_management::HomogenNumericTable<modelFPType>(dimWithoutBeta, dimWithoutBeta,
                                                              data_management::NumericTableIface::doAllocate, 0));
    _xtyTable = data_management::NumericTablePtr(
        new data_management::HomogenNumericTable<modelFPType>(dimWithoutBeta, nrhs,
                                                              data_management::NumericTableIface::doAllocate, 0));
}

} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif

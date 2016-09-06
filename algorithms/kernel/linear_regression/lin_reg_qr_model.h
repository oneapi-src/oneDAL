/* file: lin_reg_qr_model.h */
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

#ifndef __LIN_REG_QR_MODEL_
#define __LIN_REG_QR_MODEL_

#include "algorithms/linear_regression/linear_regression_qr_model.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{


/**
 * Constructs the linear regression model for the QR decomposition-based method
 * \param[in] featnum Number of features in the training data set
 * \param[in] nrhs    Number of responses in the training data
 * \param[in] par     Parameters of linear regression model-based training
 * \param[in] dummy   Dummy variable for the templated constructor
 */
template <typename modelFPType>
DAAL_EXPORT ModelQR::ModelQR(size_t featnum, size_t nrhs, const linear_regression::Parameter &par, modelFPType dummy) : Model(featnum, nrhs, par, dummy)
{
    size_t dimWithoutBeta = _coefdim;
    if(!_interceptFlag)
    {
        dimWithoutBeta--;
    };

    _rTable = data_management::NumericTablePtr(
        new data_management::HomogenNumericTable<modelFPType>(dimWithoutBeta, dimWithoutBeta,
                                                              data_management::NumericTableIface::doAllocate, 0));

    _qtyTable = data_management::NumericTablePtr(
        new data_management::HomogenNumericTable<modelFPType>(dimWithoutBeta, nrhs,
                                                              data_management::NumericTableIface::doAllocate, 0));
}

} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif

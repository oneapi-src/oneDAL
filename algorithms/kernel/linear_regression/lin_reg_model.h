/* file: lin_reg_model.h */
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

#ifndef __LIN_REG_MODEL_
#define __LIN_REG_MODEL_

#include "algorithms/linear_regression/linear_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{

/**
 * Constructs the linear regression model
 * \param[in] featnum Number of features in the training data
 * \param[in] nrhs    Number of responses in the training data
 * \param[in] par     Linear regression parameters
 * \param[in] dummy   Dummy variable for the templated constructor
 */
template<typename modelFPType>
DAAL_EXPORT Model::Model(size_t featnum, size_t nrhs, const Parameter &par, modelFPType dummy) : daal::algorithms::Model()
{
    _coefdim = featnum + 1;

    _nrhs = nrhs;
    _interceptFlag = true;
    if(!par.interceptFlag)
    {
        _interceptFlag = false;
    }

    _beta = data_management::NumericTablePtr(
                new data_management::HomogenNumericTable<modelFPType>(_coefdim, _nrhs, data_management::NumericTableIface::doAllocate, 0));
}

} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif

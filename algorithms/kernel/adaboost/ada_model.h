/* file: ada_model.h */
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
//  Implementation of class defining Ada Boost model
//--
*/

#ifndef __ADA_MODEL_
#define __ADA_MODEL_

#include "algorithms/boosting/adaboost_model.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{

/**
 * Constructs the AdaBoost model
 * \tparam modelFPType  Data type to store AdaBoost model data, double or float
 * \param[in] dummy     Dummy variable for the templated constructor
 */
template <typename modelFPType>
DAAL_EXPORT Model::Model(modelFPType dummy) : boosting::Model()
{
    _alpha = data_management::NumericTablePtr(new data_management::HomogenNumericTable<modelFPType>(NULL, 1));
    _alpha->setNumberOfColumns(1);
}

} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif

/* file: logitboost_model_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of class defining LogitBoost model.
//--
*/

#include "algorithms/boosting/logitboost_model.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
/**
 *  Constructs the logitBoost %Model
 * \tparam modelFPType  Data type to store logitBoost model data, double or float
 * \param[in] dummy     Dummy variable for the templated constructor
 * \DAAL_DEPRECATED_USE{ Model::create }
 */
template <typename modelFPType>
DAAL_EXPORT Model::Model(size_t nFeatures, const Parameter * par, modelFPType dummy)
    : _nFeatures(nFeatures), _models(new data_management::DataCollection()), _nIterations(par->maxIterations)
{}

template DAAL_EXPORT Model::Model(size_t, const Parameter *, DAAL_FPTYPE);
} // namespace logitboost
} // namespace algorithms
} // namespace daal

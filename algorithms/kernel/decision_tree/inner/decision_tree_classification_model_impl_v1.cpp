/* file: decision_tree_classification_model_impl_v1.cpp */
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
//  Implementation of the class defining the Decision tree model
//--
*/

#include "decision_tree_classification_model_impl.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;

services::Status Parameter::check() const
{
    services::Status s;
    // Inherited.
    DAAL_CHECK_STATUS(s, daal::algorithms::classifier::interface1::Parameter::check());

    DAAL_CHECK_EX(minObservationsInLeafNodes >= 1, services::ErrorIncorrectParameter, services::ParameterName, minObservationsInLeafNodesStr());
    return s;
}

} // namespace interface1
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

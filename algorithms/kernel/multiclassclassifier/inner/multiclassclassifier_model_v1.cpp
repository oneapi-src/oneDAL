/* file: multiclassclassifier_model_v1.cpp */
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
//  Implementation of multi class classifier model.
//--
*/

#include "multi_class_classifier_model.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace interface1
{
services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, interface1::ParameterBase::check());
    DAAL_CHECK_EX((accuracyThreshold > 0) && (accuracyThreshold < 1), services::ErrorIncorrectParameter, services::ParameterName,
                  accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations, services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr());
    return s;
}
} // namespace interface1
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

/* file: svm_train_v1.cpp */
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
//  Implementation of svm algorithm and types methods.
//--
*/

#include "svm_train_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace interface1
{
services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::interface1::Parameter::check());
    if (C <= 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, cBoundStr()));
    }
    if (accuracyThreshold <= 0 || accuracyThreshold >= 1)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr()));
    }
    if (tau <= 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, tauStr()));
    }
    if (maxIterations == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
    }
    if (!kernel.get())
    {
        return services::Status(services::Error::create(services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, kernelFunctionStr()));
    }
    if (shrinkingStep == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, shrinkingStepStr()));
    }
    return s;
}

} // namespace interface1

} // namespace svm
} // namespace algorithms
} // namespace daal

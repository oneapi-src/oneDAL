/* file: implicit_als_model.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the class defining the implicit als model
//--
*/

#include "implicit_als_model.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
DAAL_EXPORT Model::Model()
{}

services::Status Parameter::check() const
{
    if(nFactors == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nFactorsStr()));
    }
    if(maxIterations == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
    }
    if(alpha < 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, alphaStr()));
    }
    if(lambda < 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, lambdaStr()));
    }
    if(preferenceThreshold < 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, preferenceThresholdStr()));
    }
    return services::Status();
}

}// namespace implicit_als
}// namespace algorithms
}// namespace daal

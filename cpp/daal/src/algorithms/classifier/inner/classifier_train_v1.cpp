/* file: classifier_train_v1.cpp */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of classifier training methods.
//--
*/

#include "algorithms/classifier/classifier_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/numeric_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace interface1
{
services::Status Parameter::check() const
{
    if (nClasses == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nClassesStr()));
    }
    return services::Status();
}
} // namespace interface1
} // namespace classifier
} // namespace algorithms
} // namespace daal

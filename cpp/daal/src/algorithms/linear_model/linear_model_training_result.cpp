/* file: linear_model_training_result.cpp */
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
//  Implementation of the class defining the result of the regression training algorithm
//--
*/

#include "services/daal_defines.h"
#include "algorithms/linear_model/linear_model_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace training
{
using namespace daal::data_management;
using namespace daal::services;

Result::Result(size_t nElements) : regression::training::Result(nElements) {}

linear_model::ModelPtr Result::get(ResultId id) const
{
    return linear_model::Model::cast(regression::training::Result::get(regression::training::ResultId(id)));
}

void Result::set(ResultId id, const linear_model::ModelPtr & value)
{
    regression::training::Result::set(regression::training::ResultId(id), value);
}

Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, regression::training::Result::check(input, par, method));

    const Input * in                   = static_cast<const Input *>(input);
    const linear_model::ModelPtr model = get(training::model);
    const size_t nFeatures             = in->get(data)->getNumberOfColumns();
    DAAL_CHECK_EX(model->getNumberOfFeatures() == nFeatures, ErrorIncorrectNumberOfFeatures, services::ArgumentName, modelStr())

    const size_t nBeta      = nFeatures + 1;
    const size_t nResponses = in->get(dependentVariables)->getNumberOfColumns();

    DAAL_CHECK_STATUS(s, linear_model::checkModel(model.get(), *par, nBeta, nResponses));

    return s;
}

} // namespace training
} // namespace linear_model
} // namespace algorithms
} // namespace daal

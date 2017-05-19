/* file: df_regression_training_result.cpp */
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
//  Implementation of decision forest algorithm classes.
//--
*/

#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_FOREST_REGRESSION_TRAINING_RESULT_ID);
Result::Result() : algorithms::regression::training::Result(lastResultNumericTableId + 1) {};

decision_forest::regression::ModelPtr Result::get(ResultId id) const
{
    return decision_forest::regression::Model::cast(
        algorithms::regression::training::Result::get(algorithms::regression::training::ResultId(id)));
}

void Result::set(ResultId id, const decision_forest::regression::ModelPtr &value)
{
    algorithms::regression::training::Result::set(algorithms::regression::training::ResultId(id), value);
}

NumericTablePtr Result::get(ResultNumericTableId id) const
{
    return NumericTable::cast(Argument::get(id));
}

void Result::set(ResultNumericTableId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, algorithms::regression::training::Result::check(input, par, method));
    const Input *in = static_cast<const Input *>(input);

    //TODO: check model
    const Parameter* algParameter = static_cast<const Parameter *>(par);
    if(algParameter->resultsToCompute & decision_forest::training::computeOutOfBagError)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(outOfBagError).get(), outOfBagErrorStr(), 0, 0, 1, 1));
    }
    if(algParameter->varImportance != decision_forest::training::none)
    {
        const decision_forest::regression::training::Input *algInput = static_cast<const decision_forest::regression::training::Input *>(input);
        const auto nFeatures = algInput->get(decision_forest::regression::training::data)->getNumberOfColumns();
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(variableImportance).get(), variableImportanceStr(), 0, 0, nFeatures, 1));
    }
    return s;
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

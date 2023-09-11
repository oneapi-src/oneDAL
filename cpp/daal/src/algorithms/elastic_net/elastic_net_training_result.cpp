/* file: elastic_net_training_result.cpp */
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
//  Implementation of elastic net algorithm classes.
//--
*/

#include "algorithms/elastic_net/elastic_net_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace elastic_net
{
namespace training
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ELASTIC_NET_TRAINING_RESULT_ID);
Result::Result() : linear_model::training::Result(lastResultNumericTableId + 1) {}

/**
 * Returns the result of elastic net model-based training
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
daal::algorithms::elastic_net::ModelPtr Result::get(ResultId id) const
{
    return elastic_net::Model::cast(linear_model::training::Result::get(linear_model::training::ResultId(id)));
}

/**
 * Sets the result of elastic net model-based training
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(ResultId id, const daal::algorithms::elastic_net::ModelPtr & value)
{
    linear_model::training::Result::set(linear_model::training::ResultId(id), value);
}

/**
* Returns the result of model-based prediction
* \param[in] id    Identifier of the result
* \return          Result that corresponds to the given identifier
*/
data_management::NumericTablePtr Result::get(OptionalResultNumericTableId id) const
{
    return NumericTable::cast(Argument::get(id));
}

/**
* Sets the result of model-based prediction
* \param[in] id      Identifier of the input object
* \param[in] value   %Input object
*/
void Result::set(OptionalResultNumericTableId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks the result of elastic net model-based training
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, linear_model::training::Result::check(input, par, method));

    const Input * in    = static_cast<const Input *>(input);
    const Parameter * p = static_cast<const Parameter *>(par);
    size_t nBeta        = in->getNumberOfFeatures() + 1;
    size_t nResponses   = in->getNumberOfDependentVariables();

    const elastic_net::ModelPtr model = get(training::model);

    if (p->optResultToCompute & computeGramMatrix)
        s |= data_management::checkNumericTable(get(gramMatrixId).get(), gramMatrixStr(), 0, 0, in->getNumberOfFeatures(), in->getNumberOfFeatures());

    s |= elastic_net::checkModel(model.get(), *par, nBeta, nResponses, method);
    return s;
}

} // namespace training
} // namespace elastic_net
} // namespace algorithms
} // namespace daal

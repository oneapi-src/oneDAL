/* file: logistic_regression_predict_types.cpp */
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
//  Implementation of logistic regression algorithm classes.
//--
*/

#include "algorithms/logistic_regression/logistic_regression_predict_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"
#include "logistic_regression_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_LOGISTIC_REGRESSION_PREDICTION_RESULT_ID);

/**
 * Returns an input object for making logistic regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return algorithms::classifier::prediction::Input::get(id);
}

/**
 * Returns an input object for making logistic regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
logistic_regression::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return staticPointerCast<logistic_regression::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for making logistic regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const NumericTablePtr &value)
{
    algorithms::classifier::prediction::Input::set(id, value);
}

/**
 * Sets an input object for making logistic regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::ModelInputId id, const logistic_regression::ModelPtr &value)
{
    algorithms::classifier::prediction::Input::set(id, value);
}

/**
 * Checks an input object for making logistic regression model-based prediction
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::classifier::prediction::Input::check(parameter, method));
    ModelPtr m = get(classifier::prediction::model);
    const daal::algorithms::logistic_regression::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::logistic_regression::internal::ModelImpl*>(m.get());
    DAAL_ASSERT(pModel);
    const daal::algorithms::logistic_regression::prediction::Parameter* pPrm = static_cast<const daal::algorithms::logistic_regression::prediction::Parameter*>(parameter);
    const size_t nBetaPerClass = get(classifier::prediction::data)->getNumberOfColumns() + 1;
    return checkNumericTable(pModel->getBeta().get(), betaStr(), 0, 0, nBetaPerClass, pPrm->nClasses == 2 ? 1 : pPrm->nClasses);
}

Result::Result() : algorithms::classifier::prediction::Result(lastResultNumericTableId + 1) {}

NumericTablePtr Result::get(classifier::prediction::ResultId id) const
{
    return classifier::prediction::Result::get(id);
}

void Result::set(classifier::prediction::ResultId id, const NumericTablePtr &value)
{
    classifier::prediction::Result::set(id, value);
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
    const size_t nRows = (static_cast<const Input *>(input))->getNumberOfRows();
    const Parameter* prm = static_cast<const Parameter*>(par);
    Status s;
    const daal::algorithms::logistic_regression::prediction::Parameter* pPrm = static_cast<const daal::algorithms::logistic_regression::prediction::Parameter*>(par);
    const size_t nProb = (pPrm->nClasses == 2 ? 1 : pPrm->nClasses);
    if(prm->resultsToCompute & computeClassesLabels)
        s |= data_management::checkNumericTable(get(classifier::prediction::prediction).get(), probabilitiesStr(), packed_mask, 0, 1, nRows);
    if(prm->resultsToCompute & computeClassesProbabilities)
        s |= data_management::checkNumericTable(get(probabilities).get(), probabilitiesStr(), packed_mask, 0, nProb, nRows);
    if(prm->resultsToCompute & computeClassesLogProbabilities)
        s |= data_management::checkNumericTable(get(logProbabilities).get(), logProbabilitiesStr(), packed_mask, 0, nProb, nRows);
    return s;
}

} // namespace interface1
} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

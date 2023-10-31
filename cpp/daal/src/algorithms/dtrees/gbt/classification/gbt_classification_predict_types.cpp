/* file: gbt_classification_predict_types.cpp */
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
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_classification_predict_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{
namespace interface1
{
/**
 * Returns an input object for making gradient boosted trees model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return algorithms::classifier::prediction::Input::get(id);
}

/**
 * Returns an input object for making gradient boosted trees model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
gbt::classification::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return staticPointerCast<gbt::classification::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for making gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const NumericTablePtr & value)
{
    algorithms::classifier::prediction::Input::set(id, value);
}

/**
 * Sets an input object for making gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::ModelInputId id, const gbt::classification::ModelPtr & value)
{
    algorithms::classifier::prediction::Input::set(id, value);
}

/**
 * Checks an input object for making gradient boosted trees model-based prediction
 */
services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::classifier::prediction::Input::check(parameter, method));
    ModelPtr m = get(classifier::prediction::model);
    const daal::algorithms::gbt::classification::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::gbt::classification::internal::ModelImpl *>(m.get());
    DAAL_ASSERT(pModel);
    DAAL_CHECK(pModel->getNumberOfTrees(), services::ErrorNullModel);

    size_t nClasses = 0, nIterations = 0;

    const gbt::classification::prediction::interface2::Parameter * pPrm =
        dynamic_cast<const gbt::classification::prediction::interface2::Parameter *>(parameter);
    if (pPrm)
    {
        nClasses    = pPrm->nClasses;
        nIterations = pPrm->nIterations;
    }
    else
    {
        return services::ErrorNullParameterNotSupported;
    }

    auto maxNIterations = pModel->getNumberOfTrees();
    if (nClasses > 2) maxNIterations /= nClasses;
    DAAL_CHECK((nClasses < 3) || (pModel->getNumberOfTrees() % nClasses == 0), services::ErrorGbtIncorrectNumberOfTrees);
    DAAL_CHECK((nIterations == 0) || (nIterations <= maxNIterations), services::ErrorGbtPredictIncorrectNumberOfIterations);

    const bool predictContribs     = pPrm->resultsToCompute & shapContributions;
    const bool predictInteractions = pPrm->resultsToCompute & shapInteractions;
    DAAL_CHECK(!(predictContribs || predictInteractions), services::ErrorMethodNotImplemented);

    return s;
}

} // namespace interface1
} // namespace prediction
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal

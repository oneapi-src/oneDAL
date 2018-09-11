/* file: gbt_classification_predict_types.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_classification_predict_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"
#include "gbt_classification_model_impl.h"

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
void Input::set(classifier::prediction::NumericTableInputId id, const NumericTablePtr &value)
{
    algorithms::classifier::prediction::Input::set(id, value);
}

/**
 * Sets an input object for making gradient boosted trees model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::ModelInputId id, const gbt::classification::ModelPtr &value)
{
    algorithms::classifier::prediction::Input::set(id, value);
}

/**
 * Checks an input object for making gradient boosted trees model-based prediction
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::classifier::prediction::Input::check(parameter, method));
    ModelPtr m = get(classifier::prediction::model);
    const daal::algorithms::gbt::classification::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::gbt::classification::internal::ModelImpl*>(m.get());
    DAAL_ASSERT(pModel);
    DAAL_CHECK(pModel->numberOfTrees(), services::ErrorNullModel);
    const Parameter* pPrm = static_cast<const Parameter*>(parameter);
    DAAL_CHECK((pPrm->nClasses < 3) || (pModel->numberOfTrees() % pPrm->nClasses == 0), services::ErrorGbtIncorrectNumberOfTrees);
    auto maxNIterations = pModel->numberOfTrees();
    if(pPrm->nClasses > 2)
        maxNIterations /= pPrm->nClasses;
    DAAL_CHECK((pPrm->nIterations == 0) || (pPrm->nIterations <= maxNIterations), services::ErrorGbtPredictIncorrectNumberOfIterations);
    return s;
}

} // namespace interface1
} // namespace prediction
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal

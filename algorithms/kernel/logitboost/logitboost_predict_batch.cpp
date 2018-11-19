/* file: logitboost_predict_batch.cpp */
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
//  Implementation of the interface for LogitBoost model-based prediction
//--
*/

#include "algorithms/boosting/logitboost_predict_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace prediction
{
namespace interface1
{

/**
 * Returns the input Numeric Table object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input NumericTable object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns the input Model object in the prediction stage of the LogitBoost algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          %Input object that corresponds to the given identifier
 */
logitboost::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return staticPointerCast<logitboost::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the LogitBoost algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const logitboost::ModelPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s = classifier::prediction::Input::check(parameter, method);
    if(!s) return s;

    logitboost::ModelPtr m =
        staticPointerCast<logitboost::Model, classifier::Model>(get(classifier::prediction::model));
    DAAL_CHECK(m->getNumberOfWeakLearners() > 0, ErrorModelNotFullInitialized);
    return s;
}


} // namespace interface1
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

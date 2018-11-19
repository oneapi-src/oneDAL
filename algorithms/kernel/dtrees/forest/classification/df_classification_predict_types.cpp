/* file: df_classification_predict_types.cpp */
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
//  Implementation of decision forest algorithm classes.
//--
*/

#include "algorithms/decision_forest/decision_forest_classification_predict_types.h"
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
namespace classification
{
namespace prediction
{
namespace interface1
{

/** Default constructor */
Input::Input() : classifier::prediction::Input() {}

/**
 * Returns an input object for making decision forest model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Returns an input object for making decision forest model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
decision_forest::classification::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return staticPointerCast<decision_forest::classification::Model, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for making decision forest model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Sets an input object for making decision forest model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(classifier::prediction::ModelInputId id, const decision_forest::classification::ModelPtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks an input object for making decision forest model-based prediction
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfInputNumericTables);
    NumericTablePtr dataTable = get(classifier::prediction::data);

    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const decision_forest::classification::ModelPtr m = get(classifier::prediction::model);
    if(!m.get())
        s.add(ErrorNullModel);
    //TODO: check input model
    return s;
}

} // namespace interface1
} // namespace prediction
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

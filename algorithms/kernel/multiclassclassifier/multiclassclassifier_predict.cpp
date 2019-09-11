/* file: multiclassclassifier_predict.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of multi class classifier algorithm and types methods.
//--
*/

#include "multi_class_classifier_predict_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
namespace interface1
{
Input::Input() {}
Input::Input(const Input& other) : classifier::prediction::Input(other){}

/**
 * Returns the input Numeric Table object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input NumericTable object
 * \return          Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns the input Model object in the prediction stage of the Multi-class classifier algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          Input object that corresponds to the given identifier
 */
multi_class_classifier::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<multi_class_classifier::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the Multi-class classifier algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const multi_class_classifier::ModelPtr &ptr)
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
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::prediction::Input::check(parameter, method));
    const Parameter *par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK_EX(par->prediction.get(), services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, predictionStr());

    multi_class_classifier::ModelPtr m = get(classifier::prediction::model);
    DAAL_CHECK(m->getNumberOfTwoClassClassifierModels(), services::ErrorModelNotFullInitialized);
    return s;
}

}// namespace interface1
}// namespace prediction
}// namespace multi_class_classifier
}// namespace algorithms
}// namespace daal

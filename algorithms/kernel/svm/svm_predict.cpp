/* file: svm_predict.cpp */
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
//  Implementation of svm algorithm and types methods.
//--
*/

#include "svm_predict_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace interface1
{
Input::Input() : classifier::prediction::Input() {}
Input::Input(const Input& other) : classifier::prediction::Input(other){}

/**
 * Returns the input Numeric Table object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input NumericTable object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(classifier::prediction::NumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns the input Model object in the prediction stage of the SVM algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          %Input object that corresponds to the given identifier
 */
svm::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<svm::Model, data_management::SerializationIface>(Argument::get(id));
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
 * Sets the input Model object in the prediction stage of the SVM algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const svm::ModelPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the correctness of the input object
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::prediction::Input::check(parameter, method));

    svm::ModelPtr m =
        services::staticPointerCast<svm::Model, classifier::Model>(get(classifier::prediction::model));

    s = data_management::checkNumericTable(m->getSupportVectors().get(), supportVectorsStr());
    if(!s)
        return Status(services::Error::create(services::ErrorModelNotFullInitialized, services::ArgumentName, supportVectorsStr()));

    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(classifier::prediction::data).get(), dataStr(),
                                                            0, 0, m->getSupportVectors()->getNumberOfColumns()));
    s |= data_management::checkNumericTable(m->getClassificationCoefficients().get(), classificationCoefficientsStr());
    if(!s)
        return Status(services::Error::create(services::ErrorModelNotFullInitialized, services::ArgumentName, classificationCoefficientsStr()));
    return s;
}

}// namespace interface1
}// namespace prediction
}// namespace svm
}// namespace algorithms
}// namespace daal

/* file: naivebayes_predict_input.cpp */
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
//  Implementation of multinomial naive bayes algorithm and types methods.
//--
*/

#include "multinomial_naive_bayes_predict_types.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
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
 * Returns the input Model object in the prediction stage of the multinomial naive Bayes algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          Input object that corresponds to the given identifier
 */
multinomial_naive_bayes::ModelPtr Input::get(classifier::prediction::ModelInputId id) const
{
    return services::staticPointerCast<multinomial_naive_bayes::Model, data_management::SerializationIface>(Argument::get(id));
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
 * Sets the input Model object in the prediction stage of the multinomial naive Bayes algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(classifier::prediction::ModelInputId id, const multinomial_naive_bayes::ModelPtr &ptr)
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

    if (method == fastCSR)
    {
        int expectedLayouts = (int)NumericTableIface::csrArray;
        DAAL_CHECK_STATUS(s, checkNumericTable(get(classifier::prediction::data).get(), dataStr(), 0, expectedLayouts));
    }

    ModelPtr inputModel = get(classifier::prediction::model);
    DAAL_CHECK(inputModel, ErrorNullModel);

    DAAL_CHECK(inputModel->getLogP(), services::ErrorModelNotFullInitialized);
    DAAL_CHECK(inputModel->getLogTheta(), services::ErrorModelNotFullInitialized);
    DAAL_CHECK(inputModel->getAuxTable(), services::ErrorModelNotFullInitialized);

    const multinomial_naive_bayes::Parameter *algPar = static_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    size_t nClasses = algPar->nClasses;

    DAAL_CHECK(inputModel->getLogP()->getNumberOfRows() == nClasses, ErrorNaiveBayesIncorrectModel);

    return s;
}

}// namespace interface1
}// namespace prediction
}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal

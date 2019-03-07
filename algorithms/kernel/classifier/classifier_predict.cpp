/* file: classifier_predict.cpp */
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
//  Implementation of classifier prediction methods.
//--
*/

#include "classifier_predict_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_CLASSIFIER_PREDICTION_RESULT_ID);

InputIface::InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}

Input::Input() : InputIface(lastModelInputId + 1) {}

/**
 * Returns the number of rows in the input data set
 * \return Number of rows in the input data set
 */
size_t Input::getNumberOfRows() const
{
    size_t nRows = 0;
    data_management::NumericTablePtr dataTable = get(data);
    if(dataTable)
    {
        nRows = dataTable->getNumberOfRows();
    }
    else
    {
        /* ERROR */;
    }
    return nRows;
}

/**
 * Returns the input Numeric Table object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input NumericTable object
 * \return          Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(NumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns the input Model object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input Model object
 * \return          Input object that corresponds to the given identifier
 */
classifier::ModelPtr Input::get(ModelInputId id) const
{
    return services::staticPointerCast<classifier::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the input NumericTable object in the prediction stage of the classification algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(NumericTableInputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets the input Model object in the prediction stage of the classifier algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void Input::set(ModelInputId id, const ModelPtr &ptr)
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
    return checkImpl(parameter);
}

services::Status Input::checkImpl(const daal::algorithms::Parameter *parameter) const
{
    services::Status s;
    if(parameter != NULL)
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        DAAL_CHECK_EX(algParameter->nClasses > 1, services::ErrorIncorrectParameter, services::ParameterName, nClassesStr());
    }

    data_management::NumericTablePtr dataTable = get(data);
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(dataTable.get(), dataStr()));

    classifier::ModelPtr m = get(model);
    DAAL_CHECK(m, services::ErrorNullModel);

    const size_t trainingDataFeatures = m->getNFeatures();
    DAAL_CHECK(trainingDataFeatures, services::ErrorModelNotFullInitialized);
    const size_t predictionDataFeatures = dataTable->getNumberOfColumns();
    DAAL_CHECK_EX(trainingDataFeatures == predictionDataFeatures, services::ErrorIncorrectNumberOfColumns, services::ArgumentName, dataStr());
    return s;
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}
Result::Result(size_t n) : daal::algorithms::Result(n) {}

/**
 * Returns the prediction result of the classification algorithm
 * \param[in] id   Identifier of the prediction result, \ref ResultId
 * \return         Prediction result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the prediction result of the classification algorithm
 * \param[in] id    Identifier of the prediction result, \ref ResultId
 * \param[in] value Pointer to the prediction result
 */
void Result::set(ResultId id, const data_management::NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the the input object
 * \param[in] parameter Pointer to the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
           int method) const
{
    return checkImpl(input, parameter);
}

services::Status Result::checkImpl(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter) const
{
    size_t nRows = (static_cast<const InputIface *>(input))->getNumberOfRows();
    data_management::NumericTablePtr resTable = get(prediction);

    const int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray;
    return data_management::checkNumericTable(resTable.get(), predictionStr(), unexpectedLayouts, 0, 1, nRows);
}

}
}
}
}
}

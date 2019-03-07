/* file: classifier_train.cpp */
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
//  Implementation of classifier training methods.
//--
*/

#include "classifier_training_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{

namespace interface1
{
services::Status Parameter::check() const
{
    if(nClasses == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nClassesStr()));
    }
    return services::Status();
}
}

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_CLASSIFIER_TRAINING_RESULT_ID);

InputIface::InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
Input::Input(size_t nElements) : InputIface(nElements) {}

size_t Input::getNumberOfFeatures() const
{
    return get(classifier::training::data)->getNumberOfColumns();
}

/**
 * Returns the input object in the training stage of the classification algorithm
 * \param[in] id   Identifier of the input object, \ref InputId
 * \return         Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return data_management::NumericTable::cast(Argument::get(id));
}

/**
 * Sets the input object in the training stage of the classification algorithm
 * \param[in] id    Identifier of the input object, \ref InputId
 * \param[in] value Pointer to the input object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &value)
{
    Argument::set(id, value);
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
    if (parameter != NULL)
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        DAAL_CHECK_EX(algParameter->nClasses > 1, services::ErrorIncorrectParameter, services::ParameterName, nClassesStr());
    }

    data_management::NumericTablePtr dataTable = get(data);
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(dataTable.get(), dataStr()));

    const size_t nRows = dataTable->getNumberOfRows();
    data_management::NumericTablePtr labelsTable = get(labels);
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(labelsTable.get(), labelsStr(), 0, 0, 1, nRows));

    data_management::NumericTablePtr weightsTable = get(weights);
    if(weightsTable)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(weightsTable.get(), weightsStr(), 0, 0, 1, nRows));
    }
    return s;
}

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}
Result::Result(const size_t n) : daal::algorithms::Result(n){}

/**
 * Returns the model trained with the classification algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \return          Model trained with the classification algorithm
 */
classifier::ModelPtr Result::get(ResultId id) const
{
    return classifier::Model::cast(Argument::get(id));
}

/**
 * Sets the result of the training stage of the classification algorithm
 * \param[in] id    Identifier of the result, \ref ResultId
 * \param[in] value Pointer to the training result
 */
void Result::set(ResultId id, const classifier::ModelPtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
           int method) const
{
    return checkImpl(input, parameter);
}

services::Status Result::checkImpl(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter) const
{
    daal::algorithms::classifier::ModelPtr m = get(model);
    DAAL_CHECK(m, services::ErrorNullModel);
    return services::Status();
}

}
}
}
}
}

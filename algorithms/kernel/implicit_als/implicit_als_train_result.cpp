/* file: implicit_als_train_result.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "implicit_als_training_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{

namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_IMPLICIT_ALS_MODEL_ID);
}

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_IMPLICIT_ALS_TRAINING_RESULT_ID);

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
daal::algorithms::implicit_als::ModelPtr Result::get(ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::implicit_als::Model,
           data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const daal::algorithms::implicit_als::ModelPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the implicit ALS training algorithm
 * \param[in] input       %Input object for the algorithm
 * \param[in] parameter   %Parameter of the algorithm
 * \param[in] method      Computation method of the algorithm
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    NumericTablePtr dataTable = algInput->get(data);
    const size_t nUsers = dataTable->getNumberOfRows();
    const size_t nItems = dataTable->getNumberOfColumns();

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    const size_t nFactors = algParameter->nFactors;

    ModelPtr trainedModel = get(model);
    DAAL_CHECK(trainedModel, ErrorNullModel);

    const int unexpectedLayouts = (int)packed_mask;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(trainedModel->getUsersFactors().get(), usersFactorsStr(), unexpectedLayouts, 0, nFactors, nUsers));
    DAAL_CHECK_STATUS(s, checkNumericTable(trainedModel->getItemsFactors().get(), itemsFactorsStr(), unexpectedLayouts, 0, nFactors, nItems));
    return s;
}

}// namespace interface1
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal

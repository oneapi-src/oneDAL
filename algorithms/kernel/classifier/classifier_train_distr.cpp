/* file: classifier_train_distr.cpp */
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
//  Implementation of classifier training methods.
//--
*/

#include "classifier_training_types.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_CLASSIFIER_TRAINING_PARTIAL_RESULT_ID);

PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {};

/**
 * Returns the partial result in the training stage of the classification algorithm
 * \param[in] id   Identifier of the partial result, \ref PartialResultId
 * \return         Partial result that corresponds to the given identifier
 */
classifier::ModelPtr PartialResult::get(PartialResultId id) const
{
    return services::staticPointerCast<classifier::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the partial result in the training stage of the classification algorithm
 * \param[in] id    Identifier of the partial result, \ref PartialResultId
 * \param[in] value Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const daal::algorithms::classifier::ModelPtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks the correctness of the PartialResult object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    return checkImpl(input, parameter);
}

services::Status PartialResult::checkImpl(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter) const
{
    daal::algorithms::classifier::ModelPtr m = get(partialModel);
    DAAL_CHECK(m, services::ErrorNullModel);
    return services::Status();
}

}
}
}
}
}

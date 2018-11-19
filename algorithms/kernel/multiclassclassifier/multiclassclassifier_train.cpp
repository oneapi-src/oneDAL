/* file: multiclassclassifier_train.cpp */
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
//  Implementation of multi class classifier algorithm and types methods.
//--
*/

#include "multi_class_classifier_train_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_MULTICLASS_CLASSIFIER_RESULT_ID);
Result::Result() {}

/**
 * Returns the model trained with the Multi class classifier algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the Multi class classifier algorithm
 */
daal::algorithms::multi_class_classifier::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::multi_class_classifier::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Checks the correctness of the Result object
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, checkImpl(input, parameter));

    const Parameter *par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK_EX(par->training.get(), services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, trainingStr());
    daal::algorithms::multi_class_classifier::ModelPtr m = get(classifier::training::model);
    if(m->getNumberOfTwoClassClassifierModels() == 0)
    {
        return services::Status(services::ErrorModelNotFullInitialized);
    }
    if(m->getNumberOfTwoClassClassifierModels() != par->nClasses * (par->nClasses - 1) / 2)
    {
        return services::Status(services::ErrorModelNotFullInitialized);
    }
    return s;
}

}// namespace interface1
}// namespace training
}// namespace multi_class_classifier
}// namespace algorithms
}// namespace daal

/* file: svm_train.cpp */
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
//  Implementation of svm algorithm and types methods.
//--
*/

#include "svm_train_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svm
{

namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_SVM_MODEL_ID);

services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());
    if(C <= 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, cBoundStr()));
    }
    if(accuracyThreshold <= 0 || accuracyThreshold >= 1)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr()));
    }
    if(tau <= 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, tauStr()));
    }
    if(maxIterations == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
    }
    if(!kernel.get())
    {
        return services::Status(services::Error::create(services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, kernelFunctionStr()));
    }
    if(shrinkingStep == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, shrinkingStepStr()));
    }
    return s;
}

}

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SVM_TRAINING_RESULT_ID);
Result::Result() : classifier::training::Result() {}

/**
 * Returns the model trained with the SVM algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the SVM algorithm
 */
daal::algorithms::svm::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::svm::Model, data_management::SerializationIface>(Argument::get(id));
}

Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::training::Result::check(input, parameter, method));
    daal::algorithms::svm::ModelPtr m = get(classifier::training::model);
    if(!m->getSupportVectors())
        s.add(services::Error::create(ErrorModelNotFullInitialized, services::ArgumentName, supportVectorsStr()));
    if(!m->getClassificationCoefficients())
        s.add(services::Error::create(ErrorModelNotFullInitialized, services::ArgumentName, classificationCoefficientsStr()));
    return s;
}

}// namespace interface1
}// namespace training
}// namespace svm
}// namespace algorithms
}// namespace daal

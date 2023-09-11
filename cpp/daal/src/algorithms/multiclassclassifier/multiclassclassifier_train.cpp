/* file: multiclassclassifier_train.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of multi class classifier algorithm and types methods.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_train_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
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
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, checkImpl(input, parameter));

    size_t nClasses;
    {
        auto par2 = dynamic_cast<const multi_class_classifier::Parameter *>(parameter);
        if (par2)
        {
            nClasses = par2->nClasses;
            DAAL_CHECK_EX(par2->training.get(), services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, trainingStr());
        }

        if (par2 == nullptr) return services::Status(services::ErrorNullParameterNotSupported);
    }
    daal::algorithms::multi_class_classifier::ModelPtr m = get(classifier::training::model);
    if (m->getNumberOfTwoClassClassifierModels() == 0)
    {
        return services::Status(services::ErrorModelNotFullInitialized);
    }
    if (m->getNumberOfTwoClassClassifierModels() != nClasses * (nClasses - 1) / 2)
    {
        return services::Status(services::ErrorModelNotFullInitialized);
    }
    return s;
}

} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

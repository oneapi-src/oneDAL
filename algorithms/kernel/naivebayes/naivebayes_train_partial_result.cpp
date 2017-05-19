/* file: naivebayes_train_partial_result.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of multinomial naive bayes algorithm and types methods.
//--
*/

#include "multinomial_naive_bayes_training_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{

namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialModel, SERIALIZATION_NAIVE_BAYES_PARTIALMODEL_ID);
}

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_NAIVE_BAYES_PARTIAL_RESULT_ID);

PartialResult::PartialResult() {}

/**
 * Returns the partial model trained with the classification algorithm
 * \param[in] id    Identifier of the partial model, \ref classifier::training::PartialResultId
 * \return          Model trained with the classification algorithm
 */
multinomial_naive_bayes::PartialModelPtr PartialResult::get(classifier::training::PartialResultId id) const
{
    return services::staticPointerCast<multinomial_naive_bayes::PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
* Returns number of columns in the naive Bayes partial result
* \return Number of columns in the partial result
*/
size_t PartialResult::getNumberOfFeatures() const
{
    PartialModelPtr ntPtr =
        services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(classifier::training::partialModel));
    return ntPtr ? ntPtr->getNFeatures() : 0;
}

/**
 * Checks partial result of the naive Bayes training algorithm
 * \param[in] input      Algorithm %input object
 * \param[in] parameter  Algorithm %parameter
 * \param[in] method     Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, checkImpl(input, parameter));

    PartialModelPtr presModel = get(classifier::training::partialModel);
    DAAL_CHECK(presModel.get(), ErrorNullModel);

    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);
    const classifier::Parameter *algPar = static_cast<const classifier::Parameter *>(parameter);

    size_t nFeatures = algInput->getNumberOfFeatures();
    size_t nClasses = algPar->nClasses;

    if(presModel->getClassSize()->getNumberOfColumns() != 1) return services::Status(services::ErrorIncorrectSizeOfModel);
    if(presModel->getClassSize()->getNumberOfRows() != nClasses) return services::Status(services::ErrorIncorrectSizeOfModel);

    if(presModel->getClassGroupSum()->getNumberOfColumns() != nFeatures) return services::Status(services::ErrorIncorrectSizeOfModel);
    if(presModel->getClassGroupSum()->getNumberOfRows() != nClasses) return services::Status(services::ErrorIncorrectSizeOfModel);
    return s;
}

/**
* Checks partial result of the naive Bayes training algorithm
* \param[in] parameter  Algorithm %parameter
* \param[in] method     Computation method
*/
services::Status PartialResult::check(const daal::algorithms::Parameter *parameter, int method)  const
{
    PartialModelPtr presModel = get(classifier::training::partialModel);

    const classifier::Parameter *algPar = static_cast<const classifier::Parameter *>(parameter);

    size_t nFeatures = getNumberOfFeatures();
    size_t nClasses = algPar->nClasses;

    if(presModel->getClassSize()->getNumberOfColumns() != 1) return services::Status(services::ErrorIncorrectSizeOfModel);
    if(presModel->getClassSize()->getNumberOfRows() != nClasses) return services::Status(services::ErrorIncorrectSizeOfModel);

    if(presModel->getClassGroupSum()->getNumberOfColumns() != nFeatures) return services::Status(services::ErrorIncorrectSizeOfModel);
    if(presModel->getClassGroupSum()->getNumberOfRows() != nClasses) return services::Status(services::ErrorIncorrectSizeOfModel);
    return services::Status();
}

}// namespace interface1
}// namespace training
}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal

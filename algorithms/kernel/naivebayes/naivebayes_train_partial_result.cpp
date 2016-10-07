/* file: naivebayes_train_partial_result.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
namespace interface1
{
PartialResult::PartialResult() {}

/**
 * Returns the partial model trained with the classification algorithm
 * \param[in] id    Identifier of the partial model, \ref classifier::training::PartialResultId
 * \return          Model trained with the classification algorithm
 */
services::SharedPtr<multinomial_naive_bayes::PartialModel> PartialResult::get(classifier::training::PartialResultId id) const
{
    return services::staticPointerCast<multinomial_naive_bayes::PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Allocates memory for storing partial results of the naive Bayes training algorithm
 * \param[in] input        Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
void PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);
    Parameter *algPar = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    size_t nFeatures = algInput->getNumberOfFeatures();
    algorithmFPType dummy = 0;
    set(classifier::training::partialModel,
        services::SharedPtr<classifier::Model>(new PartialModel(nFeatures, *algPar, dummy)));
}

/**
* Returns number of columns in the naive Bayes partial result
* \return Number of columns in the partial result
*/
size_t PartialResult::getNumberOfFeatures() const
{
    services::SharedPtr<PartialModel> ntPtr =
        services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(classifier::training::partialModel));
    if(ntPtr)
    {
        return ntPtr->getNFeatures();
    }
    else
    {
        this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable);
        return 0;
    }
}

/**
 * Checks partial result of the naive Bayes training algorithm
 * \param[in] input      Algorithm %input object
 * \param[in] parameter  Algorithm %parameter
 * \param[in] method     Computation method
 */
void PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    checkImpl(input, parameter);
    if (!this->_errors->isEmpty()) { return; }

    services::SharedPtr<PartialModel> presModel = get(classifier::training::partialModel);

    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);
    const classifier::Parameter *algPar = static_cast<const classifier::Parameter *>(parameter);

    size_t nFeatures = algInput->getNumberOfFeatures();
    size_t nClasses = algPar->nClasses;

    if(presModel->getClassSize()->getNumberOfColumns() != 1) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return;}
    if(presModel->getClassSize()->getNumberOfRows() != nClasses) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return;}

    if(presModel->getClassGroupSum()->getNumberOfColumns() != nFeatures) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return;}
    if(presModel->getClassGroupSum()->getNumberOfRows() != nClasses) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return;}
}

/**
* Checks partial result of the naive Bayes training algorithm
* \param[in] parameter  Algorithm %parameter
* \param[in] method     Computation method
*/
void PartialResult::check(const daal::algorithms::Parameter *parameter, int method)  const
{
    if (!this->_errors->isEmpty()) { return; }

    services::SharedPtr<PartialModel> presModel = get(classifier::training::partialModel);

    const classifier::Parameter *algPar = static_cast<const classifier::Parameter *>(parameter);

    size_t nFeatures = getNumberOfFeatures();
    size_t nClasses = algPar->nClasses;

    if(presModel->getClassSize()->getNumberOfColumns() != 1) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return;}
    if(presModel->getClassSize()->getNumberOfRows() != nClasses) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return;}

    if(presModel->getClassGroupSum()->getNumberOfColumns() != nFeatures) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return;}
    if(presModel->getClassGroupSum()->getNumberOfRows() != nClasses) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return;}
}

}// namespace interface1
}// namespace training
}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal

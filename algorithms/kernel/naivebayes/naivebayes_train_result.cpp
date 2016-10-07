/* file: naivebayes_train_result.cpp */
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
Result::Result() {}

/**
 * Returns the model trained with the naive Bayes training algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the classification algorithm
 */
services::SharedPtr<multinomial_naive_bayes::Model> Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<multinomial_naive_bayes::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
* Checks the correctness of Result object
* \param[in] partialResult Pointer to the partial results structure
* \param[in] parameter     Parameter of the algorithm
* \param[in] method        Computation method
*/
void Result::check(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, int method) const
{
    const PartialResult *pres = static_cast<const PartialResult *>(partialResult);
    size_t nFeatures = pres->getNumberOfFeatures();
    services::SharedPtr<Model> resModel = get(classifier::training::model);

    const classifier::Parameter *algPar = static_cast<const classifier::Parameter *>(parameter);

    size_t nClasses = algPar->nClasses;

    if(resModel->getLogP()->getNumberOfColumns() != 1) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return; }
    if(resModel->getLogP()->getNumberOfRows() != nClasses) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return; }

    if(resModel->getLogTheta()->getNumberOfColumns() != nFeatures) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return; }
    if(resModel->getLogTheta()->getNumberOfRows() != nClasses) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return; }
}

/**
 * Checks the final result of the naive Bayes training algorithm
 * \param[in] input      %Input of algorithm
 * \param[in] parameter  %Parameter of algorithm
 * \param[in] method     Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    checkImpl(input, parameter);
    if (!this->_errors->isEmpty()) { return; }
    services::SharedPtr<Model> resModel = get(classifier::training::model);

    const classifier::training::InputIface *algInput = static_cast<const classifier::training::InputIface *>(input);
    const classifier::Parameter *algPar = static_cast<const classifier::Parameter *>(parameter);

    size_t nFeatures = algInput->getNumberOfFeatures();
    size_t nClasses = algPar->nClasses;

    if(resModel->getLogP()->getNumberOfColumns() != 1) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return; }
    if(resModel->getLogP()->getNumberOfRows() != nClasses) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return; }

    if(resModel->getLogTheta()->getNumberOfColumns() != nFeatures) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return; }
    if(resModel->getLogTheta()->getNumberOfRows() != nClasses) {this->_errors->add(services::ErrorIncorrectSizeOfModel); return; }
}

}// namespace interface1
}// namespace training
}// namespace multinomial_naive_bayes
}// namespace algorithms
}// namespace daal

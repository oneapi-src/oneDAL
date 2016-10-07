/* file: svm_train.cpp */
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
//  Implementation of svm algorithm and types methods.
//--
*/

#include "svm_train_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface1
{
Result::Result() : classifier::training::Result() {}

/**
 * Returns the model trained with the SVM algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the SVM algorithm
 */
services::SharedPtr<daal::algorithms::svm::Model> Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::svm::Model, data_management::SerializationIface>(Argument::get(id));
}

void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    classifier::training::Result::check(input, parameter, method);
    if(this->_errors->size() != 0) { return; }
    services::SharedPtr<daal::algorithms::svm::Model> m = get(classifier::training::model);
    if(!m->getSupportVectors())
    {
        this->_errors->add(services::Error::create(services::ErrorModelNotFullInitialized, services::ArgumentName, supportVectorsStr()));
        return;
    }
    if(!m->getClassificationCoefficients())
    {
        this->_errors->add(services::Error::create(services::ErrorModelNotFullInitialized, services::ArgumentName, classificationCoefficientsStr()));
        return;
    }
}

}// namespace interface1
}// namespace training
}// namespace svm
}// namespace algorithms
}// namespace daal

/* file: stump_train.cpp */
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
//  Implementation of stump algorithm and types methods.
//--
*/

#include "stump_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{
namespace interface1
{
Result::Result() {}

/**
 * Returns the model trained with the Stump algorithm
 * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
 * \return          Model trained with the Stump algorithm
 */
services::SharedPtr<daal::algorithms::stump::Model> Result::get(classifier::training::ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::stump::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Check the correctness of the Result object
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameters structure
 * \param[in] method    Algorithm computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    checkImpl(input, parameter);

    if(this->_errors->size() != 0) { return; }

    const classifier::Parameter *algPar = static_cast<const classifier::Parameter *>(parameter);
    if(algPar->nClasses != 2) { this->_errors->add(services::ErrorModelNotFullInitialized); return; }

    services::SharedPtr<daal::algorithms::stump::Model> m = get(classifier::training::model);
    if(!data_management::checkNumericTable(m->values.get(), this->_errors.get(), valueStr(), 0, 0, 3, 1)) { return; }
}

}// namespace interface1
}// namespace training
}// namespace stump
}// namespace algorithms
}// namespace daal

/* file: implicit_als_train_init_partial_result.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "implicit_als_training_init_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace interface1
{
PartialResult::PartialResult() : daal::algorithms::PartialResult(1) {}

/**
 * Returns a partial result of the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the partial result
 * \return          Partial result that corresponds to the given identifier
 */
services::SharedPtr<PartialModel> PartialResult::get(PartialResultId id) const
{
    return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const services::SharedPtr<PartialModel> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks a partial result of the implicit ALS initialization algorithm
 * \param[in] input       %Input object for the algorithm
 * \param[in] parameter   %Parameter of the algorithm
 * \param[in] method      Computation method of the algorithm
 */
void PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);
    int unexpectedLayouts = (int)packed_mask;

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    const Input *algInput = static_cast<const Input *>(input);
    size_t nRows = algInput->get(data)->getNumberOfRows();
    DAAL_CHECK_EX(algParameter->fullNUsers > nRows, ErrorIncorrectParameter, ParameterName, fullNUsersStr());

    PartialModelPtr model = get(partialModel);
    DAAL_CHECK(model, ErrorNullPartialModel);

    size_t nFactors = algParameter->nFactors;
    if(!checkNumericTable(model->getFactors().get(), this->_errors.get(), factorsStr(), unexpectedLayouts, 0, nFactors, nRows)) { return; }
    if(!checkNumericTable(model->getIndices().get(), this->_errors.get(), indicesStr(), unexpectedLayouts, 0, 1, nRows)) { return; }
}

}// namespace interface1
}// namespace init
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal

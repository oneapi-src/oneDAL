/* file: objective_function_types_fpt.cpp */
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
//  Implementation of objective function classes.
//--
*/

#include "algorithms/optimization_solver/objective_function/objective_function_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace objective_function
{
namespace interface1
{
/**
 * Allocates memory for storing results of the Objective function
 * \param[in] input     Pointer to the input structure
 * \param[in] parameter Pointer to the parameter structure
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using namespace services;
    using namespace data_management;

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    if(algParameter == 0)
    {
        this->_errors->add(ErrorNullParameterNotSupported); return;
    }

    size_t nCols = algInput->get(argument)->getNumberOfColumns();

    DataCollectionPtr collection = DataCollectionPtr(new DataCollection(3));

    if(algParameter->resultsToCompute & gradient)
    {
        (*collection)[(int)gradientIdx] =
            SerializationIfacePtr(new HomogenNumericTable<algorithmFPType>(nCols, 1, NumericTable::doAllocate, 0));
    }
    if(algParameter->resultsToCompute & value)
    {
        (*collection)[(int)valueIdx] =
            SerializationIfacePtr(new HomogenNumericTable<algorithmFPType>(1, 1, NumericTable::doAllocate, 0));
    }
    if(algParameter->resultsToCompute & hessian)
    {
        (*collection)[(int)hessianIdx] =
            SerializationIfacePtr(new HomogenNumericTable<algorithmFPType>(nCols, nCols, NumericTable::doAllocate, 0));
    }

    Argument::set(resultCollection, staticPointerCast<DataCollection, SerializationIface>(collection));
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace interface1
} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

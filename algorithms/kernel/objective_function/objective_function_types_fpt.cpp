/* file: objective_function_types_fpt.cpp */
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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using namespace services;
    using namespace data_management;

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);

    size_t nRows = algInput->get(argument)->getNumberOfRows();

    if(algParameter->resultsToCompute & gradient)
    {
        NumericTablePtr nt = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, nRows, NumericTable::doAllocate, 0));
        Argument::set(gradientIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if(algParameter->resultsToCompute & value)
    {
        NumericTablePtr nt = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, 1, NumericTable::doAllocate, 0));
        Argument::set(valueIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if(algParameter->resultsToCompute & hessian)
    {
        NumericTablePtr nt = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(nRows, nRows, NumericTable::doAllocate, 0));
        Argument::set(hessianIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    return services::Status();
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace interface1
} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

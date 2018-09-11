/* file: objective_function_types.cpp */
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
//  Implementation of objective function classes.
//--
*/

#include "algorithms/optimization_solver/objective_function/objective_function_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;

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
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_OBJECTIVE_FUNCTION_RESULT_ID);
Parameter::Parameter(const DAAL_UINT64 resultsToCompute) : resultsToCompute(resultsToCompute) {}

Parameter::Parameter(const Parameter &other) : resultsToCompute(other.resultsToCompute) {}

/** Default constructor */
Input::Input(size_t n) : daal::algorithms::Input(n)
{}
Input::Input(const Input& other) : daal::algorithms::Input(other){}

/**
 * Sets one input object for Objective function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Objective function
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Checks the correctness of the input
 * \param[in] par       Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(par != 0, services::ErrorNullParameterNotSupported);
    return checkNumericTable(get(argument).get(), argumentStr(), 0, 0, 1);
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultId + 1)
{}

/**
 * Sets the result of the Objective function
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the numeric table with the result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the collection of the results of the Objective function
 * \param[in] id   Identifier of the result
 * \return         %Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}


/**
* Checks the result of the Objective function
* \param[in] input   %Input of the algorithm
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method
*/
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    using namespace services;
    DAAL_CHECK(Argument::size() == 3, ErrorIncorrectNumberOfArguments);

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(par);
    DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);
    const size_t nRows = algInput->get(argument)->getNumberOfRows();

    services::Status s;
    if(algParameter->resultsToCompute & value)
    {
        s = checkNumericTable(get(valueIdx).get(), valueIdxStr(), 0, 0, 1, 1);
    }
    if(algParameter->resultsToCompute & gradient)
    {
        s |= checkNumericTable(get(gradientIdx).get(), gradientIdxStr(), 0, 0, 1, nRows);
    }
    if(algParameter->resultsToCompute & hessian)
    {
        s |= checkNumericTable(get(hessianIdx).get(), hessianIdxStr(), 0, 0, nRows, nRows);
    }
    return s;
}

} // namespace interface1
} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

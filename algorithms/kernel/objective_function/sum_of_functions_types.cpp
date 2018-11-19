/* file: sum_of_functions_types.cpp */
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

#include "algorithms/optimization_solver/objective_function/sum_of_functions_types.h"
#include "numeric_table.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sum_of_functions
{
namespace interface1
{
Parameter::Parameter(size_t numberOfTerms, data_management::NumericTablePtr batchIndices,
                     const DAAL_UINT64 resultsToCompute) :
    numberOfTerms(numberOfTerms),
    objective_function::Parameter(resultsToCompute),
    batchIndices(batchIndices) {}

Parameter::Parameter(const Parameter &other) :
    numberOfTerms(other.numberOfTerms),
    objective_function::Parameter(other.resultsToCompute),
    batchIndices(other.batchIndices) {}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    if(batchIndices.get() != NULL)
        DAAL_CHECK_EX(batchIndices->getNumberOfRows() == 1, ErrorIncorrectNumberOfObservations, ArgumentName, batchIndicesStr());
    DAAL_CHECK(numberOfTerms != 0, ErrorZeroNumberOfTerms);
    return services::Status();
}

/** Default constructor */
Input::Input(size_t n) : objective_function::Input(n)
{}

Input::Input(const Input& other) : objective_function::Input(other){}

/**
 * Sets one input object for Sum of functions
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Sum of functions
 * \param[in] id    Identifier of the input numeric table
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Checks the correctness of the input
 * \param[in] par       Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(par);
    DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);
    return checkNumericTable(get(argument).get(), argumentStr(), 0, 0, 1);
}

} // namespace interface1
} // namespace sum_of_functions
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

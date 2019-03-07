/* file: logistic_loss_types.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of logloss classes.
//--
*/

#include "algorithms/optimization_solver/objective_function/logistic_loss_types.h"
#include "numeric_table.h"
#include "daal_strings.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace interface1
{
/**
 * Constructs the parameter of Logistic loss objective function
 * \param[in] numberOfTerms    The number of terms in the function
 * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                               a batch of indices used to compute the function results, e.g.,
                               value of the sum of the functions. If no indices are provided,
                               all terms will be used in the computations.
 * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
 */
Parameter::Parameter(size_t numberOfTerms, data_management::NumericTablePtr batchIndices, const DAAL_UINT64 resultsToCompute) :
    sum_of_functions::Parameter(numberOfTerms, batchIndices, resultsToCompute), penaltyL1(0), penaltyL2(0), interceptFlag(true)
{}

/**
 * Constructs an Parameter by copying input objects and parameters of another Parameter
 * \param[in] other An object to be used as the source to initialize object
 */
Parameter::Parameter(const Parameter &other) : sum_of_functions::Parameter(other),
    penaltyL1(other.penaltyL1), penaltyL2(other.penaltyL2), interceptFlag(other.interceptFlag)
{}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    DAAL_CHECK_EX(penaltyL1 >= 0, services::ErrorIncorrectParameter, services::ParameterName, penaltyL1Str());
    DAAL_CHECK_EX(penaltyL2 >= 0, services::ErrorIncorrectParameter, services::ParameterName, penaltyL2Str());
    return sum_of_functions::Parameter::check();
}

/** Default constructor */
Input::Input() : sum_of_functions::Input(lastInputId + 1)
{}

Input::Input(const Input& other) : sum_of_functions::Input(other){}

/**
 * Sets one input object for Logistic loss objective function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the input numeric table for Logistic loss objective function
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
    sum_of_functions::Input::check(par, method);
    DAAL_CHECK(Argument::size() == 3, services::ErrorIncorrectNumberOfInputNumericTables);

    services::Status s = checkNumericTable(get(data).get(), dataStr(), 0, 0);
    if(!s)
        return s;

    const size_t nColsInData = get(data)->getNumberOfColumns();
    const size_t nRowsInData = get(data)->getNumberOfRows();

    s = checkNumericTable(get(dependentVariables).get(), dependentVariablesStr(), 0, 0, 1, nRowsInData);
    s |= checkNumericTable(get(argument).get(), argumentStr(), 0, 0, 1, nColsInData + 1);
    return s;
}

} // namespace interface1
} // namespace logistic_loss
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

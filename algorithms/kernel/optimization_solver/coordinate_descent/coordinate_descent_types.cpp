/* file: coordinate_descent_types.cpp */
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
//  Implementation of coordinate_descent solver classes.
//--
*/

#include "algorithms/optimization_solver/coordinate_descent/coordinate_descent_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace coordinate_descent
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_COORDINATE_DESCENT_RESULT_ID);

Parameter::Parameter(
    const sum_of_functions::BatchPtr& function, size_t nIterations,
    double accuracyThreshold, size_t seed) :

    optimization_solver::iterative_solver::Parameter(function, nIterations, accuracyThreshold, false, 1),
    seed(seed),
    engine(engines::mt19937::Batch<>::create()), selection(cyclic), positive(false), skipTheFirstComponents(false)
{}

services::Status Parameter::check() const
{
    services::Status s = iterative_solver::Parameter::check();
    if(!s) return s;

    if(batchSize > function->sumOfFunctionsParameter->numberOfTerms || batchSize == 0)
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, batchSizeStr()));

    return s;
}

Input::Input() {}
Input::Input(const Input& other) {}

services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{

    services::Status s;
    if(this->size() != 2)
        return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);

    s = data_management::checkNumericTable(get(iterative_solver::inputArgument).get(), inputArgumentStr(), 0, 0);

    if(!s) return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(!pOpt.get())
        return services::Status();//ok

    return s;
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
    int method) const
{
    services::Status s;// = super::check(input, par, method);
    if(!s || !static_cast<const Parameter*>(par)->optionalResultRequired)
        return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(!pOpt.get())
    {
        return services::Status(services::ErrorNullOptionalResult);
    }

    return s;
}

} // namespace interface1
} // namespace coordinate_descent
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

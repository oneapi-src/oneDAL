/* file: coordinate_descent_types_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of coordinate_descent solver classes.
//--
*/

#include "algorithms/optimization_solver/coordinate_descent/coordinate_descent_types.h"

using namespace daal::data_management;

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
/**
* Allocates memory to store the results of the iterative solver algorithm
* \param[in] input  Pointer to the input structure
* \param[in] par    Pointer to the parameter structure
* \param[in] method Computation method of the algorithm
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    services::Status s;
    const Input * algInput = static_cast<const Input *>(input);
    size_t nRows           = algInput->get(optimization_solver::iterative_solver::inputArgument)->getNumberOfRows();
    size_t nColumns        = algInput->get(optimization_solver::iterative_solver::inputArgument)->getNumberOfColumns();

    if (!get(optimization_solver::iterative_solver::minimum))
    {
        set(optimization_solver::iterative_solver::minimum,
            HomogenNumericTable<algorithmFPType>::create(nColumns, nRows, NumericTable::doAllocate, &s));
    }
    if (!get(optimization_solver::iterative_solver::nIterations))
    {
        set(optimization_solver::iterative_solver::nIterations, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate, (size_t)0, &s));
    }

    const Parameter * algParam = static_cast<const Parameter *>(par);
    return s;
}
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace interface1
} // namespace coordinate_descent
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

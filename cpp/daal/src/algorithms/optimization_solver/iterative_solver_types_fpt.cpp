/* file: iterative_solver_types_fpt.cpp */
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
//  Implementation of iterative solver classes.
//--
*/

#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace iterative_solver
{
namespace interface2
{

namespace dm  = daal::data_management;
namespace dmi = daal::data_management::internal;
/**
 * Allocates memory to store the results of the iterative solver algorithm
 * \param[in] input  Pointer to the input structure
 * \param[in] par    Pointer to the parameter structure
 * \param[in] method Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    const Input * algInput = static_cast<const Input *>(input);
    size_t nRows           = algInput->get(inputArgument)->getNumberOfRows();

    services::Status status;

    if (!get(minimum))
    {
        dm::NumericTablePtr nt;
        nt = dm::HomogenNumericTable<algorithmFPType>::create(1, nRows, dm::NumericTable::doAllocate, &status);
        set(minimum, nt);
    }
    if (!get(nIterations))
    {
        dm::NumericTablePtr nt = dm::HomogenNumericTable<int>::create(1, 1, dm::NumericTable::doAllocate, int(0));
        set(nIterations, nt);
    }
    return status;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace interface2
} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

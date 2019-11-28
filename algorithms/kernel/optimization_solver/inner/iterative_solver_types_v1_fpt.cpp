/* file: iterative_solver_types_v1_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace iterative_solver
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
    using namespace daal::data_management;
    const Input * algInput = static_cast<const Input *>(input);
    size_t nRows           = algInput->get(inputArgument)->getNumberOfRows();
    if (!get(minimum))
    {
        set(minimum, NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, nRows, NumericTable::doAllocate)));
    }
    if (!get(nIterations))
    {
        set(nIterations, NumericTablePtr(new HomogenNumericTable<size_t>(1, 1, NumericTable::doAllocate, (size_t)0)));
    }
    return services::Status();
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace interface1

} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

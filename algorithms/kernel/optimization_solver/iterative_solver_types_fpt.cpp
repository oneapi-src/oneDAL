/* file: iterative_solver_types_fpt.cpp */
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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    using namespace daal::data_management;
    const Input *algInput = static_cast<const Input *>(input);
    size_t nRows = algInput->get(inputArgument)->getNumberOfRows();
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

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace interface1
} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

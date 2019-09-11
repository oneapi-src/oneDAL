/* file: coordinate_descent_types_fpt.cpp */
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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    services::Status s;
    const Input *algInput = static_cast<const Input *>(input);
    size_t nRows = algInput->get(optimization_solver::iterative_solver::inputArgument)->getNumberOfRows();
    size_t nColumns = algInput->get(optimization_solver::iterative_solver::inputArgument)->getNumberOfColumns();

    if (!get(optimization_solver::iterative_solver::minimum))
    {
        set(optimization_solver::iterative_solver::minimum, HomogenNumericTable<algorithmFPType>::create(nColumns, nRows, NumericTable::doAllocate, &s));
    }
    if (!get(optimization_solver::iterative_solver::nIterations))
    {
        set(optimization_solver::iterative_solver::nIterations, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate, (size_t)0, &s));
    }

    const Parameter *algParam = static_cast<const Parameter *>(par);
    if(!algParam->optionalResultRequired)
    {
        return s;
    }
    return s;
}
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace interface1
} // namespace coordinate_descent
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

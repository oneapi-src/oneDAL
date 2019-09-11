/* file: objective_function_types_fpt.cpp */
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
    services::Status s;
    algorithmFPType zero = 0;
    using namespace services;
    using namespace data_management;

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);

    const size_t nRows = algInput->get(argument)->getNumberOfRows();
    const size_t nCols = algInput->get(argument)->getNumberOfColumns();
    if(algParameter->resultsToCompute & gradient)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(gradientIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if(algParameter->resultsToCompute & value)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(valueIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if(algParameter->resultsToCompute & hessian)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(nRows, nRows, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(hessianIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if(algParameter->resultsToCompute & nonSmoothTermValue)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(nonSmoothTermValueIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if(algParameter->resultsToCompute & proximalProjection)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(proximalProjectionIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }

    if(algParameter->resultsToCompute & lipschitzConstant)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(lipschitzConstantIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }

    if(algParameter->resultsToCompute & componentOfGradient)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(nCols, 1, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(componentOfGradientIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if(algParameter->resultsToCompute & componentOfHessianDiagonal)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(nCols, 1, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(componentOfHessianDiagonalIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if(algParameter->resultsToCompute & componentOfProximalProjection)
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(nCols, 1, NumericTable::doAllocate,zero,&s));
        DAAL_CHECK_STATUS_VAR(s);
        Argument::set(componentOfProximalProjectionIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace interface1
} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

/* file: objective_function_types_fpt.cpp */
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
//  Implementation of objective function classes.
//--
*/

#include "algorithms/optimization_solver/objective_function/objective_function_types.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace objective_function
{
using daal::data_management::internal::SyclHomogenNumericTable;

/**
 * Allocates memory for storing results of the Objective function
 * \param[in] input     Pointer to the input structure
 * \param[in] parameter Pointer to the parameter structure
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    algorithmFPType zero = 0;
    using namespace services;
    using namespace data_management;

    const Input * algInput         = static_cast<const Input *>(input);
    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);

    services::Status status;

    auto & context    = services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    const size_t nCols = algInput->get(argument)->getNumberOfColumns();
    const size_t nRows = algInput->get(argument)->getNumberOfRows();

    if (algParameter->resultsToCompute & gradient && !Argument::get(gradientIdx))
    {
        NumericTablePtr nt;
        if (deviceInfo.isCpu)
        {
            nt = HomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTable::doAllocate, zero, &status);
        }
        else
        {
            nt = SyclHomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTable::doAllocate, zero, &status);
        }
        Argument::set(gradientIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if (algParameter->resultsToCompute & value && !Argument::get(valueIdx))
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, zero, &status));
        DAAL_CHECK_STATUS_VAR(status);
        Argument::set(valueIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if (algParameter->resultsToCompute & hessian && !Argument::get(hessianIdx))
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(nRows, nRows, NumericTable::doAllocate, zero, &status));
        DAAL_CHECK_STATUS_VAR(status);
        Argument::set(hessianIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if (algParameter->resultsToCompute & nonSmoothTermValue && !Argument::get(nonSmoothTermValueIdx))
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, zero, &status));
        DAAL_CHECK_STATUS_VAR(status);
        Argument::set(nonSmoothTermValueIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if (algParameter->resultsToCompute & proximalProjection && !Argument::get(proximalProjectionIdx))
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTable::doAllocate, zero, &status));
        DAAL_CHECK_STATUS_VAR(status);
        Argument::set(proximalProjectionIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if (algParameter->resultsToCompute & lipschitzConstant && !Argument::get(lipschitzConstantIdx))
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, zero, &status));
        DAAL_CHECK_STATUS_VAR(status);
        Argument::set(lipschitzConstantIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if (algParameter->resultsToCompute & componentOfGradient && !Argument::get(componentOfGradientIdx))
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(nCols, 1, NumericTable::doAllocate, zero, &status));
        DAAL_CHECK_STATUS_VAR(status);
        Argument::set(componentOfGradientIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if (algParameter->resultsToCompute & componentOfHessianDiagonal && !Argument::get(componentOfHessianDiagonalIdx))
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(nCols, 1, NumericTable::doAllocate, zero, &status));
        DAAL_CHECK_STATUS_VAR(status);
        Argument::set(componentOfHessianDiagonalIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    if (algParameter->resultsToCompute & componentOfProximalProjection && !Argument::get(componentOfProximalProjectionIdx))
    {
        NumericTablePtr nt = NumericTablePtr(HomogenNumericTable<algorithmFPType>::create(nCols, 1, NumericTable::doAllocate, zero, &status));
        DAAL_CHECK_STATUS_VAR(status);
        Argument::set(componentOfProximalProjectionIdx, staticPointerCast<NumericTable, SerializationIface>(nt));
    }
    return status;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

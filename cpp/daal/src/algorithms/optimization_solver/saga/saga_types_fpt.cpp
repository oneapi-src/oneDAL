/* file: saga_types_fpt.cpp */
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
//  Implementation of saga solver classes.
//--
*/

#include "algorithms/optimization_solver/saga/saga_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace saga
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
    services::Status s = super::allocate<algorithmFPType>(input, par, method);
    if (!s) return s;
    const Parameter * algParam = static_cast<const Parameter *>(par);
    if (!algParam->optionalResultRequired)
    {
        return s;
    }
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (pOpt.get())
    {
        if (pOpt->size() != lastOptionalData + 1)
        {
            return s; //error, will be found in check
        }
    }
    else
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        Argument::set(iterative_solver::optionalResult, pOpt);
    }
    const Input * algInput           = static_cast<const Input *>(input);
    const size_t nRows               = algInput->get(iterative_solver::inputArgument)->getNumberOfRows();
    NumericTablePtr pTbl             = NumericTable::cast(pOpt->get(gradientsTable));
    NumericTablePtr gradientsInputNt = NumericTable::cast(algInput->get(gradientsTable));
    if (!pTbl.get())
    {
        if (gradientsInputNt.get())
        {
            pOpt->set(gradientsTable, gradientsInputNt);
        }
        else
        {
            pTbl = HomogenNumericTable<algorithmFPType>::create(nRows, algParam->function->sumOfFunctionsParameter->numberOfTerms,
                                                                NumericTable::doAllocate, 0.0, &s);
            pOpt->set(gradientsTable, pTbl);
        }
    }
    return s;
}
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace saga
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

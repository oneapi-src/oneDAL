/* file: sgd_types_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of sgd solver classes.
//--
*/

#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "algorithms/optimization_solver/sgd/sgd_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace interface1
{

template <typename algorithmFPType>
services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    services::Status s = super::allocate<algorithmFPType>(input, par, method);
    if(!s) return s;
    const BaseParameter *algParam = static_cast<const BaseParameter *>(par);
    if(!algParam->optionalResultRequired)
        return services::Status();
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(pOpt.get())
    {
        if(pOpt->size() != lastOptionalData + 1)
            return services::Status(services::ErrorIncorrectParameter);//error, will be found in check
    }
    else
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        set(iterative_solver::optionalResult, pOpt);
    }

    // NumericTablePtr pTbl = NumericTable::cast(pOpt->get(pastUpdateVector));
    if(algParam->optionalResultRequired)
    {
        const Input *algInput = static_cast<const Input *>(input);
        size_t argumentSize = algInput->get(iterative_solver::inputArgument)->getNumberOfRows();
        NumericTablePtr pTbl = NumericTablePtr(new HomogenNumericTable<int>(1, 1, NumericTable::doAllocate, 0));
        pOpt->set(iterative_solver::lastIteration, pTbl);
        if(method == (int) momentum)
        {
            if(!pOpt->get(pastUpdateVector))
            {
                pTbl = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, argumentSize, NumericTable::doAllocate, 0.0));
                pOpt->set(pastUpdateVector, pTbl);
            }
        }
        if(method == (int) miniBatch)
        {
            if(!pOpt->get(pastWorkValue))
            {
                pTbl = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, argumentSize, NumericTable::doAllocate, 0.0));
                pOpt->set(pastWorkValue, pTbl);
            }
        }
    }
    return services::Status();
}
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

} // namespace interface1
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

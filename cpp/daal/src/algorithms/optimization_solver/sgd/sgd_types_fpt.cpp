/* file: sgd_types_fpt.cpp */
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
//  Implementation of sgd solver classes.
//--
*/

#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "algorithms/optimization_solver/sgd/sgd_types.h"
#include "src/services/service_data_utils.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace dm  = daal::data_management;
namespace dmi = daal::data_management::internal;

template <typename algorithmFPType>
services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    services::Status s = super::allocate<algorithmFPType>(input, par, method);
    if (!s) return s;
    const BaseParameter * algParam = static_cast<const BaseParameter *>(par);
    if (!algParam->optionalResultRequired) return services::Status();
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (pOpt.get())
    {
        if (pOpt->size() != lastOptionalData + 1) return services::Status(services::ErrorIncorrectParameter); //error, will be found in check
    }
    else
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        DAAL_CHECK_MALLOC(pOpt.get())
        set(iterative_solver::optionalResult, pOpt);
    }

    // NumericTablePtr pTbl = NumericTable::cast(pOpt->get(pastUpdateVector));
    if (algParam->optionalResultRequired)
    {
        const Input * algInput   = static_cast<const Input *>(input);
        size_t argumentSize      = algInput->get(iterative_solver::inputArgument)->getNumberOfRows();
        dm::NumericTablePtr pTbl = dm::NumericTablePtr(new dm::HomogenNumericTable<int>(1, 1, dm::NumericTable::doAllocate, 0));

        DAAL_CHECK_MALLOC(pTbl)
        pOpt->set(iterative_solver::lastIteration, pTbl);
        DAAL_ASSERT(momentum <= services::internal::MaxVal<int>::get())
        if (method == (int)momentum)
        {
            if (!pOpt->get(pastUpdateVector))
            {
                pTbl = dm::NumericTablePtr(new dm::HomogenNumericTable<algorithmFPType>(1, argumentSize, dm::NumericTable::doAllocate, 0.0));
                DAAL_CHECK_MALLOC(pTbl.get())
                pOpt->set(pastUpdateVector, pTbl);
            }
        }
        DAAL_ASSERT(miniBatch <= services::internal::MaxVal<int>::get())
        if (method == (int)miniBatch)
        {
            auto & context    = services::internal::getDefaultContext();
            auto & deviceInfo = context.getInfoDevice();

            if (!pOpt->get(pastWorkValue))
            {
                if (deviceInfo.isCpu)
                {
                    pTbl = dm::HomogenNumericTable<algorithmFPType>::create(1, argumentSize, dm::NumericTable::doAllocate, 0.0, &s);
                }
                else
                {
                    pTbl = dmi::SyclHomogenNumericTable<algorithmFPType>::create(1, argumentSize, dm::NumericTable::doAllocate, 0.0, &s);
                }
                DAAL_CHECK_MALLOC(pTbl.get())
                pOpt->set(pastWorkValue, pTbl);
            }
        }
    }
    return services::Status();
}
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

/* file: sgd_types.cpp */
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
#include "data_management/data/numeric_table.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/services/service_data_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SGD_RESULT_ID);

BaseParameter::BaseParameter(const sum_of_functions::BatchPtr & function, size_t nIterations, double accuracyThreshold, NumericTablePtr batchIndices,
                             NumericTablePtr learningRateSequence, size_t batchSize, size_t seed)
    : optimization_solver::iterative_solver::Parameter(function, nIterations, accuracyThreshold, false, batchSize),
      batchIndices(batchIndices),
      learningRateSequence(learningRateSequence),
      seed(seed),
      engine(engines::mt19937::Batch<>::create())
{}

/**
 * Checks the correctness of the parameter
 */
services::Status BaseParameter::check() const
{
    services::Status s = iterative_solver::Parameter::check();
    if (!s) return s;

    if (learningRateSequence.get() != NULL)
    {
        DAAL_CHECK_EX(learningRateSequence->getNumberOfRows() > 0, ErrorIncorrectNumberOfObservations, ArgumentName, "learningRateSequence");
        DAAL_CHECK_EX(learningRateSequence->getNumberOfColumns() == 1, ErrorIncorrectNumberOfFeatures, ArgumentName, "learningRateSequence");
    }
    return s;
}

Parameter<defaultDense>::Parameter(const sum_of_functions::BatchPtr & function, size_t nIterations, double accuracyThreshold,
                                   NumericTablePtr batchIndices, NumericTablePtr learningRateSequence, size_t seed)
    : BaseParameter(function, nIterations, accuracyThreshold, batchIndices, learningRateSequence,
                    1, // batchSize
                    seed)
{}
/**
 * Checks the correctness of the parameter
 */
services::Status Parameter<defaultDense>::check() const
{
    services::Status s = BaseParameter::check();
    if (!s) return s;
    if (batchIndices.get() != NULL)
    {
        return checkNumericTable(batchIndices.get(), batchIndicesStr(), 0, 0, 1, nIterations);
    }
    return s;
}

Parameter<miniBatch>::Parameter(const sum_of_functions::BatchPtr & function, size_t nIterations, double accuracyThreshold,
                                NumericTablePtr batchIndices, size_t batchSize, NumericTablePtr conservativeSequence, size_t innerNIterations,
                                NumericTablePtr learningRateSequence, size_t seed)
    : BaseParameter(function, nIterations, accuracyThreshold, batchIndices, learningRateSequence, batchSize, seed),
      conservativeSequence(conservativeSequence),
      innerNIterations(innerNIterations)
{}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter<miniBatch>::check() const
{
    services::Status s = BaseParameter::check();
    if (!s) return s;
    if (batchIndices.get() != NULL)
    {
        s |= checkNumericTable(batchIndices.get(), batchIndicesStr(), 0, 0, batchSize, nIterations);
        DAAL_CHECK_STATUS_VAR(s);
    }

    if (conservativeSequence.get() != NULL)
    {
        DAAL_CHECK_EX(conservativeSequence->getNumberOfRows() == nIterations || conservativeSequence->getNumberOfRows() == 1,
                      ErrorIncorrectNumberOfObservations, ArgumentName, conservativeSequenceStr());
        s |= checkNumericTable(conservativeSequence.get(), conservativeSequenceStr(), 0, 0, 1);
        DAAL_CHECK_STATUS_VAR(s);
    }

    DAAL_CHECK_EX(batchSize <= function->sumOfFunctionsParameter->numberOfTerms && batchSize > 0, ErrorIncorrectParameter, ArgumentName, "batchSize");
    return s;
}

Parameter<momentum>::Parameter(const sum_of_functions::BatchPtr & function, double momentum_, size_t nIterations, double accuracyThreshold,
                               NumericTablePtr batchIndices, size_t batchSize, NumericTablePtr learningRateSequence, size_t seed)
    : BaseParameter(function, nIterations, accuracyThreshold, batchIndices, learningRateSequence, batchSize, seed), momentum(momentum_)
{}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter<momentum>::check() const
{
    services::Status s = BaseParameter::check();
    if (!s) return s;
    if (batchIndices.get() != NULL)
    {
        s |= checkNumericTable(batchIndices.get(), batchIndicesStr(), 0, 0, batchSize, nIterations);
        DAAL_CHECK_STATUS_VAR(s);
    }

    DAAL_CHECK_EX(batchSize <= function->sumOfFunctionsParameter->numberOfTerms && batchSize > 0, ErrorIncorrectParameter, ArgumentName, "batchSize");
    return s;
}

Input::Input() {}
Input::Input(const Input & other) {}

services::Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = super::check(par, method);
    if (!s) return s;

    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (!pOpt.get())
    {
        return services::Status(); //ok
    }
    DAAL_CHECK(pOpt->size() == lastOptionalData + 1, ErrorIncorrectOptionalInput);
    size_t argumentSize = get(iterative_solver::inputArgument)->getNumberOfRows();
    DAAL_ASSERT(momentum <= services::internal::MaxVal<int>::get())
    if (method == (int)momentum)
    {
        return checkNumericTable(get(pastUpdateVector).get(), pastUpdateVectorStr(), 0, 0, 1, argumentSize);
    }
    return s;
}

NumericTablePtr Input::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (pOpt.get())
    {
        return NumericTable::cast(pOpt->get(id));
    }
    return NumericTablePtr();
}

void Input::set(OptionalDataId id, const NumericTablePtr & ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if (!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        set(iterative_solver::optionalArgument, pOpt);
    }
    pOpt->set(id, ptr);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    services::Status s = super::check(input, par, method);
    if (!s || !static_cast<const BaseParameter *>(par)->optionalResultRequired)
    {
        return s;
    }
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);

    DAAL_CHECK(pOpt.get(), ErrorNullOptionalResult);
    DAAL_CHECK(pOpt->size() == lastOptionalData + 1, ErrorIncorrectOptionalResult);
    const Input * algInput = static_cast<const Input *>(input);
    size_t argumentSize    = algInput->get(iterative_solver::inputArgument)->getNumberOfRows();
    DAAL_ASSERT(momentum <= services::internal::MaxVal<int>::get())
    if (method == (int)momentum)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(pastUpdateVector).get(), pastUpdateVectorStr(), 0, 0, 1, argumentSize));
    }
    return s;
}

NumericTablePtr Result::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (pOpt.get())
    {
        return NumericTable::cast(pOpt->get(id));
    }
    return NumericTablePtr();
}

void Result::set(OptionalDataId id, const NumericTablePtr & ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        set(iterative_solver::optionalResult, pOpt);
    }
    pOpt->set(id, ptr);
}

} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

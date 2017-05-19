/* file: sgd_types.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "data_management/data/memory_block.h"
#include "numeric_table.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_SGD_RESULT_ID);
/**
 * Constructs the parameter base class of the Stochastic gradient descent algorithm
 * \param[in] function             Objective function represented as sum of functions
 * \param[in] nIterations          Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
 * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function.
 *                                 If no indices are provided, the implementation will generate random indices.
 * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
 * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
 */
BaseParameter::BaseParameter(
    const sum_of_functions::BatchPtr &function,
    size_t nIterations,
    double accuracyThreshold,
    NumericTablePtr batchIndices,
    NumericTablePtr learningRateSequence,
    size_t batchSize,
    size_t seed) :
    optimization_solver::iterative_solver::Parameter(function, nIterations, accuracyThreshold, false, batchSize),
    batchIndices(batchIndices),
    learningRateSequence(learningRateSequence),
    seed(seed)
{}

/**
 * Checks the correctness of the parameter
 */
services::Status BaseParameter::check() const
{
    services::Status s = iterative_solver::Parameter::check();
    if(!s) return s;

    if(learningRateSequence.get() != NULL)
    {
        DAAL_CHECK_EX(learningRateSequence->getNumberOfRows() > 0, \
                      ErrorIncorrectNumberOfObservations, ArgumentName, "learningRateSequence");
        DAAL_CHECK_EX(learningRateSequence->getNumberOfColumns() == 1, ErrorIncorrectNumberOfFeatures, ArgumentName, "learningRateSequence");
    }
    return s;
}
/**
 * \param[in] function             Objective function represented as sum of functions
 * \param[in] nIterations          Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
 * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are
                                   provided, the implementation will generate random indices.
 * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
 * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
 */
Parameter<defaultDense>::Parameter(
    const sum_of_functions::BatchPtr &function,
    size_t nIterations,
    double accuracyThreshold,
    NumericTablePtr batchIndices,
    NumericTablePtr learningRateSequence,
    size_t seed) :
    BaseParameter(
        function,
        nIterations,
        accuracyThreshold,
        batchIndices,
        learningRateSequence,
        1, // batchSize
        seed
    )
{}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter<defaultDense>::check() const
{
    services::Status s = BaseParameter::check();
    if(!s) return s;
    if(batchIndices.get() != NULL)
    {
        return checkNumericTable(batchIndices.get(), batchIndicesStr(), 0, 0, 1, nIterations);
    }
    return s;
}

/**
 * Constructs the parameter class of the Stochastic gradient descent algorithm
 * \param[in] function             Objective function represented as sum of functions
 * \param[in] nIterations          Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
 * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices
                                   are provided, the implementation will generate random indices.
 * \param[in] batchSize            Number of batch indices to compute the stochastic gradient. If batchSize is equal to the number of terms
                                   in objective function then no random sampling is performed, and all terms are used to calculate the gradient.
                                   This parameter is ignored if batchIndices is provided.
 * \param[in] conservativeSequence Numeric table of values of the conservative coefficient sequence
 * \param[in] innerNIterations     Number of inner iterations
 * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
 * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
 */
Parameter<miniBatch>::Parameter(
    const sum_of_functions::BatchPtr &function,
    size_t nIterations,
    double accuracyThreshold,
    NumericTablePtr batchIndices,
    size_t batchSize,
    NumericTablePtr conservativeSequence,
    size_t innerNIterations,
    NumericTablePtr learningRateSequence,
    size_t seed) :
    BaseParameter(
        function,
        nIterations,
        accuracyThreshold,
        batchIndices,
        learningRateSequence,
        batchSize,
        seed
    ),
    conservativeSequence(conservativeSequence),
    innerNIterations(innerNIterations)
{}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter<miniBatch>::check() const
{
    services::Status s = BaseParameter::check();
    if(!s) return s;
    if(batchIndices.get() != NULL)
    {
        s |= checkNumericTable(batchIndices.get(), batchIndicesStr(), 0, 0, batchSize, nIterations);
        if(!s) return s;
    }

    if(conservativeSequence.get() != NULL)
    {
        DAAL_CHECK_EX(conservativeSequence->getNumberOfRows() == nIterations || conservativeSequence->getNumberOfRows() == 1, \
                      ErrorIncorrectNumberOfObservations, ArgumentName, conservativeSequenceStr());
        s |= checkNumericTable(conservativeSequence.get(), conservativeSequenceStr(), 0, 0, 1);
        if(!s) return s;
    }

    DAAL_CHECK_EX(batchSize <= function->sumOfFunctionsParameter->numberOfTerms && batchSize > 0, ErrorIncorrectParameter, \
                  ArgumentName, "batchSize");
    return s;
}

Parameter<momentum>::Parameter(
    const sum_of_functions::BatchPtr &function,
    double momentum_,
    size_t nIterations,
    double accuracyThreshold,
    NumericTablePtr batchIndices,
    size_t batchSize,
    NumericTablePtr learningRateSequence,
    size_t seed) :
    BaseParameter(function,
                  nIterations,
                  accuracyThreshold,
                  batchIndices,
                  learningRateSequence,
                  batchSize,
                  seed),
    momentum(momentum_)
{}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter<momentum>::check() const
{
    services::Status s = BaseParameter::check();
    if(!s) return s;
    if(batchIndices.get() != NULL)
    {
        s |= checkNumericTable(batchIndices.get(), batchIndicesStr(), 0, 0, batchSize, nIterations);
        if(!s) return s;
    }

    DAAL_CHECK_EX(batchSize <= function->sumOfFunctionsParameter->numberOfTerms && batchSize > 0, ErrorIncorrectParameter, \
                  ArgumentName, "batchSize");
    return s;
}

static services::Status checkRngState(const daal::algorithms::Input *input,
                          const daal::algorithms::Parameter *par,
                          const SerializationIface *pItem, bool bInput)
{
    const sgd::BaseParameter *algParam = static_cast<const sgd::BaseParameter *>(par);
    //if random numbers generator in the algorithm is not required
    if(algParam->batchIndices.get())
    {
        return services::Status();    // rgnState doesn't matter
    }

    //but if it is present then the SerializationIface should be an instance of expected type
    if(pItem)
    {
        if(!dynamic_cast<const MemoryBlock *>(pItem))
        {
            const ErrorDetailID det = bInput ? OptionalInput : OptionalResult;
            return services::Status(Error::create(bInput ? ErrorIncorrectOptionalInput : ErrorIncorrectOptionalResult, det, rngStateStr()));
        }
    }
    else if(!bInput)
    {
        return services::Status(Error::create(ErrorNullOptionalResult, OptionalResult, rngStateStr()));
    }
    return services::Status();
}

Input::Input() {}
Input::Input(const Input& other) {}

services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    services::Status s = super::check(par, method);
    if(!s) return s;

    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(!pOpt.get())
    {
        return services::Status();    //ok
    }
    if(pOpt->size() != optionalDataSize)
    {
        return services::Status(ErrorIncorrectOptionalInput);
    }
    s |= checkRngState(this, par, pOpt->get(rngState).get(), true);
    if(!s) return s;
    size_t argumentSize = get(iterative_solver::inputArgument)->getNumberOfRows();
    if(method == (int)momentum)
    {
        return checkNumericTable(get(pastUpdateVector).get(), pastUpdateVectorStr(), 0, 0, 1, argumentSize);
    }
    return s;
}

NumericTablePtr Input::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(pOpt.get())
    {
        return NumericTable::cast(pOpt->get(id));
    }
    return NumericTablePtr();
}

void Input::set(OptionalDataId id, const NumericTablePtr &ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(optionalDataSize));
        set(iterative_solver::optionalArgument, pOpt);
    }
    pOpt->set(id, ptr);
}


services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s = super::check(input, par, method);
    if(!s || !static_cast<const BaseParameter *>(par)->optionalResultRequired)
    {
        return s;
    }
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(!pOpt.get())
    {
        return services::Status(ErrorNullOptionalResult);
    }
    if(pOpt->size() != optionalDataSize)
    {
        return services::Status(ErrorIncorrectOptionalResult);
    }
    DAAL_CHECK_STATUS(s, checkRngState(input, par, pOpt->get(rngState).get(), false));

    const Input *algInput = static_cast<const Input *>(input);
    size_t argumentSize = algInput->get(iterative_solver::inputArgument)->getNumberOfRows();
    if(method == (int)momentum)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(pastUpdateVector).get(), pastUpdateVectorStr(), 0, 0, 1, argumentSize));
    }
    return s;
}

NumericTablePtr Result::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(pOpt.get())
    {
        return NumericTable::cast(pOpt->get(id));
    }
    return NumericTablePtr();
}

void Result::set(OptionalDataId id, const NumericTablePtr &ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(optionalDataSize));
        set(iterative_solver::optionalResult, pOpt);
    }
    pOpt->set(id, ptr);
}

} // namespace interface1
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

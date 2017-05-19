/* file: adagrad_types.cpp */
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
//  Implementation of adagrad solver classes.
//--
*/

#include "algorithms/optimization_solver/adagrad/adagrad_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ADAGRAD_RESULT_ID);
/**
 * Constructs the parameter base class of the Adaptive gradient descent algorithm
 * \param[in] function_                 Objective function represented as sum of functions
 * \param[in] nIterations_              Maximal number of iterations of the algorithm
 * \param[in] accuracyThreshold_        Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
 * \param[in] batchIndices_             Numeric table that represents 32 bit integer indices of terms in the objective function.
 *                                      If no indices are provided, the implementation will generate random indices.
 * \param[in] batchSize_                Number of batch indices to compute the stochastic gradient. If batchSize is equal to the number of terms
                                        in objective function then no random sampling is performed, and all terms are used to calculate the gradient.
                                        This parameter is ignored if batchIndices is provided.
 * \param[in] learningRate_             Numeric table that contains value of the learning rate
 * \param[in] degenerateCasesThreshold_ Value needed to avoid degenerate cases in square root computing.
 * \param[in] seed_                     Seed for random generation of 32 bit integer indices of terms in the objective function.
 */
Parameter::Parameter(
    const sum_of_functions::BatchPtr& function_, size_t nIterations_,
    double accuracyThreshold_, data_management::NumericTablePtr batchIndices_,
    const size_t batchSize_, data_management::NumericTablePtr learningRate_,
    double degenerateCasesThreshold_, size_t seed_) :

    optimization_solver::iterative_solver::Parameter(function_, nIterations_, accuracyThreshold_, false, batchSize_),
    batchIndices(batchIndices_),
    learningRate(learningRate_),
    degenerateCasesThreshold(degenerateCasesThreshold_),
    seed(seed_)
{}

services::Status Parameter::check() const
{
    services::Status s = iterative_solver::Parameter::check();
    if(!s) return s;

    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(learningRate.get(), learningRateStr(), 0, 0, 1, 1));

    if(batchSize > function->sumOfFunctionsParameter->numberOfTerms || batchSize == 0)
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, batchSizeStr()));

    if(batchIndices)
        return data_management::checkNumericTable(batchIndices.get(), batchIndicesStr(), 0, 0, batchSize, nIterations);
    return s;
}

Input::Input() {}
Input::Input(const Input& other) {}

data_management::NumericTablePtr Input::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(pOpt.get())
        return data_management::NumericTable::cast(pOpt->get(id));
    return data_management::NumericTablePtr();
}

void Input::set(OptionalDataId id, const data_management::NumericTablePtr &ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(optionalDataSize));
        set(iterative_solver::optionalArgument, pOpt);
    }
    pOpt->set(id, ptr);
}

static services::Status checkGradientSquareSumData(const daal::algorithms::Input *input, const data_management::SerializationIfacePtr& pItem, bool bInput)
{
    const services::ErrorDetailID det = bInput ? services::OptionalInput : services::OptionalResult;
    data_management::NumericTablePtr pData = data_management::NumericTable::cast(pItem);
    if(!pData.get())
    {
        return services::Status(services::Error::create(bInput ? services::ErrorIncorrectOptionalInput : services::ErrorIncorrectOptionalResult, det, gradientSquareSumStr()));
    }
    const Input* algInput = static_cast<const Input *>(input);
    auto arg = algInput->get(iterative_solver::inputArgument);
    if(pData->getNumberOfColumns() != arg->getNumberOfColumns())
    {
        return services::Status(services::Error::create(services::ErrorIncorrectNumberOfColumns, det, gradientSquareSumStr()));
    }
    if(pData->getNumberOfRows() != arg->getNumberOfRows())
    {
        return services::Status(services::Error::create(services::ErrorIncorrectNumberOfRows, det, gradientSquareSumStr()));
    }
    return services::Status();
}

static services::Status checkRngState(const daal::algorithms::Input *input,
    const daal::algorithms::Parameter *par, const data_management::SerializationIface* pItem, bool bInput)
{
    const Parameter *algParam = static_cast<const Parameter *>(par);
    //if random numbers generator in the algorithm is not required
    if(algParam->batchIndices.get())
        return services::Status();// rgnState doesn't matter

    //but if it is present then the SerializationIface should be an instance of expected type
    if(pItem)
    {
        if(!dynamic_cast<const data_management::MemoryBlock*>(pItem))
        {
            const services::ErrorDetailID det = bInput ? services::OptionalInput : services::OptionalResult;
            return services::Status(services::Error::create(bInput ?
                services::ErrorIncorrectOptionalInput : services::ErrorIncorrectOptionalResult, det, rngStateStr()));
        }
    }
    else if(!bInput)
    {
       return services::Status(services::Error::create(services::ErrorNullOptionalResult, services::OptionalResult, rngStateStr()));
    }
    return services::Status();
}

services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    services::Status s = super::check(par, method);
    if(!s) return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(!pOpt.get())
        return services::Status();//ok
    if(pOpt->size() != optionalDataSize)
    {
        return services::Status(services::ErrorIncorrectOptionalInput);
    }
    auto pItem = pOpt->get(gradientSquareSum);
    if(pItem.get())
    {
        s |= checkGradientSquareSumData(this, pItem, true);
        if(!s) return s;
    }
    return checkRngState(this, par, pOpt->get(rngState).get(),true);
}

data_management::NumericTablePtr Result::get(OptionalDataId id) const
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(pOpt.get())
        return data_management::NumericTable::cast(pOpt->get(id));
    return data_management::NumericTablePtr();
}

void Result::set(OptionalDataId id, const data_management::NumericTablePtr &ptr)
{
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(!pOpt.get())
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(optionalDataSize));
        set(iterative_solver::optionalResult, pOpt);
    }
    pOpt->set(id, ptr);
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
    int method) const
{
    services::Status s = super::check(input, par, method);
    if(!s || !static_cast<const Parameter*>(par)->optionalResultRequired)
        return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(!pOpt.get())
    {
        return services::Status(services::ErrorNullOptionalResult);
    }
    if(pOpt->size() != optionalDataSize)
    {
        return services::Status(services::ErrorIncorrectOptionalResult);
    }
    auto pItem = pOpt->get(gradientSquareSum);
    if(!pItem.get())
    {
        return services::Status(services::ErrorNullOptionalResult);
    }
    s |= checkGradientSquareSumData(input, pItem, false);
    if(!s) return s;
    s |= checkRngState(input, par, pOpt->get(rngState).get(), false);
    return s;
}

} // namespace interface1
} // namespace adagrad
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

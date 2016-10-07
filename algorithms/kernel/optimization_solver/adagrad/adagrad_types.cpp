/* file: adagrad_types.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

    optimization_solver::iterative_solver::Parameter(function_, nIterations_, accuracyThreshold_),
    batchIndices(batchIndices_),
    batchSize(batchSize_),
    learningRate(learningRate_),
    degenerateCasesThreshold(degenerateCasesThreshold_),
    seed(seed_)
{}

void Parameter::check() const
{
    iterative_solver::Parameter::check();

    if(!data_management::checkNumericTable(learningRate.get(), this->_errors.get(), learningRateStr(), 0, 0, 1, 1))
        return;

    if(batchSize > function->sumOfFunctionsParameter->numberOfTerms || batchSize == 0)
        this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, batchSizeStr()));

    if(batchIndices)
        data_management::checkNumericTable(batchIndices.get(), this->_errors.get(), batchIndicesStr(), 0, 0, batchSize, nIterations);
}

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

static bool checkGradientSquareSumData(const daal::algorithms::Input *input, const data_management::SerializationIfacePtr& pItem,
    services::ErrorCollection* errors, bool bInput)
{
    const services::ErrorDetailID det = bInput ? services::OptionalInput : services::OptionalResult;
    data_management::NumericTablePtr pData = data_management::NumericTable::cast(pItem);
    if(!pData.get())
    {
        errors->add(services::Error::create(bInput ?
            services::ErrorIncorrectOptionalInput : services::ErrorIncorrectOptionalResult, det, gradientSquareSumStr()));
        return false;
    }
    const Input* algInput = static_cast<const Input *>(input);
    auto arg = algInput->get(iterative_solver::inputArgument);
    if(pData->getNumberOfColumns() != arg->getNumberOfColumns())
    {
        errors->add(services::Error::create(services::ErrorIncorrectNumberOfColumns, det, gradientSquareSumStr()));
        return false;
    }
    if(pData->getNumberOfRows() != arg->getNumberOfRows())
    {
        errors->add(services::Error::create(services::ErrorIncorrectNumberOfRows, det, gradientSquareSumStr()));
        return false;
    }
    return true;
}

static bool checkRngState(const daal::algorithms::Input *input,
    const daal::algorithms::Parameter *par,
    const data_management::SerializationIface* pItem,
    services::ErrorCollection* errors, bool bInput)
{
    const Parameter *algParam = static_cast<const Parameter *>(par);
    //if random numbers generator in the algorithm is not required
    if(algParam->batchIndices.get())
        return true;// rgnState doesn't matter

    //but if it is present then the SerializationIface should be an instance of expected type
    if(pItem)
    {
        if(!dynamic_cast<const data_management::MemoryBlock*>(pItem))
        {
            const services::ErrorDetailID det = bInput ? services::OptionalInput : services::OptionalResult;
            errors->add(services::Error::create(bInput ?
                services::ErrorIncorrectOptionalInput : services::ErrorIncorrectOptionalResult, det, rngStateStr()));
            return false;
        }
    }
    else if(!bInput)
    {
        errors->add(services::Error::create(services::ErrorNullOptionalResult, services::OptionalResult, rngStateStr()));
        return false;
    }
    return true;
}

void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    super::check(par, method);
    if(this->_errors->size())
        return;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(!pOpt.get())
        return;//ok
    if(pOpt->size() != optionalDataSize)
    {
        this->_errors->add(services::ErrorIncorrectOptionalInput);
        return;
    }
    auto pItem = pOpt->get(gradientSquareSum);
    if(pItem.get() && !checkGradientSquareSumData(this, pItem, this->_errors.get(), true))
        return;
    if(!checkRngState(this, par, pOpt->get(rngState).get(), this->_errors.get(), true))
        return;
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

void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
    int method) const
{
    super::check(input, par, method);
    if(this->_errors->size() || !static_cast<const Parameter*>(par)->optionalResultRequired)
        return;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(!pOpt.get())
    {
        this->_errors->add(services::ErrorNullOptionalResult);
        return;
    }
    if(pOpt->size() != optionalDataSize)
    {
        this->_errors->add(services::ErrorIncorrectOptionalResult);
        return;
    }
    auto pItem = pOpt->get(gradientSquareSum);
    if(!pItem.get())
    {
        this->_errors->add(services::ErrorNullOptionalResult);
        return;
    }
    if(!checkGradientSquareSumData(input, pItem, this->_errors.get(), false))
        return;
    if(!checkRngState(input, par, pOpt->get(rngState).get(), this->_errors.get(), false))
        return;
}

} // namespace interface1
} // namespace adagrad
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

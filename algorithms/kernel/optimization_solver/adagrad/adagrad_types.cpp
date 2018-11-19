/* file: adagrad_types.cpp */
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

Parameter::Parameter(
    const sum_of_functions::BatchPtr& function_, size_t nIterations_,
    double accuracyThreshold_, data_management::NumericTablePtr batchIndices_,
    const size_t batchSize_, data_management::NumericTablePtr learningRate_,
    double degenerateCasesThreshold_, size_t seed_) :

    optimization_solver::iterative_solver::Parameter(function_, nIterations_, accuracyThreshold_, false, batchSize_),
    batchIndices(batchIndices_),
    learningRate(learningRate_),
    degenerateCasesThreshold(degenerateCasesThreshold_),
    seed(seed_),
    engine(engines::mt19937::Batch<>::create())
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
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
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

services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    services::Status s = super::check(par, method);
    if(!s) return s;
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(!pOpt.get())
        return services::Status();//ok
    if(pOpt->size() != lastOptionalData + 1)
    {
        return services::Status(services::ErrorIncorrectOptionalInput);
    }
    auto pItem = pOpt->get(gradientSquareSum);
    if(pItem.get())
    {
        s |= checkGradientSquareSumData(this, pItem, true);
        if(!s) return s;
    }
    return s;
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
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
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
    if(pOpt->size() != lastOptionalData + 1)
    {
        return services::Status(services::ErrorIncorrectOptionalResult);
    }
    auto pItem = pOpt->get(gradientSquareSum);
    if(!pItem.get())
    {
        return services::Status(services::ErrorNullOptionalResult);
    }
    s |= checkGradientSquareSumData(input, pItem, false);
    return s;
}

} // namespace interface1
} // namespace adagrad
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

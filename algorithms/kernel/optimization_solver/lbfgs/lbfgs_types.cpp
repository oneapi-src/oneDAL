/* file: lbfgs_types.cpp */
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
//  Implementation of lbfgs solver classes.
//--
*/

#include "algorithms/optimization_solver/lbfgs/lbfgs_types.h"
#include "numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace interface1
{

Parameter::Parameter(sum_of_functions::BatchPtr function, size_t nIterations, double accuracyThreshold,
                     size_t batchSize, size_t correctionPairBatchSize, size_t m, size_t L, size_t seed) :
    optimization_solver::iterative_solver::Parameter(function, nIterations, accuracyThreshold),
    batchSize(batchSize), correctionPairBatchSize(correctionPairBatchSize), m(m), L(L), seed(seed),
    stepLengthSequence(new HomogenNumericTable<>(1, 1, NumericTableIface::doAllocate, 1.0))
{}

void Parameter::check() const
{
    iterative_solver::Parameter::check();

    DAAL_CHECK_EX(m != 0, ErrorIncorrectParameter, ArgumentName, "m");
    DAAL_CHECK_EX(L != 0, ErrorIncorrectParameter, ArgumentName, "L");
    DAAL_CHECK_EX(batchSize != 0, ErrorIncorrectParameter, ArgumentName, "batchSize");

    if(batchIndices.get() != NULL)
    {
        if(!checkNumericTable(batchIndices.get(), this->_errors.get(), batchIndicesStr(), 0, 0, batchSize, nIterations)) {return;}
    }

    if(correctionPairBatchIndices.get() != NULL)
    {
        if(!checkNumericTable(correctionPairBatchIndices.get(), this->_errors.get(), correctionPairBatchIndicesStr(), 0, 0,
                              correctionPairBatchSize, (nIterations / L))) {return;}
    }

    if(stepLengthSequence.get() != NULL)
    {
        if(stepLengthSequence->getNumberOfColumns() != 1 && stepLengthSequence->getNumberOfColumns() != nIterations)
        {
            this->_errors->add(Error::create(ErrorIncorrectNumberOfFeatures,
                                             ArgumentName, stepLengthSequenceStr()));
            return;
        }
        if(!checkNumericTable(stepLengthSequence.get(), this->_errors.get(), stepLengthSequenceStr(), 0, 0, 0, 1)) {return;}
    }
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

static bool checkCorrectionPairsData(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
                                     const SerializationIfacePtr &pItem,
                                     ErrorCollection *errors, bool bInput)
{
    const ErrorDetailID det = bInput ? OptionalInput : OptionalResult;
    NumericTablePtr pData = NumericTable::cast(pItem);
    if(!pData.get())
    {
        errors->add(Error::create(bInput ?
                                  ErrorIncorrectOptionalInput : ErrorIncorrectOptionalResult, det, correctionPairsStr()));
        return false;
    }
    const Input *algInput = static_cast<const Input *>(input);
    auto arg = algInput->get(iterative_solver::inputArgument);
    if(pData->getNumberOfColumns() != arg->getNumberOfRows())
    {
        errors->add(Error::create(ErrorIncorrectNumberOfColumns, det, correctionPairsStr()));
        return false;
    }
    const Parameter *algParam = static_cast<const Parameter *>(par);
    if(pData->getNumberOfRows() != 2 * algParam->m)
    {
        errors->add(Error::create(ErrorIncorrectNumberOfRows, det, correctionPairsStr()));
        return false;
    }
    return true;
}

static bool checkCorrectionIndexData(const SerializationIfacePtr &pItem,
                                     ErrorCollection *errors, bool bInput)
{
    const ErrorDetailID det = bInput ? OptionalInput : OptionalResult;
    NumericTablePtr pData = NumericTable::cast(pItem);
    if(!pData.get())
    {
        errors->add(Error::create(bInput ?
                                  ErrorIncorrectOptionalInput : ErrorIncorrectOptionalResult, det, correctionIndicesStr()));
        return false;
    }
    if(pData->getNumberOfColumns() != 2)
    {
        errors->add(Error::create(ErrorIncorrectNumberOfColumns, det, correctionIndicesStr()));
        return false;
    }
    if(pData->getNumberOfRows() != 1)
    {
        errors->add(Error::create(ErrorIncorrectNumberOfRows, det, correctionIndicesStr()));
        return false;
    }
    return true;
}

static bool checkAverageArgumentLIterations(const daal::algorithms::Input *input,
        const SerializationIfacePtr &pItem,
        ErrorCollection *errors, bool bInput)
{
    const ErrorDetailID det = bInput ? OptionalInput : OptionalResult;
    NumericTablePtr pData = NumericTable::cast(pItem);
    if(!pData.get())
    {
        errors->add(Error::create(bInput ?
                                  ErrorIncorrectOptionalInput : ErrorIncorrectOptionalResult, det, averageArgumentLIterationsStr()));
        return false;
    }
    const Input *algInput = static_cast<const Input *>(input);
    auto arg = algInput->get(iterative_solver::inputArgument);
    if(pData->getNumberOfColumns() != arg->getNumberOfRows())
    {
        errors->add(Error::create(ErrorIncorrectNumberOfColumns, det, averageArgumentLIterationsStr()));
        return false;
    }
    if(pData->getNumberOfRows() != 2)
    {
        errors->add(Error::create(ErrorIncorrectNumberOfRows, det, averageArgumentLIterationsStr()));
        return false;
    }
    return true;
}

static bool checkRngState(const daal::algorithms::Input *input,
                          const daal::algorithms::Parameter *par,
                          const SerializationIface *pItem,
                          ErrorCollection *errors, bool bInput)
{
    const Parameter *algParam = static_cast<const Parameter *>(par);
    //if random numbers generator in the algorithm is not required
    if(algParam->batchIndices.get() && algParam->correctionPairBatchIndices.get())
    {
        return true;    // rgnState doesn't matter
    }

    //but if it is present then the SerializationIface should be an instance of expected type
    if(pItem)
    {
        if(!dynamic_cast<const MemoryBlock *>(pItem))
        {
            const ErrorDetailID det = bInput ? OptionalInput : OptionalResult;
            errors->add(Error::create(bInput ?
                                      ErrorIncorrectOptionalInput : ErrorIncorrectOptionalResult, det, rngStateStr()));
            return false;
        }
    }
    else if(!bInput)
    {
        errors->add(Error::create(ErrorNullOptionalResult, OptionalResult, rngStateStr()));
        return false;
    }
    return true;
}

void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    super::check(par, method);
    if(this->_errors->size())
    {
        return;
    }
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalArgument);
    if(!pOpt.get())
    {
        return;    //ok
    }

    if(pOpt->size() != optionalDataSize)
    {
        this->_errors->add(ErrorIncorrectOptionalInput);
        return;
    }
    //checking correction pairs table
    {
        auto pItem = pOpt->get(correctionPairs);
        if(pItem.get() && !checkCorrectionPairsData(this, par, pItem, this->_errors.get(), true))
        {
            return;
        }
    }

    //checking correction index table
    {
        auto pItem = pOpt->get(correctionIndices);
        if(pItem.get() && !checkCorrectionIndexData(pItem, this->_errors.get(), true))
        {
            return;
        }
    }

    //checking average argument for L iterations table
    {
        auto pItem = pOpt->get(averageArgumentLIterations);
        if(pItem.get() && !checkAverageArgumentLIterations(this, pItem, this->_errors.get(), true))
        {
            return;
        }
    }
    //checking rng state
    {
        if(!checkRngState(this, par, pOpt->get(rngState).get(), this->_errors.get(), true))
        {
            return;
        }
    }
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

void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
                   int method) const
{
    super::check(input, par, method);
    if(this->_errors->size() || !static_cast<const Parameter *>(par)->optionalResultRequired)
    {
        return;
    }
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if(!pOpt.get())
    {
        this->_errors->add(ErrorNullOptionalResult);
        return;
    }
    if(pOpt->size() != optionalDataSize)
    {
        this->_errors->add(ErrorIncorrectOptionalResult);
        return;
    }
    //checking correction pairs table
    {
        auto pItem = pOpt->get(correctionPairs);
        if(!pItem.get())
        {
            this->_errors->add(Error::create(ErrorNullOptionalResult, OptionalResult, correctionPairsStr()));
            return;
        }
        if(!checkCorrectionPairsData(input, par, pItem, this->_errors.get(), false))
        {
            return;
        }
    }

    //checking correction index table
    {
        auto pItem = pOpt->get(correctionIndices);
        if(!pItem.get())
        {
            this->_errors->add(Error::create(ErrorNullOptionalResult, OptionalResult, correctionIndicesStr()));
            return;
        }
        if(!checkCorrectionIndexData(pItem, this->_errors.get(), false))
        {
            return;
        }
    }

    //checking average argument for L iterations table
    {
        auto pItem = pOpt->get(averageArgumentLIterations);
        if(!pItem.get())
        {
            this->_errors->add(Error::create(ErrorNullOptionalResult,
                                             OptionalResult, averageArgumentLIterationsStr()));
            return;
        }
        if(!checkAverageArgumentLIterations(input, pItem, this->_errors.get(), false))
        {
            return;
        }
    }
    //checking rng state
    {
        if(!checkRngState(input, par, pOpt->get(rngState).get(), this->_errors.get(), false))
        {
            return;
        }
    }
}

} // namespace interface1
} // namespace lbfgs
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

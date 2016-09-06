/* file: adagrad_dense_default_impl.i */
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
//  Implementation of adagrad algorithm
//--
*/

#ifndef __ADAGRAD_DENSE_DEFAULT_IMPL_I__
#define __ADAGRAD_DENSE_DEFAULT_IMPL_I__

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_utils.h"
#include "service_numeric_table.h"
#include "algorithms/optimization_solver/adagrad/adagrad_types.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void initAccumulatedGrad(algorithmFPType *accumulatedG,
    size_t nFeatures, NumericTable* pOptInput, algorithmFPType degenerateCasesThreshold)
{
    if(pOptInput)
    {
        /* Initialize accumulatedG and invAccumGrad from optional input data */
        ReadRows<algorithmFPType, cpu> optInputBD(*pOptInput, 0, 1);
        daal_memcpy_s(accumulatedG, nFeatures * sizeof(algorithmFPType), optInputBD.get(), nFeatures * sizeof(algorithmFPType));
    }
    else
    {
        for(size_t i = 0; i < nFeatures; i++)
            accumulatedG[i] = algorithmFPType(0.0);
    }
}

template<typename algorithmFPType, CpuType cpu>
class RngTask
{
public:
    RngTask(const int* predefined, size_t size) :
        _predefined(predefined), _size(size), _rngChanged(false), _maxVal(0), _values(0), _rng(nullptr){}
    ~RngTask()
    {
        delete _rng;
    }
    bool init(OptionalArgument *optionalArgument, const Parameter *parameter)
    {
        _values.reset(_size);
        if(!_values.get())
            return false;
        _maxVal = parameter->function->sumOfFunctionsParameter->numberOfTerms;
        _rng = new IntRngType((int)parameter->seed);
        auto pOpt = optionalArgument;
        if(pOpt)
        {
            data_management::MemoryBlock* pState = dynamic_cast<data_management::MemoryBlock*>(pOpt->get(adagrad::rngState).get());
            if(pState && pState->size())
                _rng->loadStream(pState->get());
        }
        return true;
    }
    const int* get()
    {
        if(_predefined)
        {
            auto ptr = _predefined;
            _predefined += _size;
            return ptr;
        }
        _rng->uniform((int)_size, 0, _maxVal, _values.get());
        _rngChanged = true;
        return _values.get();
    }
    bool save(OptionalArgument *optionalResult) const
    {
        if(!_rng || !_rngChanged)
            return true;
        auto pOpt = optionalResult;
        //pOpt should exist by now
        data_management::MemoryBlock* pState = dynamic_cast<data_management::MemoryBlock*>(pOpt->get(adagrad::rngState).get());
        //pState should exist by now
        auto stateSize = _rng->getStreamSize();
        if(!stateSize)
            return true;
        pState->reserve(stateSize);
        if(!pState->get())
            return false;
        _rng->saveStream(pState->get());
        return true;
    }

protected:
    typedef daal::internal::IntRng <int, cpu> IntRngType;
    IntRngType* _rng;
    const int* _predefined;
    size_t _size;
    bool _rngChanged;
    TSmartPtr<int, cpu> _values;
    int _maxVal;
};

/**
 *  \brief Kernel for Adagrad calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void AdagradKernel<algorithmFPType, method, cpu>::compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
                                                          NumericTable *gradientSquareSumResult, NumericTable *gradientSquareSumInput,
                                                          OptionalArgument *optionalArgument, OptionalArgument *optionalResult, Parameter *parameter)
{
    const size_t nFeatures = inputArgument->getNumberOfColumns();

    WriteRows<algorithmFPType, cpu> workValueBD(*minimum, 0, 1);
    algorithmFPType *workValue = workValueBD.get();

    //init workValue
    {
        ReadRows<algorithmFPType, cpu> startValueBD(*inputArgument, 0, 1);
        const algorithmFPType *startValueArray = startValueBD.get();
        daal_memcpy_s(workValue, nFeatures * sizeof(algorithmFPType), startValueArray, nFeatures * sizeof(algorithmFPType));
    }

    const size_t maxIterations = parameter->nIterations;
    WriteRows<int, cpu> nIterationsBD(*nIterations, 0, 1);
    int *nProceededIterations = nIterationsBD.get();
    /* if maxIterations == 0, set result as start point, the number of executed iters to 0 */
    if(maxIterations == 0)
    {
        nProceededIterations[0] = 0;
        return;
    }

    sum_of_functions::BatchPtr function = parameter->function;

    /*Get random indices for SGD from parameter or from rng generator*/
    const bool isPredefinedBatchIndices = parameter->batchIndices;
    const size_t batchSize = parameter->batchSize;
    const algorithmFPType degenerateCasesThreshold = (algorithmFPType)parameter->degenerateCasesThreshold;
    ReadRows<int, cpu> predefinedBatchIndicesBD(parameter->batchIndices.get(), 0, maxIterations);
    RngTask<int, cpu> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    if((!parameter->batchIndices) && !rngTask.init(optionalArgument, parameter))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    SharedPtr<HomogenNumericTableCPU<int, cpu>> ntBatchIndices(new HomogenNumericTableCPU<int, cpu>(NULL, batchSize, 1));
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, NumericTablePtr(new HomogenNumericTableCPU<algorithmFPType, cpu>(workValue, nFeatures, 1)));

    NumericTablePtr ntlearningRate = parameter->learningRate;
    ReadRows<algorithmFPType, cpu> learningRateBD(*ntlearningRate, 0, 1);
    const algorithmFPType learningRate = *learningRateBD.get();

    *nProceededIterations = (int)maxIterations;

    SmartPtr<cpu> smAccumulatedG(nFeatures * sizeof(algorithmFPType));
    algorithmFPType *accumulatedG = (algorithmFPType *)smAccumulatedG.get();

    initAccumulatedGrad<algorithmFPType, method, cpu>(accumulatedG, nFeatures, gradientSquareSumInput, degenerateCasesThreshold);

    for(size_t epoch = 0; epoch < maxIterations; epoch++)
    {
        ntBatchIndices->setArray(const_cast<int*>(rngTask.get()));
        function->computeNoThrow();
        if(function->getErrors()->size() != 0)
        {
            this->_errors->add(function->getErrors()->getErrors());
            *nProceededIterations = (int)epoch;
            break;
        }

        auto ntGradient = function->getResult()->get(objective_function::resultCollection, objective_function::gradientIdx);
        ReadRows<algorithmFPType, cpu> ntGradientBD(*ntGradient, 0, 1);
        const algorithmFPType *gradient = ntGradientBD.get();
        const algorithmFPType pointNorm = vectorNorm(workValue, nFeatures);
        const algorithmFPType gradientNorm = vectorNorm(gradient, nFeatures);
        const algorithmFPType gradientThreshold = parameter->accuracyThreshold * daal::internal::Math<algorithmFPType,cpu>::sMax(algorithmFPType(1.0), pointNorm);
        if(gradientNorm <= gradientThreshold)
        {
            *nProceededIterations = (int)epoch;
            if(parameter->optionalResultRequired)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for(size_t j = 0; j < nFeatures; j++)
                    accumulatedG[j] += gradient[j] * gradient[j];
            }
            break;
        }

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t j = 0; j < nFeatures; j++)
        {
            accumulatedG[j] += gradient[j] * gradient[j];
            const algorithmFPType invAccumGrad = algorithmFPType(1.0) / daal::internal::Math<algorithmFPType,cpu>::sSqrt(accumulatedG[j] + degenerateCasesThreshold);
            workValue[j] = workValue[j] - learningRate * gradient[j] * invAccumGrad;
        }
    }
    if(!parameter->optionalResultRequired)
        return;
    NumericTable *pOptResult = gradientSquareSumResult;
    /* Copy accumulatedG to output */
    WriteRows<algorithmFPType, cpu> optResultBD(*pOptResult, 0, 1);
    daal_memcpy_s(optResultBD.get(), nFeatures * sizeof(algorithmFPType), accumulatedG, nFeatures * sizeof(algorithmFPType));
    if(!rngTask.save(optionalResult))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }
}

} // namespace daal::internal

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

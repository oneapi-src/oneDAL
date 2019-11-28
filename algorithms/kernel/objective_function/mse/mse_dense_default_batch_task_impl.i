/* file: mse_dense_default_batch_task_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of mse algorithm
//--
*/

#include "mse_dense_default_batch_kernel.h"
#include "service_blas.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace mse
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;
static const size_t blockSizeDefault = 512;

template <typename algorithmFPType, CpuType cpu>
MSETask<algorithmFPType, cpu>::MSETask(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument, NumericTable * value,
                                       NumericTable * hessian, NumericTable * gradient, Parameter * parameter)
    : ntData(data), ntDependentVariables(dependentVariables), ntArgument(argument), ntValue(value), ntHessian(hessian), ntGradient(gradient)
{
    valueFlag    = ((parameter->resultsToCompute & objective_function::value) != 0) ? true : false;
    hessianFlag  = ((parameter->resultsToCompute & objective_function::hessian) != 0) ? true : false;
    gradientFlag = ((parameter->resultsToCompute & objective_function::gradient) != 0) ? true : false;

    argumentSize = ntArgument->getNumberOfRows();
    nTheta       = argumentSize - 1;
}

template <typename algorithmFPType, CpuType cpu>
Status MSETask<algorithmFPType, cpu>::init(algorithmFPType *& pArgumentArray)
{
    Status s       = ntArgument->getBlockOfRows(0, argumentSize, readOnly, argumentBlock);
    pArgumentArray = argumentBlock.getBlockPtr();
    return s;
}

template <typename algorithmFPType, CpuType cpu>
MSETask<algorithmFPType, cpu>::~MSETask()
{
    ntArgument->releaseBlockOfRows(argumentBlock);
}

template <typename algorithmFPType, CpuType cpu>
Status MSETask<algorithmFPType, cpu>::getResultValues(algorithmFPType *& value, algorithmFPType *& gradient, algorithmFPType *& hessian)
{
    Status s;
    if (valueFlag)
    {
        DAAL_CHECK_STATUS(s, ntValue->getBlockOfRows(0, 1, writeOnly, valueBlock));
        value = valueBlock.getBlockPtr();
    }
    if (hessianFlag)
    {
        DAAL_CHECK_STATUS(s, ntHessian->getBlockOfRows(0, argumentSize, writeOnly, hessianBlock));
        hessian = hessianBlock.getBlockPtr();
    }
    if (gradientFlag)
    {
        DAAL_CHECK_STATUS(s, ntGradient->getBlockOfRows(0, argumentSize, writeOnly, gradientBlock));
        gradient = gradientBlock.getBlockPtr();
    }
    return s;
}

template <typename algorithmFPType, CpuType cpu>
void MSETask<algorithmFPType, cpu>::setResultValuesToZero(algorithmFPType * value, algorithmFPType * gradient, algorithmFPType * hessian)
{
    algorithmFPType zero = (algorithmFPType)0.0;
    if (valueFlag)
    {
        value[0] = zero;
    }
    if (hessianFlag)
    {
        for (size_t j = 0; j < argumentSize * argumentSize; j++)
        {
            hessian[j] = zero;
        }
    }
    if (gradientFlag)
    {
        for (size_t j = 0; j < argumentSize; j++)
        {
            gradient[j] = zero;
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
void MSETask<algorithmFPType, cpu>::releaseResultValues()
{
    if (valueFlag)
    {
        ntValue->releaseBlockOfRows(valueBlock);
    }
    if (hessianFlag)
    {
        ntHessian->releaseBlockOfRows(hessianBlock);
    }
    if (gradientFlag)
    {
        ntGradient->releaseBlockOfRows(gradientBlock);
    }
}

template <typename algorithmFPType, CpuType cpu>
MSETaskAll<algorithmFPType, cpu>::MSETaskAll(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument, NumericTable * value,
                                             NumericTable * hessian, NumericTable * gradient, Parameter * parameter, size_t blockSizeDefault)
    : super(data, dependentVariables, argument, value, hessian, gradient, parameter)
{
    batchSize = ntData->getNumberOfRows();
    xMultTheta.reset(batchSize < blockSizeDefault ? batchSize : blockSizeDefault);
}

template <typename algorithmFPType, CpuType cpu>
Status MSETaskAll<algorithmFPType, cpu>::init(algorithmFPType *& pArgumentArray)
{
    Status s = super::init(pArgumentArray);
    if (!s) return s;
    DAAL_CHECK_MALLOC(xMultTheta.get());
    return s;
}

template <typename algorithmFPType, CpuType cpu>
MSETaskAll<algorithmFPType, cpu>::~MSETaskAll()
{}

template <typename algorithmFPType, CpuType cpu>
Status MSETaskAll<algorithmFPType, cpu>::getCurrentBlock(size_t startIdx, size_t blockSize, algorithmFPType *& pBlockData,
                                                         algorithmFPType *& pBlockDependentVariables)
{
    Status s;
    DAAL_CHECK_STATUS(s, ntData->getBlockOfRows(startIdx, blockSize, readOnly, dataBlock));
    pBlockData = dataBlock.getBlockPtr();
    DAAL_CHECK_STATUS(s, ntDependentVariables->getBlockOfRows(startIdx, blockSize, readOnly, dependentVariablesBlock));
    pBlockDependentVariables = dependentVariablesBlock.getBlockPtr();
    return s;
}

template <typename algorithmFPType, CpuType cpu>
void MSETaskAll<algorithmFPType, cpu>::releaseCurrentBlock()
{
    ntData->releaseBlockOfRows(dataBlock);
    ntDependentVariables->releaseBlockOfRows(dependentVariablesBlock);
}

template <typename algorithmFPType, CpuType cpu>
MSETaskSample<algorithmFPType, cpu>::MSETaskSample(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument,
                                                   NumericTable * value, NumericTable * hessian, NumericTable * gradient, Parameter * parameter,
                                                   size_t blockSizeDefault)
    : MSETask<algorithmFPType, cpu>(data, dependentVariables, argument, value, hessian, gradient, parameter), ntIndices(parameter->batchIndices.get())
{
    batchSize = parameter->batchIndices->getNumberOfColumns();
}

template <typename algorithmFPType, CpuType cpu>
Status MSETaskSample<algorithmFPType, cpu>::init(algorithmFPType *& pArgumentArray)
{
    Status s = super::init(pArgumentArray);
    if (s) s = ntIndices->getBlockOfRows(0, 1, readOnly, indicesBlock);
    if (!s) return s;
    indicesArray                = indicesBlock.getBlockPtr();
    const size_t allocationSize = (batchSize < blockSizeDefault ? batchSize : blockSizeDefault);

    if (nTheta > 0)
    {
        dataBlockMemory.reset(allocationSize * nTheta);
        DAAL_CHECK_MALLOC(dataBlockMemory.get());
    }
    dependentVariablesBlockMemory.reset(allocationSize);
    xMultTheta.reset(allocationSize);
    DAAL_CHECK_MALLOC(dependentVariablesBlockMemory.get() && xMultTheta.get());
    return s;
}

template <typename algorithmFPType, CpuType cpu>
MSETaskSample<algorithmFPType, cpu>::~MSETaskSample()
{
    ntIndices->releaseBlockOfRows(indicesBlock);
}

template <typename algorithmFPType, CpuType cpu>
Status MSETaskSample<algorithmFPType, cpu>::getCurrentBlock(size_t startIdx, size_t blockSize, algorithmFPType *& pBlockData,
                                                            algorithmFPType *& pBlockDependentVariables)
{
    Status s;
    algorithmFPType *dataArray = nullptr, *dependentVariablesArray = nullptr;
    size_t index;
    pBlockData               = dataBlockMemory.get();
    pBlockDependentVariables = dependentVariablesBlockMemory.get();
    for (size_t idx = 0; idx < blockSize; idx++)
    {
        index = indicesArray[startIdx + idx];
        DAAL_CHECK_STATUS(s, ntData->getBlockOfRows(index, 1, readOnly, dataBlock));
        dataArray = dataBlock.getBlockPtr();
        DAAL_CHECK_STATUS(s, ntDependentVariables->getBlockOfRows(index, 1, readOnly, dependentVariablesBlock));
        dependentVariablesArray = dependentVariablesBlock.getBlockPtr();

        for (size_t j = 0; j < nTheta; j++)
        {
            pBlockData[idx * nTheta + j] = dataArray[j];
        }
        pBlockDependentVariables[idx] = dependentVariablesArray[0];

        ntData->releaseBlockOfRows(dataBlock);
        ntDependentVariables->releaseBlockOfRows(dependentVariablesBlock);
    }
    return s;
}

template <typename algorithmFPType, CpuType cpu>
void MSETaskSample<algorithmFPType, cpu>::releaseCurrentBlock()
{}

} // namespace internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

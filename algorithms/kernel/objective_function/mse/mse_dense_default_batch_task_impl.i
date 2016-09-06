/* file: mse_dense_default_batch_task_impl.i */
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


template<typename algorithmFPType, CpuType cpu>
MSETask<algorithmFPType, cpu>::MSETask(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
                                       NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter,
                                       algorithmFPType **pArgumentArray) :
    error(NoErrorMessageFound),
    ntData(data),
    ntDependentVariables(dependentVariables),
    ntArgument(argument)
{
    ntValue    = value;
    ntHessian  = hessian;
    ntGradient = gradient;

    valueFlag    = ((parameter->resultsToCompute & objective_function::value) != 0) ? true : false;
    hessianFlag  = ((parameter->resultsToCompute & objective_function::hessian) != 0) ? true : false;
    gradientFlag = ((parameter->resultsToCompute & objective_function::gradient) != 0) ? true : false;

    nFeatures = ntArgument->getNumberOfColumns();
    nTheta = nFeatures - 1;

    ntArgument->getBlockOfRows(0, 1, readOnly, argumentBlock);
    *pArgumentArray = argumentBlock.getBlockPtr();
    if(!(*pArgumentArray)) {error.setId(ErrorMemoryAllocationFailed); return;}
}

template<typename algorithmFPType, CpuType cpu>
MSETask<algorithmFPType, cpu>::~MSETask()
{
    ntArgument->releaseBlockOfRows(argumentBlock);
}

template<typename algorithmFPType, CpuType cpu>
void MSETask<algorithmFPType, cpu>::getResultValues(
    algorithmFPType **value,
    algorithmFPType **gradient,
    algorithmFPType **hessian)
{
    if(valueFlag)
    {
        ntValue->getBlockOfRows(0, 1, writeOnly, valueBlock);
        *value = valueBlock.getBlockPtr();
        if(!value) {error.setId(ErrorMemoryAllocationFailed); return;}
    }
    if(hessianFlag)
    {
        ntHessian->getBlockOfRows(0, nFeatures, writeOnly, hessianBlock);
        *hessian = hessianBlock.getBlockPtr();
        if(!hessian) {error.setId(ErrorMemoryAllocationFailed); return;}
    }
    if(gradientFlag)
    {
        ntGradient->getBlockOfRows(0, 1, writeOnly, gradientBlock);
        *gradient = gradientBlock.getBlockPtr();
        if(!gradient) {error.setId(ErrorMemoryAllocationFailed); return;}
    }
}

template<typename algorithmFPType, CpuType cpu>
void MSETask<algorithmFPType, cpu>::setResultValuesToZero(
    algorithmFPType **value,
    algorithmFPType **gradient,
    algorithmFPType **hessian)
{
    algorithmFPType zero = (algorithmFPType) 0.0;
    if(valueFlag)
    {
        (*value)[0] = zero;
    }
    if(hessianFlag)
    {
        for(size_t j = 0; j < nFeatures * nFeatures; j++)
        {
            (*hessian)[j] = zero;
        }
    }
    if(gradientFlag)
    {
        for(size_t j = 0; j < nFeatures; j++)
        {
            (*gradient)[j] = zero;
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
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

template<typename algorithmFPType, CpuType cpu>
MSETaskAll<algorithmFPType, cpu>::MSETaskAll(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument, NumericTable *value,
                                             NumericTable *hessian, NumericTable *gradient, Parameter *parameter, size_t blockSizeDefault,
                                             algorithmFPType **pArgumentArray) :
    MSETask<algorithmFPType, cpu>(data, dependentVariables, argument, value, hessian, gradient, parameter, pArgumentArray)
{
    batchSize = ntData->getNumberOfRows();
    if(batchSize < blockSizeDefault)
    {
        xMultTheta = (algorithmFPType *) daal_malloc(batchSize * sizeof(algorithmFPType));
    }
    else
    {
        xMultTheta = (algorithmFPType *) daal_malloc(blockSizeDefault * sizeof(algorithmFPType));
    }
    if(!xMultTheta) {error.setId(ErrorMemoryAllocationFailed); return;}
}

template<typename algorithmFPType, CpuType cpu>
MSETaskAll<algorithmFPType, cpu>::~MSETaskAll()
{
    daal_free(xMultTheta);
}

template<typename algorithmFPType, CpuType cpu>
void MSETaskAll<algorithmFPType, cpu>::getCurrentBlock(
    size_t startIdx,
    size_t blockSize,
    algorithmFPType **pBlockData,
    algorithmFPType **pBlockDependentVariables)
{
    ntData->getBlockOfRows(startIdx, blockSize, readOnly, dataBlock);
    *pBlockData = dataBlock.getBlockPtr();
    if(!(*pBlockData)) {error.setId(ErrorMemoryAllocationFailed); return;}

    ntDependentVariables->getBlockOfRows(startIdx, blockSize, readOnly, dependentVariablesBlock);
    *pBlockDependentVariables = dependentVariablesBlock.getBlockPtr();
    if(!(*pBlockDependentVariables)) {error.setId(ErrorMemoryAllocationFailed); return;}
}

template<typename algorithmFPType, CpuType cpu>
void MSETaskAll<algorithmFPType, cpu>::releaseCurrentBlock()
{
    ntData->releaseBlockOfRows(dataBlock);
    ntDependentVariables->releaseBlockOfRows(dependentVariablesBlock);
}

template<typename algorithmFPType, CpuType cpu>
MSETaskSample<algorithmFPType, cpu>::MSETaskSample(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
                                                   NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter, size_t blockSizeDefault,
                                                   algorithmFPType **pArgumentArray) :
    MSETask<algorithmFPType, cpu>(data, dependentVariables, argument, value, hessian, gradient, parameter, pArgumentArray),
    dataBlockMemory(NULL),
    dependentVariablesBlockMemory(NULL),
    ntIndices(parameter->batchIndices.get())
{
    ntIndices->getBlockOfRows(0, 1, readOnly, indicesBlock);
    indicesArray = indicesBlock.getBlockPtr();
    if(!indicesArray) {error.setId(ErrorMemoryAllocationFailed); return;}

    batchSize = ntIndices->getNumberOfColumns();
    size_t allocationSize = blockSizeDefault;
    if(batchSize < blockSizeDefault)
    {
        allocationSize = batchSize;
    }

    if(nTheta > 0)
    {
        dataBlockMemory = (algorithmFPType *) daal_malloc(allocationSize * nTheta * sizeof(algorithmFPType));
        if(!dataBlockMemory) {error.setId(ErrorMemoryAllocationFailed); return;}
    }
    dependentVariablesBlockMemory = (algorithmFPType *) daal_malloc(allocationSize * sizeof(algorithmFPType));
    if(!dependentVariablesBlockMemory) {error.setId(ErrorMemoryAllocationFailed); return;}

    xMultTheta = (algorithmFPType *) daal_malloc(allocationSize * sizeof(algorithmFPType));
    if(!xMultTheta) {error.setId(ErrorMemoryAllocationFailed); return;}
}

template<typename algorithmFPType, CpuType cpu>
MSETaskSample<algorithmFPType, cpu>::~MSETaskSample()
{
    ntIndices->releaseBlockOfRows(indicesBlock);

    if(dataBlockMemory) {daal_free(dataBlockMemory);}
    if(dependentVariablesBlockMemory) {daal_free(dependentVariablesBlockMemory);}

    daal_free(xMultTheta);
}

template<typename algorithmFPType, CpuType cpu>
void MSETaskSample<algorithmFPType, cpu>::getCurrentBlock(
    size_t startIdx,
    size_t blockSize,
    algorithmFPType **pBlockData,
    algorithmFPType **pBlockDependentVariables)
{
    algorithmFPType *dataArray = NULL, *dependentVariablesArray = NULL;
    size_t index;
    *pBlockData = dataBlockMemory;
    *pBlockDependentVariables = dependentVariablesBlockMemory;
    for(size_t idx = 0; idx < blockSize; idx++)
    {
        index = indicesArray[startIdx + idx];
        ntData->getBlockOfRows(index, 1, readOnly, dataBlock);
        dataArray = dataBlock.getBlockPtr();
        ntDependentVariables->getBlockOfRows(index, 1, readOnly, dependentVariablesBlock);
        dependentVariablesArray = dependentVariablesBlock.getBlockPtr();

        for(size_t j = 0; j < nTheta; j++)
        {
            (*pBlockData)[idx * nTheta + j] = dataArray[j];
        }
        (*pBlockDependentVariables)[idx] = dependentVariablesArray[0];

        ntData->releaseBlockOfRows(dataBlock);
        ntDependentVariables->releaseBlockOfRows(dependentVariablesBlock);
    }
}

template<typename algorithmFPType, CpuType cpu>
void MSETaskSample<algorithmFPType, cpu>::releaseCurrentBlock() {}

} // namespace daal::internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

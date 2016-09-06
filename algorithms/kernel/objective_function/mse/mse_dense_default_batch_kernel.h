/* file: mse_dense_default_batch_kernel.h */
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

//++
//  Declaration of template function that calculate mse.
//--


#ifndef __MSE_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __MSE_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "mse_batch.h"
#include "kernel.h"
#include "service_numeric_table.h"
#include "service_blas.h"
#include "numeric_table.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

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
struct MSETask
{
    MSETask(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
            NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter,
            algorithmFPType **pArgumentArray);
    virtual ~MSETask();

    virtual void getCurrentBlock(size_t startIdx, size_t blockSize, algorithmFPType **pBlockData, algorithmFPType **pBlockDependentVariables) = 0;
    virtual void releaseCurrentBlock() = 0;

    void setResultValuesToZero(algorithmFPType **value, algorithmFPType **gradient, algorithmFPType **hessian);
    void getResultValues(algorithmFPType **value, algorithmFPType **gradient, algorithmFPType **hessian);
    void releaseResultValues();

    BlockDescriptor<algorithmFPType> dataBlock;
    BlockDescriptor<algorithmFPType> dependentVariablesBlock;
    BlockDescriptor<algorithmFPType> argumentBlock;
    BlockDescriptor<algorithmFPType> gradientBlock;
    BlockDescriptor<algorithmFPType> valueBlock;
    BlockDescriptor<algorithmFPType> hessianBlock;

    NumericTable *ntData;
    NumericTable *ntDependentVariables;
    NumericTable *ntArgument;
    NumericTable *ntGradient;
    NumericTable *ntValue;
    NumericTable *ntHessian;

    bool valueFlag;
    bool hessianFlag;
    bool gradientFlag;
    size_t nFeatures;
    size_t nTheta;
    size_t batchSize;
    algorithmFPType *xMultTheta;
    Error error;
};

template<typename algorithmFPType, CpuType cpu>
struct MSETaskAll : public MSETask<algorithmFPType, cpu>
{
    using MSETask<algorithmFPType, cpu>::ntData;
    using MSETask<algorithmFPType, cpu>::dataBlock;

    using MSETask<algorithmFPType, cpu>::ntDependentVariables;
    using MSETask<algorithmFPType, cpu>::dependentVariablesBlock;

    using MSETask<algorithmFPType, cpu>::ntArgument;
    using MSETask<algorithmFPType, cpu>::argumentBlock;

    using MSETask<algorithmFPType, cpu>::ntGradient;
    using MSETask<algorithmFPType, cpu>::gradientBlock;
    using MSETask<algorithmFPType, cpu>::gradientFlag;

    using MSETask<algorithmFPType, cpu>::ntValue;
    using MSETask<algorithmFPType, cpu>::valueBlock;
    using MSETask<algorithmFPType, cpu>::valueFlag;

    using MSETask<algorithmFPType, cpu>::ntHessian;
    using MSETask<algorithmFPType, cpu>::hessianBlock;
    using MSETask<algorithmFPType, cpu>::hessianFlag;

    using MSETask<algorithmFPType, cpu>::nFeatures;
    using MSETask<algorithmFPType, cpu>::nTheta;
    using MSETask<algorithmFPType, cpu>::batchSize;
    using MSETask<algorithmFPType, cpu>::xMultTheta;
    using MSETask<algorithmFPType, cpu>::error;

    MSETaskAll(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument, NumericTable *value, NumericTable *hessian, NumericTable *gradient,
               Parameter *parameter, size_t blockSizeDefault, algorithmFPType **pArgumentArray);
    virtual ~MSETaskAll();

    virtual void getCurrentBlock(size_t startIdx, size_t blockSize, algorithmFPType **pBlockData, algorithmFPType **pBlockDependentVariables);
    virtual void releaseCurrentBlock();
};

template<typename algorithmFPType, CpuType cpu>
struct MSETaskSample : public MSETask<algorithmFPType, cpu>
{
    using MSETask<algorithmFPType, cpu>::ntData;
    using MSETask<algorithmFPType, cpu>::dataBlock;

    using MSETask<algorithmFPType, cpu>::ntDependentVariables;
    using MSETask<algorithmFPType, cpu>::dependentVariablesBlock;

    using MSETask<algorithmFPType, cpu>::ntArgument;
    using MSETask<algorithmFPType, cpu>::argumentBlock;

    using MSETask<algorithmFPType, cpu>::ntGradient;
    using MSETask<algorithmFPType, cpu>::gradientBlock;
    using MSETask<algorithmFPType, cpu>::gradientFlag;

    using MSETask<algorithmFPType, cpu>::ntValue;
    using MSETask<algorithmFPType, cpu>::valueBlock;
    using MSETask<algorithmFPType, cpu>::valueFlag;

    using MSETask<algorithmFPType, cpu>::ntHessian;
    using MSETask<algorithmFPType, cpu>::hessianBlock;
    using MSETask<algorithmFPType, cpu>::hessianFlag;

    using MSETask<algorithmFPType, cpu>::nFeatures;
    using MSETask<algorithmFPType, cpu>::nTheta;
    using MSETask<algorithmFPType, cpu>::batchSize;
    using MSETask<algorithmFPType, cpu>::xMultTheta;
    using MSETask<algorithmFPType, cpu>::error;

    MSETaskSample(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
                  NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter, size_t blockSizeDefault,
                  algorithmFPType **pArgumentArray);
    virtual ~MSETaskSample();

    virtual void getCurrentBlock(size_t startIdx, size_t blockSize, algorithmFPType **pBlockData, algorithmFPType **pBlockDependentVariables);
    virtual void releaseCurrentBlock();

    NumericTable *ntIndices;
    BlockDescriptor<int> indicesBlock;
    int *indicesArray;

    algorithmFPType *dataBlockMemory;
    algorithmFPType *dependentVariablesBlockMemory;
};

template<typename algorithmFPType, Method method, CpuType cpu>
class MSEKernel : public Kernel
{
public:
    void compute(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
                 NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter);
private:
    void computeMSE(
        size_t blockSize,
        MSETask<algorithmFPType, cpu> *task,
        algorithmFPType *data,
        algorithmFPType *argumentArray,
        algorithmFPType *dependentVariablesArray,
        algorithmFPType *value,
        algorithmFPType *gradient,
        algorithmFPType *hessian);

    void normalizeResults(
        MSETask<algorithmFPType, cpu> *task,
        algorithmFPType *value,
        algorithmFPType *gradient,
        algorithmFPType *hessian);

    static const size_t blockSizeDefault = 512;
};

} // namespace daal::internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

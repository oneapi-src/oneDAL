/* file: mse_dense_default_batch_kernel.h */
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

template<typename algorithmFPType, CpuType cpu>
struct MSETask
{
    DAAL_NEW_DELETE();
    MSETask(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
            NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter);
    virtual ~MSETask();

    virtual Status init(algorithmFPType *&pArgumentArray);
    virtual Status getCurrentBlock(size_t startIdx, size_t blockSize,
        algorithmFPType *&pBlockData, algorithmFPType *&pBlockDependentVariables) = 0;
    virtual void releaseCurrentBlock() = 0;

    void setResultValuesToZero(algorithmFPType *value, algorithmFPType *gradient, algorithmFPType *hessian);
    Status getResultValues(algorithmFPType *&value, algorithmFPType *&gradient, algorithmFPType *&hessian);
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
    size_t argumentSize;
    size_t nTheta;
    size_t batchSize;
    TArray<algorithmFPType, cpu> xMultTheta;
};

template<typename algorithmFPType, CpuType cpu>
struct MSETaskAll : public MSETask<algorithmFPType, cpu>
{
    typedef MSETask<algorithmFPType, cpu> super;
    using super::ntData;
    using super::dataBlock;

    using super::ntDependentVariables;
    using super::dependentVariablesBlock;

    using super::ntArgument;
    using super::argumentBlock;

    using super::ntGradient;
    using super::gradientBlock;
    using super::gradientFlag;

    using super::ntValue;
    using super::valueBlock;
    using super::valueFlag;

    using super::ntHessian;
    using super::hessianBlock;
    using super::hessianFlag;

    using super::argumentSize;
    using super::nTheta;
    using super::batchSize;
    using super::xMultTheta;

    MSETaskAll(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument, NumericTable *value, NumericTable *hessian, NumericTable *gradient,
               Parameter *parameter, size_t blockSizeDefault);
    virtual ~MSETaskAll();

    virtual Status init(algorithmFPType *&pArgumentArray) DAAL_C11_OVERRIDE;

    virtual Status getCurrentBlock(size_t startIdx, size_t blockSize,
        algorithmFPType *&pBlockData, algorithmFPType *&pBlockDependentVariables) DAAL_C11_OVERRIDE;
    virtual void releaseCurrentBlock() DAAL_C11_OVERRIDE;
};

template<typename algorithmFPType, CpuType cpu>
struct MSETaskSample : public MSETask<algorithmFPType, cpu>
{
    typedef MSETask<algorithmFPType, cpu> super;

    using super::ntData;
    using super::dataBlock;

    using super::ntDependentVariables;
    using super::dependentVariablesBlock;

    using super::ntArgument;
    using super::argumentBlock;

    using super::ntGradient;
    using super::gradientBlock;
    using super::gradientFlag;

    using super::ntValue;
    using super::valueBlock;
    using super::valueFlag;

    using super::ntHessian;
    using super::hessianBlock;
    using super::hessianFlag;

    using super::argumentSize;
    using super::nTheta;
    using super::batchSize;
    using super::xMultTheta;

    MSETaskSample(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
                  NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter, size_t blockSizeDefault);
    virtual ~MSETaskSample();

    virtual Status init(algorithmFPType *&pArgumentArray) DAAL_C11_OVERRIDE;
    virtual Status getCurrentBlock(size_t startIdx, size_t blockSize,
        algorithmFPType *&pBlockData, algorithmFPType *&pBlockDependentVariables) DAAL_C11_OVERRIDE;
    virtual void releaseCurrentBlock() DAAL_C11_OVERRIDE;

    NumericTable *ntIndices;
    BlockDescriptor<int> indicesBlock;
    int *indicesArray;

    TArray<algorithmFPType, cpu> dataBlockMemory;
    TArray<algorithmFPType, cpu> dependentVariablesBlockMemory;
};

template<typename algorithmFPType, Method method, CpuType cpu>
class MSEKernel : public Kernel
{
public:
    services::Status compute(NumericTable *data, NumericTable *dependentVariables, NumericTable *argument,
                          NumericTable *value, NumericTable *hessian, NumericTable *gradient, Parameter *parameter);
private:
    void computeMSE(
        size_t blockSize,
        MSETask<algorithmFPType, cpu>& task,
        algorithmFPType *data,
        algorithmFPType *argumentArray,
        algorithmFPType *dependentVariablesArray,
        algorithmFPType *value,
        algorithmFPType *gradient,
        algorithmFPType *hessian);

    void normalizeResults(
        MSETask<algorithmFPType, cpu> &task,
        algorithmFPType *value,
        algorithmFPType *gradient,
        algorithmFPType *hessian);

    Status run(MSETask<algorithmFPType, cpu>& task);
};

} // namespace daal::internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

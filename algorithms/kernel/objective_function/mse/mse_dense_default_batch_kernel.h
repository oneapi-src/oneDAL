/* file: mse_dense_default_batch_kernel.h */
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
#include "soa_numeric_table.h"

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

template <typename algorithmFPType, CpuType cpu>
struct MSETask
{
    DAAL_NEW_DELETE();
    MSETask(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument, NumericTable * value, NumericTable * hessian,
            NumericTable * gradient, Parameter * parameter);
    virtual ~MSETask();

    virtual Status init(algorithmFPType *& pArgumentArray);
    virtual Status getCurrentBlock(size_t startIdx, size_t blockSize, algorithmFPType *& pBlockData, algorithmFPType *& pBlockDependentVariables) = 0;
    virtual void releaseCurrentBlock()                                                                                                            = 0;

    void setResultValuesToZero(algorithmFPType * value, algorithmFPType * gradient, algorithmFPType * hessian);
    Status getResultValues(algorithmFPType *& value, algorithmFPType *& gradient, algorithmFPType *& hessian);
    void releaseResultValues();

    BlockDescriptor<algorithmFPType> dataBlock;
    BlockDescriptor<algorithmFPType> dependentVariablesBlock;
    BlockDescriptor<algorithmFPType> argumentBlock;
    BlockDescriptor<algorithmFPType> gradientBlock;
    BlockDescriptor<algorithmFPType> valueBlock;
    BlockDescriptor<algorithmFPType> hessianBlock;

    NumericTable * ntData;
    NumericTable * ntDependentVariables;
    NumericTable * ntArgument;
    NumericTable * ntGradient;
    NumericTable * ntValue;
    NumericTable * ntHessian;

    bool valueFlag;
    bool hessianFlag;
    bool gradientFlag;
    size_t argumentSize;
    size_t nTheta;
    size_t batchSize;
    TArray<algorithmFPType, cpu> xMultTheta;
};

template <typename algorithmFPType, CpuType cpu>
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

    MSETaskAll(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument, NumericTable * value, NumericTable * hessian,
               NumericTable * gradient, Parameter * parameter, size_t blockSizeDefault);
    virtual ~MSETaskAll();

    virtual Status init(algorithmFPType *& pArgumentArray) DAAL_C11_OVERRIDE;

    virtual Status getCurrentBlock(size_t startIdx, size_t blockSize, algorithmFPType *& pBlockData,
                                   algorithmFPType *& pBlockDependentVariables) DAAL_C11_OVERRIDE;
    virtual void releaseCurrentBlock() DAAL_C11_OVERRIDE;
};

template <typename algorithmFPType, CpuType cpu>
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

    MSETaskSample(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument, NumericTable * value, NumericTable * hessian,
                  NumericTable * gradient, Parameter * parameter, size_t blockSizeDefault);
    virtual ~MSETaskSample();

    virtual Status init(algorithmFPType *& pArgumentArray) DAAL_C11_OVERRIDE;
    virtual Status getCurrentBlock(size_t startIdx, size_t blockSize, algorithmFPType *& pBlockData,
                                   algorithmFPType *& pBlockDependentVariables) DAAL_C11_OVERRIDE;
    virtual void releaseCurrentBlock() DAAL_C11_OVERRIDE;

    NumericTable * ntIndices;
    BlockDescriptor<int> indicesBlock;
    int * indicesArray;

    TArray<algorithmFPType, cpu> dataBlockMemory;
    TArray<algorithmFPType, cpu> dependentVariablesBlockMemory;
};

template <typename algorithmFPType, Method method, CpuType cpu>
class MSEKernel : public Kernel
{
public:
    services::Status compute(NumericTable * data, NumericTable * dependentVariables, NumericTable * argument, NumericTable * value,
                             NumericTable * hessian, NumericTable * gradient, NumericTable * nonSmoothTermValue, NumericTable * proximalProjection,
                             NumericTable * lipschitzConstant, NumericTable * componentOfGradient, NumericTable * componentOfHessianDiagonal,
                             NumericTable * componentOfProximalProjection, Parameter * parameter);
    MSEKernel()
        : hessianDiagonal(0),
          hessianDiagonalPtr(nullptr),
          residual(0),
          residualPtr(nullptr),
          previousInputData(nullptr),
          previousFeatureId(-1),
          previousFeatureValuesPtr(nullptr),
          previousFeatureValues(0),
          computeViaGramMatrix(false),
          gramMatrix(0),
          gramMatrixPtr(nullptr),
          XY(0),
          XYPtr(nullptr),
          gradientForGram(0),
          gradientForGramPtr(nullptr),
          xNT(nullptr),
          X(nullptr),
          dot(0),
          dotPtr(nullptr),
          betaNT(nullptr),
          b(nullptr),
          gradNT(nullptr),
          gr(nullptr),
          hesDiagonalNT(nullptr),
          h(nullptr),
          penaltyL1NT(nullptr),
          penaltyL1Ptr(nullptr),
          proxNT(nullptr),
          proxPtr(nullptr),
          transposedData(false) {};

private:
    void computeMSE(size_t blockSize, MSETask<algorithmFPType, cpu> & task, algorithmFPType * data, algorithmFPType * argumentArray,
                    algorithmFPType * dependentVariablesArray, algorithmFPType * value, algorithmFPType * gradient, algorithmFPType * hessian);

    void normalizeResults(MSETask<algorithmFPType, cpu> & task, algorithmFPType * value, algorithmFPType * gradient, algorithmFPType * hessian);

    Status run(MSETask<algorithmFPType, cpu> & task);

    TArray<algorithmFPType, cpu> residual;
    TArray<algorithmFPType, cpu> gramMatrix;
    TArray<algorithmFPType, cpu> XY;

    TArray<algorithmFPType, cpu> hessianDiagonal;
    TArray<algorithmFPType, cpu> gradientForGram;
    TArray<algorithmFPType, cpu> previousFeatureValues;
    algorithmFPType * residualPtr;
    algorithmFPType * hessianDiagonalPtr;
    algorithmFPType * previousInputData;
    algorithmFPType * gramMatrixPtr;
    algorithmFPType * XYPtr;
    algorithmFPType * gradientForGramPtr;
    bool computeViaGramMatrix;

    int previousFeatureId;
    algorithmFPType * previousFeatureValuesPtr;

    NumericTable * xNT;
    algorithmFPType * X;
    TArray<algorithmFPType, cpu> dot;
    algorithmFPType * dotPtr;
    NumericTable * betaNT;
    algorithmFPType * b;

    NumericTable * gradNT;
    algorithmFPType * gr;

    NumericTable * hesDiagonalNT;
    algorithmFPType * h;

    NumericTable * penaltyL1NT;
    float * penaltyL1Ptr;
    NumericTable * proxNT;
    algorithmFPType * proxPtr;
    ReadRows<algorithmFPType, cpu> XPtr;
    bool transposedData;
};

} // namespace internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif

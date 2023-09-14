/* file: sgd_dense_minibatch_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of SGD dense minibatch Batch algorithm on GPU.
//--
*/

#include "src/algorithms/optimization_solver/sgd/oneapi/cl_kernel/sgd_dense_minibatch.cl"
#include "src/algorithms/optimization_solver/iterative_solver_kernel.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"
#include "src/externals/service_math.h"

#include "src/externals/service_profiler.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
using namespace daal::services::internal::sycl;
using namespace daal::data_management;

using daal::services::internal::Buffer;
using daal::data_management::internal::SyclHomogenNumericTable;

static uint32_t getWorkgroupsCount(const uint32_t n, const uint32_t localWorkSize)
{
    DAAL_ASSERT(localWorkSize > 0);
    const uint32_t elementsPerGroup = localWorkSize;
    uint32_t workgroupsCount        = n / elementsPerGroup;

    if (workgroupsCount * elementsPerGroup < n)
    {
        workgroupsCount++;
    }
    return workgroupsCount;
}

template <typename algorithmFPType>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch>::makeStep(const uint32_t argumentSize, const Buffer<algorithmFPType> & prevWorkValueBuff,
                                                                       const Buffer<algorithmFPType> & gradientBuff,
                                                                       Buffer<algorithmFPType> & workValueBuff, const algorithmFPType learningRate,
                                                                       const algorithmFPType consCoeff)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(makeStep);
    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "makeStep";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_ASSERT(gradientBuff.size() == argumentSize);
    DAAL_ASSERT(prevWorkValueBuff.size() == argumentSize);
    DAAL_ASSERT(workValueBuff.size() == argumentSize);

    KernelArguments args(5, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, gradientBuff, AccessModeIds::read);
    args.set(1, prevWorkValueBuff, AccessModeIds::read);
    args.set(2, workValueBuff, AccessModeIds::readwrite);
    args.set(3, learningRate);
    args.set(4, consCoeff);

    KernelRange range(argumentSize);
    ctx.run(range, kernel, args, status);

    return status;
}

template <typename algorithmFPType>
static services::Status sumReduction(const Buffer<algorithmFPType> & reductionBuffer, const size_t nWorkGroups, algorithmFPType & result)
{
    services::Status status;

    DAAL_CHECK(reductionBuffer.size() == nWorkGroups, services::ErrorIncorrectSizeOfArray);

    auto sumReductionArrayPtr = reductionBuffer.toHost(data_management::readOnly, status);
    DAAL_CHECK_STATUS_VAR(status);

    const auto * sumReductionArray = sumReductionArrayPtr.get();

    // Final summation with CPU
    for (size_t i = 0; i < nWorkGroups; i++)
    {
        result += sumReductionArray[i];
    }
    return status;
}

template <typename algorithmFPType>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch>::vectorNorm(const Buffer<algorithmFPType> & x, const uint32_t n, algorithmFPType & norm)
{
    services::Status status;

    const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    status |= buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "sumSq";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    size_t workItemsPerGroup = 256;
    const size_t nWorkGroups = getWorkgroupsCount(n, workItemsPerGroup);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, workItemsPerGroup, nWorkGroups);

    KernelRange localRange(workItemsPerGroup);
    KernelRange globalRange(workItemsPerGroup * nWorkGroups);

    KernelNDRange range(1);

    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    UniversalBuffer buffer = ctx.allocate(idType, nWorkGroups, status);
    DAAL_CHECK_STATUS_VAR(status);
    Buffer<algorithmFPType> reductionBuffer = buffer.get<algorithmFPType>();

    DAAL_ASSERT(x.size() == n);

    KernelArguments args(3, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, x, AccessModeIds::read);
    args.set(1, n);
    args.set(2, reductionBuffer, AccessModeIds::write);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(vectorNorm.run);
        ctx.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    status = sumReduction<algorithmFPType>(reductionBuffer, nWorkGroups, norm);
    DAAL_CHECK_STATUS_VAR(status);

    norm = daal::internal::MathInst<algorithmFPType, DAAL_BASE_CPU>::sSqrt(norm);

    return status;
}

template <typename algorithmFPType>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch>::buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);
    services::Status status;
    services::String options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_optimization_solver_sgd_");
    cachekey.add(options);
    options.add(" -D LOCAL_SUM_SIZE=256 ");

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSGDMiniBatch, options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch>::compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTablePtr minimum,
                                                                      NumericTable * nIterations, Parameter<miniBatch> * parameter,
                                                                      NumericTable * learningRateSequence, NumericTable * batchIndices,
                                                                      OptionalArgument * optionalArgument, OptionalArgument * optionalResult,
                                                                      engines::BatchBase & engine)
{
    services::Status status;

    ExecutionContextIface & ctx = services::internal::getDefaultContext();

    DAAL_ASSERT(inputArgument != nullptr);
    DAAL_ASSERT(parameter != nullptr);

    const size_t argumentSize = inputArgument->getNumberOfRows();
    const size_t nIter        = parameter->nIterations;
    const size_t L            = parameter->innerNIterations;
    const size_t batchSize    = parameter->batchSize;

    constexpr size_t maxInt32Value = static_cast<size_t>(daal::services::internal::MaxVal<int32_t>::get());

    WriteRows<int, DAAL_BASE_CPU> nIterationsBD(*nIterations, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(nIterationsBD);
    int * nProceededIterations = nIterationsBD.get();
    DAAL_CHECK(nProceededIterations != nullptr, services::ErrorIncorrectInputNumericTable);

    // if nIter == 0, set result as start point, the number of executed iters to 0
    if (nIter == 0 || L == 0)
    {
        nProceededIterations[0] = 0;
        return status;
    }

    NumericTable * lastIterationInput = optionalArgument ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable * pastWorkValueInput = optionalArgument ? NumericTable::cast(optionalArgument->get(sgd::pastWorkValue)).get() : nullptr;

    NumericTable * lastIterationResult = optionalResult ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable * pastWorkValueResult = optionalResult ? NumericTable::cast(optionalResult->get(sgd::pastWorkValue)).get() : nullptr;

    const double accuracyThreshold = parameter->accuracyThreshold;

    sum_of_functions::BatchPtr function = parameter->function;
    const size_t nTerms                 = function->sumOfFunctionsParameter->numberOfTerms;

    DAAL_ASSERT(minimum == true);
    DAAL_ASSERT(minimum->getNumberOfRows() == argumentSize);

    BlockDescriptor<algorithmFPType> workValueBD;
    DAAL_CHECK_STATUS(status, minimum->getBlockOfRows(0, argumentSize, ReadWriteMode::readWrite, workValueBD));
    Buffer<algorithmFPType> workValueBuff = workValueBD.getBuffer();

    auto workValueSNT = SyclHomogenNumericTable<algorithmFPType>::create(workValueBuff, 1, argumentSize, &status);
    DAAL_CHECK_STATUS_VAR(status);

    NumericTablePtr previousArgument = function->sumOfFunctionsInput->get(sum_of_functions::argument);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, workValueSNT);

    ReadRows<algorithmFPType, DAAL_BASE_CPU> learningRateBD(*learningRateSequence, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(learningRateBD);
    const algorithmFPType * const learningRateArray = learningRateBD.get();
    DAAL_CHECK(learningRateArray != nullptr, services::ErrorIncorrectParameter);

    NumericTable * conservativeSequence = parameter->conservativeSequence.get();
    ReadRows<algorithmFPType, DAAL_BASE_CPU> consCoeffsBD(*conservativeSequence, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(consCoeffsBD);
    const algorithmFPType * const consCoeffsArray = consCoeffsBD.get();
    DAAL_CHECK(consCoeffsArray != nullptr, services::ErrorIncorrectParameter);

    const size_t consCoeffsLength   = conservativeSequence->getNumberOfColumns();
    const size_t learningRateLength = learningRateSequence->getNumberOfColumns();

    const IndicesStatus indicesStatus = (batchIndices ? user : (batchSize < nTerms ? random : all));
    services::SharedPtr<HomogenNumericTableCPU<int, DAAL_BASE_CPU> > ntBatchIndices;

    if (indicesStatus == user || indicesStatus == random)
    {
        // Replace by SyclNumericTable when will be RNG on GPU
        ntBatchIndices = HomogenNumericTableCPU<int, DAAL_BASE_CPU>::create(batchSize, 1, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    NumericTablePtr previousBatchIndices            = function->sumOfFunctionsParameter->batchIndices;
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;

    const TypeIds::Id idType       = TypeIds::id<algorithmFPType>();
    UniversalBuffer prevWorkValueU = ctx.allocate(idType, argumentSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    Buffer<algorithmFPType> prevWorkValueBuff = prevWorkValueU.get<algorithmFPType>();

    size_t startIteration = 0, nProceededIters = 0;
    if (lastIterationInput)
    {
        ReadRows<int, DAAL_BASE_CPU> lastIterationInputBD(lastIterationInput, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(lastIterationInputBD);
        const int * lastIterationInputArray = lastIterationInputBD.get();
        DAAL_ASSERT(lastIterationInputArray[0] > 0);
        startIteration = lastIterationInputArray[0];
    }

    if (pastWorkValueInput)
    {
        BlockDescriptor<algorithmFPType> pastWorkValueInputBD;
        DAAL_CHECK_STATUS(status, pastWorkValueInput->getBlockOfRows(0, argumentSize, ReadWriteMode::readOnly, pastWorkValueInputBD));

        const Buffer<algorithmFPType> pastWorkValueInputBuff = pastWorkValueInputBD.getBuffer();

        ctx.copy(prevWorkValueBuff, 0, pastWorkValueInputBuff, 0, argumentSize, status);
        DAAL_CHECK_STATUS(status, pastWorkValueInput->releaseBlockOfRows(pastWorkValueInputBD));
    }
    else
    {
        ctx.fill(prevWorkValueU, 0.0, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    // init workValue
    BlockDescriptor<algorithmFPType> startValueBD;
    DAAL_CHECK_STATUS(status, inputArgument->getBlockOfRows(0, argumentSize, ReadWriteMode::readOnly, startValueBD));
    const Buffer<algorithmFPType> startValueBuff = startValueBD.getBuffer();
    ctx.copy(workValueBuff, 0, startValueBuff, 0, argumentSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    DAAL_CHECK_STATUS(status, inputArgument->releaseBlockOfRows(startValueBD));

    ReadRows<int, DAAL_BASE_CPU> predefinedBatchIndicesBD(batchIndices, 0, nIter);
    DAAL_CHECK_BLOCK_STATUS(predefinedBatchIndicesBD);
    iterative_solver::internal::RngTask<int, DAAL_BASE_CPU> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    rngTask.init(nTerms, engine);

    algorithmFPType learningRate = learningRateArray[0];
    algorithmFPType consCoeff    = consCoeffsArray[0];

    UniversalBuffer gradientU = ctx.allocate(idType, argumentSize, status);
    DAAL_CHECK_STATUS_VAR(status);
    Buffer<algorithmFPType> gradientBuff = gradientU.get<algorithmFPType>();

    auto gradientSNT = SyclHomogenNumericTable<algorithmFPType>::create(gradientBuff, 1, argumentSize, &status);
    DAAL_CHECK_STATUS_VAR(status);
    function->getResult()->set(objective_function::gradientIdx, gradientSNT);

    DAAL_CHECK(nIter <= maxInt32Value, services::ErrorIncorrectParameter);
    *nProceededIterations = static_cast<int>(nIter);

    services::internal::HostAppHelper host(pHost, 10);
    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, startIteration, nIter);
    for (size_t epoch = startIteration; epoch < (startIteration + nIter); epoch++)
    {
        if (epoch % L == 0 || epoch == startIteration)
        {
            learningRate = learningRateArray[(epoch / L) % learningRateLength];
            consCoeff    = consCoeffsArray[(epoch / L) % consCoeffsLength];
            if (indicesStatus == user || indicesStatus == random)
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(generateUniform);
                const int * pValues = nullptr;
                DAAL_CHECK_STATUS(status, rngTask.get(pValues));
                DAAL_CHECK_STATUS(status, ntBatchIndices->setArray(const_cast<int *>(pValues), ntBatchIndices->getNumberOfRows()));
            }
        }

        DAAL_CHECK_STATUS(status, function->computeNoThrow());

        if (host.isCancelled(status, 1))
        {
            // overflow is checked on casting nIter to int
            // epoch - startIteration is always less then nIter
            *nProceededIterations = static_cast<int>(epoch - startIteration);
            break;
        }

        if (epoch % L == 0)
        {
            if (nIter > 1 && accuracyThreshold > 0)
            {
                algorithmFPType pointNorm = algorithmFPType(0), gradientNorm = algorithmFPType(0);
                DAAL_CHECK_STATUS(status, vectorNorm(workValueBuff, argumentSize, pointNorm));
                DAAL_CHECK_STATUS(status, vectorNorm(gradientBuff, argumentSize, gradientNorm));
                const double gradientThreshold = accuracyThreshold * daal::internal::MathInst<algorithmFPType, DAAL_BASE_CPU>::sMax(1.0, pointNorm);

                if (gradientNorm < gradientThreshold)
                {
                    // overflow is checked on casting nIter to int
                    // epoch - startIteration is always less then nIter
                    *nProceededIterations = static_cast<int>(epoch - startIteration);
                    break;
                }
            }

            ctx.copy(prevWorkValueBuff, 0, workValueBuff, 0, argumentSize, status);
            DAAL_CHECK_STATUS_VAR(status);
        }
        DAAL_CHECK_STATUS(status, makeStep(argumentSize, prevWorkValueBuff, gradientBuff, workValueBuff, learningRate, consCoeff));
        nProceededIters++;
    }

    if (lastIterationResult)
    {
        WriteRows<int, DAAL_BASE_CPU> lastIterationResultBD(lastIterationResult, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(lastIterationResultBD);
        int * lastIterationResultArray = lastIterationResultBD.get();
        lastIterationResultArray[0]    = startIteration + nProceededIters; // overflow is already checked for (startIteration + nIter)
    }

    if (pastWorkValueResult)
    {
        BlockDescriptor<algorithmFPType> pastWorkValueResultBD;
        DAAL_CHECK_STATUS(status, pastWorkValueResult->getBlockOfRows(0, argumentSize, ReadWriteMode::writeOnly, pastWorkValueResultBD));

        Buffer<algorithmFPType> pastWorkValueResultBuffer = pastWorkValueResultBD.getBuffer();

        ctx.copy(pastWorkValueResultBuffer, 0, prevWorkValueBuff, 0, argumentSize, status);
        DAAL_CHECK_STATUS(status, pastWorkValueResult->releaseBlockOfRows(pastWorkValueResultBD));
    }

    DAAL_CHECK_STATUS(status, minimum->releaseBlockOfRows(workValueBD));

    function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);
    return status;
}

} // namespace internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

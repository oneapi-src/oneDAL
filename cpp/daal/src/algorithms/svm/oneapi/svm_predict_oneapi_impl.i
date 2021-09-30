/* file: svm_predict_oneapi_impl.i */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
//  SVM prediction algorithm implementation
//--
*/

#ifndef __SVM_PREDICT_ONEAPI_IMPL_I__
#define __SVM_PREDICT_ONEAPI_IMPL_I__

#include "src/sycl/blas_gpu.h"
#include "src/externals/service_profiler.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"
#include "data_management/data/internal/numeric_table_sycl_csr.h"
#include "src/algorithms/svm/oneapi/svm_helper_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services::internal::sycl;
using daal::data_management::internal::SyclHomogenNumericTable;

template <typename algorithmFPType>
class PredictTask : public Base
{
public:
    virtual ~PredictTask() {}

    services::Status kernelCompute(const size_t startRow, const size_t nRows)
    {
        services::Status status;
        auto subbuff = _buff.get<algorithmFPType>().getSubBuffer(0, nRows * _nSV, status);
        DAAL_CHECK_STATUS_VAR(status);
        auto shResNT = SyclHomogenNumericTable<algorithmFPType>::create(subbuff, _nSV, nRows, &status);
        DAAL_CHECK_STATUS_VAR(status);

        auto xBlockNT = getBlockNTData(startRow, nRows, status);
        DAAL_CHECK_STATUS_VAR(status);

        _shRes->set(kernel_function::values, shResNT);
        _kernel->getInput()->set(kernel_function::X, xBlockNT);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        DAAL_CHECK(_kernel->computeNoThrow(), services::ErrorSVMPredictKernerFunctionCall);

        return status;
    }

    services::internal::Buffer<algorithmFPType> getBuff() const { return _buff.get<algorithmFPType>(); }

protected:
    PredictTask(const size_t nMaxRowsPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                const kernel_function::KernelIfacePtr & kernel, services::Status & status)
        : _xTable(xTable), _nSV(svTable->getNumberOfRows())
    {
        auto & context = services::internal::getDefaultContext();
        _buff          = context.allocate(TypeIds::id<algorithmFPType>(), _nSV * nMaxRowsPerBlock, status);

        if (!status)
        {
            return;
        }

        _kernel = kernel->clone();
        _shRes  = kernel_function::ResultPtr(new kernel_function::Result());
        _kernel->setResult(_shRes);
        _kernel->getInput()->set(kernel_function::Y, svTable);
    }

    virtual NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) = 0;

protected:
    const NumericTablePtr & _xTable;
    const size_t _nSV;
    UniversalBuffer _buff;
    kernel_function::KernelIfacePtr _kernel;
    kernel_function::ResultPtr _shRes;
};

template <typename algorithmFPType>
class PredictTaskDense : public PredictTask<algorithmFPType>
{
public:
    using Super = PredictTask<algorithmFPType>;

    virtual ~PredictTaskDense() { Super::_xTable->releaseBlockOfRows(_xBlock); }

    static services::SharedPtr<PredictTaskDense<algorithmFPType> > create(const size_t nRowsPerBlock, const NumericTablePtr & xTable,
                                                                          const NumericTablePtr & svTable,
                                                                          const kernel_function::KernelIfacePtr & kernel,
                                                                          services::Status * stat = nullptr)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(PredictTaskDense, algorithmFPType, nRowsPerBlock, xTable, svTable, kernel);
    }

protected:
    PredictTaskDense(const size_t nRowsPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                     const kernel_function::KernelIfacePtr & kernel, services::Status & status)
        : Super(nRowsPerBlock, xTable, svTable, kernel, status)
    {}

    NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) override
    {
        Super::_xTable->releaseBlockOfRows(_xBlock);
        status |= Super::_xTable->getBlockOfRows(startRow, nRows, ReadWriteMode::readOnly, _xBlock);
        const services::internal::Buffer<algorithmFPType> xBuf = _xBlock.getBuffer();

        NumericTablePtr xBlockNT = SyclHomogenNumericTable<algorithmFPType>::create(xBuf, Super::_xTable->getNumberOfColumns(), nRows, &status);
        if (!xBlockNT)
        {
            status |= services::Status(services::ErrorMemoryAllocationFailed);
        }
        return xBlockNT;
    }

private:
    BlockDescriptor<algorithmFPType> _xBlock;
};

template <typename algorithmFPType>
class PredictTaskCSR : public PredictTask<algorithmFPType>
{
public:
    using Super = PredictTask<algorithmFPType>;

    virtual ~PredictTaskCSR() {}

    static services::SharedPtr<PredictTaskCSR<algorithmFPType> > create(const size_t nRowsPerBlock, const NumericTablePtr & xTable,
                                                                        const NumericTablePtr & svTable,
                                                                        const kernel_function::KernelIfacePtr & kernel,
                                                                        services::Status * stat = nullptr)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(PredictTaskCSR, algorithmFPType, nRowsPerBlock, xTable, svTable, kernel);
    }

protected:
    PredictTaskCSR(const size_t nRowsPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                   const kernel_function::KernelIfacePtr & kernel, services::Status & status)
        : Super(nRowsPerBlock, xTable, svTable, kernel, status)
    {}

    NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) override
    {
        auto csrIface = services::dynamicPointerCast<CSRNumericTableIface>(Super::_xTable);
        status |= csrIface->releaseSparseBlock(_xBlock);
        status |= csrIface->getSparseBlock(startRow, nRows, readOnly, _xBlock);

        auto xValuesBuff     = _xBlock.getBlockValuesBuffer();
        auto xColIndicesBuff = _xBlock.getBlockColumnIndicesBuffer();
        auto xRowOffsetsBuff = _xBlock.getBlockRowIndicesBuffer();

        NumericTablePtr xBlockNT = SyclCSRNumericTable::create(xValuesBuff, xColIndicesBuff, xRowOffsetsBuff, Super::_xTable->getNumberOfColumns(),
                                                               nRows, CSRNumericTableIface::oneBased, &status);
        if (!xBlockNT)
        {
            status |= services::Status(services::ErrorMemoryAllocationFailed);
        }
        return xBlockNT;
    }

private:
    CSRBlockDescriptor<algorithmFPType> _xBlock;
};

template <typename algorithmFPType>
services::Status SVMPredictImplOneAPI<defaultDense, algorithmFPType>::compute(const NumericTablePtr & xTable, Model * model, NumericTable & result,
                                                                              const svm::Parameter * par)
{
    services::Status status;
    auto & context = services::internal::getDefaultContext();

    const size_t nVectors  = xTable->getNumberOfRows();
    const size_t nFeatures = xTable->getNumberOfColumns();

    DAAL_ASSERT(result.getNumberOfRows() == nVectors)
    DAAL_ASSERT(result.getNumberOfColumns() == 1)

    BlockDescriptor<algorithmFPType> resultBlock;
    DAAL_CHECK_STATUS(status, result.getBlockOfRows(0, nVectors, ReadWriteMode::writeOnly, resultBlock));
    auto distanceBuff = resultBlock.getBuffer();

    auto svCoeffTable = model->getClassificationCoefficients();
    const size_t nSV  = svCoeffTable->getNumberOfRows();

    if (nSV == 0)
    {
        context.fill(distanceBuff, 0.0, status);
        return status;
    }

    BlockDescriptor<algorithmFPType> svCoeffBlock;
    DAAL_CHECK_STATUS(status, svCoeffTable->getBlockOfRows(0, nSV, ReadWriteMode::readOnly, svCoeffBlock));
    auto svCoeffBuff = svCoeffBlock.getBuffer();

    const algorithmFPType bias(model->getBias());
    context.fill(distanceBuff, double(bias), status);
    DAAL_CHECK_STATUS_VAR(status);

    auto svTable = model->getSupportVectors();

    const size_t nRowsPerBlock = xTable->getDataLayout() == NumericTableIface::csrArray ? nVectors : 1024;
    const size_t nBlocks       = nVectors / nRowsPerBlock + !!(nVectors % nRowsPerBlock);

    kernel_function::ResultPtr shRes(new kernel_function::Result());
    DAAL_CHECK_MALLOC(shRes)

    services::SharedPtr<PredictTask<algorithmFPType> > predictTask;
    if (xTable->getDataLayout() == NumericTableIface::csrArray)
    {
        predictTask = PredictTaskCSR<algorithmFPType>::create(nRowsPerBlock, xTable, svTable, par->kernel);
    }
    else
    {
        predictTask = PredictTaskDense<algorithmFPType>::create(nRowsPerBlock, xTable, svTable, par->kernel);
    }

    for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
    {
        const size_t startRow          = iBlock * nRowsPerBlock;
        const size_t nRowsPerBlockReal = (iBlock != nBlocks - 1) ? nRowsPerBlock : nVectors - iBlock * nRowsPerBlock;

        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS(predictTask->kernelCompute(startRow, nRowsPerBlockReal), services::ErrorSVMPredictKernerFunctionCall);
        const auto kernelResBuff = predictTask->getBuff();

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(gemm);
            DAAL_CHECK_STATUS(status, BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::NoTrans,
                                                                      nRowsPerBlockReal, 1, nSV, algorithmFPType(1.0), kernelResBuff, nSV, 0,
                                                                      svCoeffBuff, 1, 0, algorithmFPType(1.0), distanceBuff, 1, startRow));
        }
    }
    DAAL_CHECK_STATUS(status, result.releaseBlockOfRows(resultBlock));
    DAAL_CHECK_STATUS(status, svCoeffTable->releaseBlockOfRows(svCoeffBlock));

    return status;
}

} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

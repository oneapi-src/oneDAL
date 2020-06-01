/* file: svm_predict_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifndef __SVM_PREDICT_IMPL_I__
#define __SVM_PREDICT_IMPL_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "externals/service_blas.h"
#include "externals/service_memory.h"

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
using namespace daal::services;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
class PredictTask
{
public:
    DAAL_NEW_DELETE();
    virtual ~PredictTask() {}

    bool isValid() const { return _buff.get(); }

    services::Status kernelCompute(const size_t startRow, const size_t nRows)
    {
        services::Status status;
        NumericTablePtr shResNT = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_buff.get(), _nSV, nRows, &status);
        DAAL_CHECK_STATUS_VAR(status);

        auto xBlockNT = getBlockNTData(startRow, nRows, status);
        DAAL_CHECK_STATUS_VAR(status);
        _shRes->set(kernel_function::values, shResNT);
        _kernel->getInput()->set(kernel_function::X, xBlockNT);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;

        return _kernel->computeNoThrow();
    }

    const algorithmFPType * getBuff() const { return _buff.get(); }

protected:
    PredictTask(const size_t nRowsPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable, kernel_function::KernelIfacePtr & kernel)
        : _xTable(xTable), _nSV(svTable->getNumberOfRows()), _nFeatures(svTable->getNumberOfColumns())
    {
        _buff.reset(_nSV * nRowsPerBlock);
        _kernel = kernel->clone();
        _shRes  = kernel_function::ResultPtr(new kernel_function::Result());
        _kernel->setResult(_shRes);
        _kernel->getInput()->set(kernel_function::Y, svTable);
    }

    virtual NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) = 0;

protected:
    const NumericTablePtr & _xTable;
    const size_t _nSV;
    const size_t _nFeatures;
    TArrayScalable<algorithmFPType, cpu> _buff;
    kernel_function::KernelIfacePtr _kernel;
    kernel_function::ResultPtr _shRes;
};

template <typename algorithmFPType, CpuType cpu>
class PredictTaskDense : PredictTask<algorithmFPType, cpu>
{
public:
    using Super = PredictTask<algorithmFPType, cpu>;
    DAAL_NEW_DELETE();
    virtual ~PredictTaskDense() {}

    static Super * create(const size_t nRowsPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                          kernel_function::KernelIfacePtr & kernel)
    {
        auto val = new PredictTaskDense(nRowsPerBlock, xTable, svTable, kernel);
        if (val && val->isValid()) return val;
        delete val;
        return nullptr;
    }

protected:
    PredictTaskDense(const size_t nRowsPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                     kernel_function::KernelIfacePtr & kernel)
        : Super(nRowsPerBlock, xTable, svTable, kernel)
    {}

    NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) override
    {
        algorithmFPType * xData = const_cast<algorithmFPType *>(_xBlock.set(*Super::_xTable, startRow, nRows));
        if (!xData) status |= services::Status(services::ErrorMemoryAllocationFailed);
        return HomogenNumericTableCPU<algorithmFPType, cpu>::create(xData, Super::_nFeatures, nRows, &status);
    }

private:
    ReadRows<algorithmFPType, cpu> _xBlock;
};

template <typename algorithmFPType, CpuType cpu>
class PredictTaskCSR : PredictTask<algorithmFPType, cpu>
{
public:
    using Super = PredictTask<algorithmFPType, cpu>;
    DAAL_NEW_DELETE();
    virtual ~PredictTaskCSR() {}

    static Super * create(const size_t nRowsPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                          kernel_function::KernelIfacePtr & kernel)
    {
        auto val = new PredictTaskCSR(nRowsPerBlock, xTable, svTable, kernel);
        if (val && val->isValid()) return val;
        delete val;
        return nullptr;
    }

protected:
    PredictTaskCSR(const size_t nRowsPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                   kernel_function::KernelIfacePtr & kernel)
        : Super(nRowsPerBlock, xTable, svTable, kernel), _rowOffsets(nRowsPerBlock + 1)
    {}

    NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) override
    {
        _xBlock.set(dynamic_cast<CSRNumericTableIface *>(Super::_xTable.get()), startRow, nRows);
        algorithmFPType * values = const_cast<algorithmFPType *>(_xBlock.values());
        size_t * cols            = const_cast<size_t *>(_xBlock.cols());
        const size_t * rows      = _xBlock.rows();
        if (!values || !cols || !rows) status |= services::Status(services::ErrorMemoryAllocationFailed);
        _rowOffsets[0] = 1;
        for (size_t i = 0; i < nRows; i++)
        {
            const size_t nNonZeroValuesInRow = rows[i + 1] - rows[i];
            _rowOffsets[i + 1]               = _rowOffsets[i] + nNonZeroValuesInRow;
        }

        return CSRNumericTable::create(values, cols, _rowOffsets.get(), Super::_nFeatures, nRows, CSRNumericTableIface::CSRIndexing::oneBased,
                                       &status);
    }

private:
    TArrayScalable<size_t, cpu> _rowOffsets;
    ReadRowsCSR<algorithmFPType, cpu> _xBlock;
};

template <typename algorithmFPType, CpuType cpu>
struct SVMPredictImpl<defaultDense, algorithmFPType, cpu> : public Kernel
{
    services::Status compute(const NumericTablePtr & xTable, Model * model, NumericTable & r, const svm::Parameter * par)
    {
        const size_t nVectors = xTable->getNumberOfRows();
        WriteOnlyColumns<algorithmFPType, cpu> mtR(r, 0, 0, nVectors);
        DAAL_CHECK_BLOCK_STATUS(mtR);
        algorithmFPType * distance = mtR.get();

        kernel_function::KernelIfacePtr kernel = par->kernel->clone();
        DAAL_CHECK(kernel, ErrorNullParameterNotSupported);

        NumericTablePtr svCoeffTable = model->getClassificationCoefficients();
        const algorithmFPType bias(model->getBias());

        const size_t nSV = svCoeffTable->getNumberOfRows();
        if (nSV == 0)
        {
            const algorithmFPType zero(0.0);
            service_memset<algorithmFPType, cpu>(distance, zero, nVectors);
            return Status();
        }
        else
        {
            service_memset<algorithmFPType, cpu>(distance, bias, nVectors);
        }
        const NumericTablePtr svTable = model->getSupportVectors();

        ReadColumns<algorithmFPType, cpu> mtSVCoeff(*svCoeffTable, 0, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSVCoeff);
        const algorithmFPType * svCoeff = mtSVCoeff.get();

        size_t nRowsPerBlock = 1;
        DAAL_SAFE_CPU_CALL((nRowsPerBlock = 256), (nRowsPerBlock = nVectors));

        const size_t nBlocks = nVectors / nRowsPerBlock + !!(nVectors % nRowsPerBlock);

        /* TLS data initialization */
        daal::tls<PredictTask<algorithmFPType, cpu> *> tlsTask([&]() {
            if (xTable->getDataLayout() == NumericTableIface::csrArray)
            {
                return PredictTaskCSR<algorithmFPType, cpu>::create(nRowsPerBlock, xTable, svTable, kernel);
            }
            else
            {
                return PredictTaskDense<algorithmFPType, cpu>::create(nRowsPerBlock, xTable, svTable, kernel);
            }
        });

        SafeStatus safeStat;

        daal::threader_for(nBlocks, nBlocks, [&](int iBlock) {
            PredictTask<algorithmFPType, cpu> * tlsLocal = tlsTask.local();

            const size_t startRow          = iBlock * nRowsPerBlock;
            const size_t endRow            = services::internal::min<cpu, size_t>(startRow + nRowsPerBlock, nVectors);
            const size_t nRowsPerBlockReal = endRow - startRow;

            DAAL_CHECK_THR(tlsLocal->kernelCompute(startRow, nRowsPerBlockReal), services::ErrorSVMPredictKernerFunctionCall);

            const algorithmFPType * buf = tlsLocal->getBuff();

            char trans  = 'T';
            DAAL_INT m_ = nSV;
            DAAL_INT n_ = nRowsPerBlockReal;
            algorithmFPType alpha(1.0);
            DAAL_INT lda = m_;
            DAAL_INT incx(1);
            algorithmFPType beta(1.0);
            DAAL_INT incy(1);

            if (nBlocks == 1)
            {
                Blas<algorithmFPType, cpu>::xgemv(&trans, &m_, &n_, &alpha, buf, &lda, svCoeff, &incx, &beta, distance + startRow, &incy);
            }
            else
            {
                Blas<algorithmFPType, cpu>::xxgemv(&trans, &m_, &n_, &alpha, buf, &lda, svCoeff, &incx, &beta, distance + startRow, &incy);
            }
        }); /* daal::threader_for */

        tlsTask.reduce([&](PredictTask<algorithmFPType, cpu> * local) { delete local; });
        return safeStat.detach();
    }
};

} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

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

#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_memory.h"
#include "src/services/service_environment.h"

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

    services::Status kernelCompute(const size_t startRow, const size_t nRows, const size_t startSV, const size_t nSV)
    {
        services::Status status;
        NumericTablePtr shResNT = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_buff.get(), nSV, nRows, &status);
        DAAL_CHECK_STATUS_VAR(status);

        auto xBlockNT = getBlockNTData(startRow, nRows, status);
        DAAL_CHECK_STATUS_VAR(status);

        auto svBlockNT = getBlockNTSV(startSV, nSV, status);
        DAAL_CHECK_STATUS_VAR(status);

        _shRes->set(kernel_function::values, shResNT);
        _kernel->getInput()->set(kernel_function::X, xBlockNT);
        _kernel->getInput()->set(kernel_function::Y, svBlockNT);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;

        return _kernel->computeNoThrow();
    }

    const algorithmFPType * getBuff() const { return _buff.get(); }

protected:
    PredictTask(const size_t nRowsPerBlock, const size_t nSVPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                kernel_function::KernelIfacePtr & kernel)
        : _xTable(xTable), _svTable(svTable), _nFeatures(svTable->getNumberOfColumns())
    {
        _buff.reset(nSVPerBlock * nRowsPerBlock);
        _kernel = kernel->clone();
        _shRes  = kernel_function::ResultPtr(new kernel_function::Result());
        _kernel->setResult(_shRes);
    }

    virtual NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) = 0;
    virtual NumericTablePtr getBlockNTSV(const size_t startSV, const size_t nSV, services::Status & status)      = 0;

protected:
    const NumericTablePtr & _xTable;
    const NumericTablePtr & _svTable;
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
    virtual ~PredictTaskDense() {}

    static Super * create(const size_t nRowsPerBlock, const size_t nSVPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                          kernel_function::KernelIfacePtr & kernel)
    {
        auto val = new PredictTaskDense(nRowsPerBlock, nSVPerBlock, xTable, svTable, kernel);
        if (val && val->isValid()) return val;
        delete val;
        return nullptr;
    }

protected:
    PredictTaskDense(const size_t nRowsPerBlock, const size_t nSVPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                     kernel_function::KernelIfacePtr & kernel)
        : Super(nRowsPerBlock, nSVPerBlock, xTable, svTable, kernel)
    {}

    NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) override
    {
        algorithmFPType * xData = const_cast<algorithmFPType *>(_xBlock.set(*Super::_xTable, startRow, nRows));
        if (!xData) status |= services::Status(services::ErrorMemoryAllocationFailed);
        return HomogenNumericTableCPU<algorithmFPType, cpu>::create(xData, Super::_nFeatures, nRows, &status);
    }

    NumericTablePtr getBlockNTSV(const size_t startSV, const size_t nSV, services::Status & status) override
    {
        algorithmFPType * svData = const_cast<algorithmFPType *>(_svBlock.set(*Super::_svTable, startSV, nSV));
        if (!svData) status |= services::Status(services::ErrorMemoryAllocationFailed);
        return HomogenNumericTableCPU<algorithmFPType, cpu>::create(svData, Super::_nFeatures, nSV, &status);
    }

private:
    ReadRows<algorithmFPType, cpu> _xBlock;
    ReadRows<algorithmFPType, cpu> _svBlock;
};

template <typename algorithmFPType, CpuType cpu>
class PredictTaskCSR : PredictTask<algorithmFPType, cpu>
{
public:
    using Super = PredictTask<algorithmFPType, cpu>;
    virtual ~PredictTaskCSR() {}

    static Super * create(const size_t nRowsPerBlock, const size_t nSVPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                          kernel_function::KernelIfacePtr & kernel)
    {
        auto val = new PredictTaskCSR(nRowsPerBlock, nSVPerBlock, xTable, svTable, kernel);
        if (val && val->isValid()) return val;
        delete val;
        return nullptr;
    }

protected:
    PredictTaskCSR(const size_t nRowsPerBlock, const size_t nSVPerBlock, const NumericTablePtr & xTable, const NumericTablePtr & svTable,
                   kernel_function::KernelIfacePtr & kernel)
        : Super(nRowsPerBlock, nSVPerBlock, xTable, svTable, kernel)
    {}

    NumericTablePtr getBlockNTData(const size_t startRow, const size_t nRows, services::Status & status) override
    {
        const bool toOneBaseRowIndices = true;
        _xBlock.set(dynamic_cast<CSRNumericTableIface *>(Super::_xTable.get()), startRow, nRows, toOneBaseRowIndices);
        algorithmFPType * const values = const_cast<algorithmFPType *>(_xBlock.values());
        size_t * const cols            = const_cast<size_t *>(_xBlock.cols());
        size_t * const rows            = const_cast<size_t *>(_xBlock.rows());

        return CSRNumericTable::create(values, cols, rows, Super::_nFeatures, nRows, CSRNumericTableIface::CSRIndexing::oneBased, &status);
    }

    NumericTablePtr getBlockNTSV(const size_t startSV, const size_t nSV, services::Status & status) override
    {
        const bool toOneBaseRowIndices = true;
        _svBlock.set(dynamic_cast<CSRNumericTableIface *>(Super::_svTable.get()), startSV, nSV, toOneBaseRowIndices);
        algorithmFPType * const values = const_cast<algorithmFPType *>(_svBlock.values());
        size_t * const cols            = const_cast<size_t *>(_svBlock.cols());
        size_t * const rows            = const_cast<size_t *>(_svBlock.rows());

        return CSRNumericTable::create(values, cols, rows, Super::_nFeatures, nSV, CSRNumericTableIface::CSRIndexing::oneBased, &status);
    }

private:
    ReadRowsCSR<algorithmFPType, cpu> _xBlock;
    ReadRowsCSR<algorithmFPType, cpu> _svBlock;
};

template <typename algorithmFPType, CpuType cpu>
struct SVMPredictImpl<defaultDense, algorithmFPType, cpu> : public Kernel
{
    services::Status compute(const NumericTablePtr & xTable, Model * model, NumericTable & r, const svm::Parameter * par)
    {
        const size_t nVectors = xTable->getNumberOfRows();

        kernel_function::KernelIfacePtr kernel = par->kernel->clone();
        DAAL_CHECK(kernel, ErrorNullParameterNotSupported);

        NumericTablePtr svCoeffTable = model->getClassificationCoefficients();
        const algorithmFPType bias(model->getBias());

        const size_t nSV = svCoeffTable->getNumberOfRows();

        size_t nRowsPerBlock = 0;
        DAAL_SAFE_CPU_CALL((nRowsPerBlock = 256), (nRowsPerBlock = nVectors));
        const size_t nBlocks = nVectors / nRowsPerBlock + !!(nVectors % nRowsPerBlock);

        size_t nSVPerBlock = 0;
        DAAL_SAFE_CPU_CALL((nSVPerBlock = 256), (nRowsPerBlock = nSV));
        const size_t nBlocksSV        = nSV / nSVPerBlock + !!(nSV % nSVPerBlock);
        const NumericTablePtr svTable = model->getSupportVectors();

        const bool isSparse = xTable->getDataLayout() == NumericTableIface::csrArray;

        /* TLS data initialization */
        using TPredictTask = PredictTask<algorithmFPType, cpu>;
        daal::ls<TPredictTask *> lsTask(
            [&]() {
                if (isSparse)
                {
                    return PredictTaskCSR<algorithmFPType, cpu>::create(nRowsPerBlock, nSVPerBlock, xTable, svTable, kernel);
                }
                else
                {
                    return PredictTaskDense<algorithmFPType, cpu>::create(nRowsPerBlock, nSVPerBlock, xTable, svTable, kernel);
                }
            },
            !isSparse);

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocksSV, nRowsPerBlock);
        daal::LsMem<algorithmFPType, cpu> lsDistance(nBlocksSV * nRowsPerBlock, nSV <= 256);
        SafeStatus safeStat;
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow          = iBlock * nRowsPerBlock;
            const size_t nRowsPerBlockReal = (iBlock != nBlocks - 1) ? nRowsPerBlock : nVectors - startRow;

            algorithmFPType * const distanceLocal = lsDistance.local();
            DAAL_CHECK_MALLOC_THR(distanceLocal);
            DAAL_LS_RELEASE(algorithmFPType, lsDistance, distanceLocal); //releases local storage when leaving this scope

            daal::conditional_threader_for((nSV > 256), nBlocksSV, [&](const size_t iBlockSV) {
                TPredictTask * lsLocal = lsTask.local();
                DAAL_CHECK_MALLOC_THR(lsLocal);
                DAAL_LS_RELEASE(TPredictTask, lsTask, lsLocal); //releases local storage when leaving this scope

                const size_t startSV         = iBlockSV * nSVPerBlock;
                const size_t nSVPerBlockReal = (iBlockSV != nBlocksSV - 1) ? nSVPerBlock : nSV - startSV;

                DAAL_CHECK_THR(lsLocal->kernelCompute(startRow, nRowsPerBlockReal, startSV, nSVPerBlockReal),
                               services::ErrorSVMPredictKernerFunctionCall);

                const algorithmFPType * const buffBlock = lsLocal->getBuff();

                char trans = 'T';
                DAAL_INT m = nSVPerBlockReal;
                DAAL_INT n = nRowsPerBlockReal;
                algorithmFPType alpha(1.0);
                DAAL_INT ldA = nSVPerBlockReal;
                DAAL_INT incX(1);
                algorithmFPType beta(0.0);
                DAAL_INT incY(1);

                ReadColumns<algorithmFPType, cpu> mtSVCoeff(*svCoeffTable, 0, startSV, nSVPerBlockReal);
                DAAL_CHECK_BLOCK_STATUS_THR(mtSVCoeff);
                const algorithmFPType * const svCoeff = mtSVCoeff.get();
                algorithmFPType * const distanceSV    = &distanceLocal[iBlockSV * nRowsPerBlock];

                if (nBlocks == 1)
                {
                    Blas<algorithmFPType, cpu>::xgemv(&trans, &m, &n, &alpha, buffBlock, &ldA, svCoeff, &incX, &beta, distanceSV, &incY);
                }
                else
                {
                    Blas<algorithmFPType, cpu>::xxgemv(&trans, &m, &n, &alpha, buffBlock, &ldA, svCoeff, &incX, &beta, distanceSV, &incY);
                }
            });

            WriteOnlyColumns<algorithmFPType, cpu> mtR(r, 0, startRow, nRowsPerBlockReal);
            DAAL_CHECK_BLOCK_STATUS_THR(mtR);
            algorithmFPType * const distanceBlock = mtR.get();
            service_memset_seq<algorithmFPType, cpu>(distanceBlock, bias, nRowsPerBlockReal);
            DAAL_INT n = nRowsPerBlockReal;
            algorithmFPType alpha(1.0);
            DAAL_INT incY(1);
            DAAL_INT incX(1);
            for (size_t iBlockSV = 0; iBlockSV < nBlocksSV; ++iBlockSV)
            {
                Blas<algorithmFPType, cpu>::xxaxpy(&n, &alpha, &distanceLocal[iBlockSV * nRowsPerBlock], &incX, distanceBlock, &incY);
            }
        });

        lsTask.reduce([](PredictTask<algorithmFPType, cpu> * local) { delete local; });
        return safeStat.detach();
    }
}; // namespace internal

} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

/* file: svm_train_thunder_cache.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  SVM cache structure implementation
//--
*/

#ifndef __SVM_TRAIN_THUNDER_CACHE_H__
#define __SVM_TRAIN_THUNDER_CACHE_H__

#include "service/kernel/service_utils.h"
#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "data_management/data/numeric_table_sycl_homogen.h"
#include "algorithms/kernel/svm/svm_train_cache.h"
#include "externals/service_service.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::data_management;

/**
 * Common interface for cache that stores kernel function values
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCacheIface<thunder, algorithmFPType, cpu> : public SVMCacheCommonIface<algorithmFPType, cpu>
{
public:
    virtual ~SVMCacheIface() {}

    virtual services::Status getRowsBlock(const uint32_t * indices, algorithmFPType *& block) = 0;

    virtual services::Status copyLastToFirst() = 0;

    virtual size_t getDataRowIndex(size_t rowIndex) const override { return rowIndex; }

protected:
    SVMCacheIface(const size_t cacheSize, const size_t lineSize, const kernel_function::KernelIfacePtr & kernel)
        : _lineSize(lineSize), _cacheSize(cacheSize), _kernel(kernel)
    {}

    const size_t _lineSize;                        /*!< Number of elements in the cache line */
    const size_t _cacheSize;                       /*!< Number of cache lines */
    const kernel_function::KernelIfacePtr _kernel; /*!< Kernel function */
};

/**
 * No cache: kernel function values are not cached
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCache<thunder, noCache, algorithmFPType, cpu> : public SVMCacheIface<thunder, algorithmFPType, cpu>
{
    using super    = SVMCacheIface<thunder, algorithmFPType, cpu>;
    using thisType = SVMCache<thunder, noCache, algorithmFPType, cpu>;
    using super::_kernel;
    using super::_lineSize;
    using super::_cacheSize;

public:
    ~SVMCache() {}

    DAAL_NEW_DELETE();

    static SVMCachePtr<thunder, algorithmFPType, cpu> create(const size_t cacheSize, const size_t lineSize, const NumericTablePtr & xTable,
                                                             const kernel_function::KernelIfacePtr & kernel, services::Status & status)
    {
        services::SharedPtr<thisType> res = services::SharedPtr<thisType>(new thisType(cacheSize, lineSize, xTable, kernel));
        if (!res)
        {
            status.add(ErrorMemoryAllocationFailed);
        }
        else
        {
            status = res->init(cacheSize);
            if (!status)
            {
                res.reset();
            }
        }
        return SVMCachePtr<thunder, algorithmFPType, cpu>(res);
    }

    services::Status getRowsBlock(const uint32_t * indices, algorithmFPType *& block) override
    {
        services::Status status;

        uint32_t * indicesNew = const_cast<uint32_t *>(indices);
        if (_isComputeSubKernel)
        {
            indicesNew = indicesNew + _nSelectRows;
        }

        DAAL_CHECK_STATUS(status, _blockTask->copyDataByIndices(indicesNew, _xTable));
        DAAL_ITTNOTIFY_SCOPED_TASK(cacheCompute);
        DAAL_CHECK_STATUS(status, _kernel->computeNoThrow());
        block = _cache.get();
        return status;
    }

    services::Status copyLastToFirst() override
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.copyLastToFirst);

        _nSelectRows = _cacheSize / 2;
        if (!_isComputeSubKernel)
        {
            reinit(_nSelectRows);
            _isComputeSubKernel = true;
        }
        const algorithmFPType * const dataIn = _cache.get() + _nSelectRows * _lineSize;
        algorithmFPType * const dataOut      = _cache.get();

        const size_t blockSize     = _cacheSize;
        const size_t nCopyElements = _nSelectRows * _lineSize;
        const size_t blockNum      = nCopyElements / blockSize;
        SafeStatus safeStat;
        daal::threader_for(blockNum, blockNum, [&](const size_t iBlock) {
            const size_t startRow = iBlock * blockSize;
            DAAL_CHECK_THR(!services::internal::daal_memcpy_s(&dataOut[startRow], blockSize * sizeof(algorithmFPType), &dataIn[startRow],
                                                              blockSize * sizeof(algorithmFPType)),
                           services::ErrorMemoryCopyFailedInternal);
        });
        return safeStat.detach();
    }

protected:
    SVMCache(const size_t cacheSize, const size_t lineSize, const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel)
        : super(cacheSize, lineSize, kernel), _nSelectRows(0), _isComputeSubKernel(false), _xTable(xTable)
    {}

    services::Status reinit(const size_t cacheSize)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.reinit);

        services::Status status;
        algorithmFPType * cacheHalf = _cache.get() + _lineSize * _nSelectRows;
        auto cacheTable             = HomogenNumericTableCPU<algorithmFPType, cpu>::create(cacheHalf, _lineSize, cacheSize, &status);

        const size_t p = _xTable->getNumberOfColumns();
        DAAL_CHECK_STATUS_VAR(status);

        SubDataTaskBase<algorithmFPType, cpu> * task = nullptr;
        if (_xTable->getDataLayout() == NumericTableIface::csrArray)
        {
            task = SubDataTaskCSR<algorithmFPType, cpu>::create(_xTable, cacheSize);
        }
        else
        {
            task = SubDataTaskDense<algorithmFPType, cpu>::create(p, cacheSize);
        }
        DAAL_CHECK_MALLOC(task);
        _blockTask = SubDataTaskBasePtr<algorithmFPType, cpu>(task);

        DAAL_CHECK_STATUS_VAR(status);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        _kernel->getInput()->set(kernel_function::X, _blockTask->getTableData());
        _kernel->getInput()->set(kernel_function::Y, _xTable);

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, cacheTable);
        _kernel->setResult(shRes);

        return status;
    }

    services::Status init(const size_t cacheSize)
    {
        services::Status status;
        _cache.reset(_lineSize * _cacheSize);
        DAAL_CHECK_MALLOC(_cache.get());
        auto cacheTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_cache.get(), _lineSize, _cacheSize, &status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS(status, reinit(_cacheSize));
        return status;
    }

protected:
    size_t _nSelectRows;
    bool _isComputeSubKernel;
    const NumericTablePtr & _xTable;
    SubDataTaskBasePtr<algorithmFPType, cpu> _blockTask;
    TArray<algorithmFPType, cpu> _cache;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

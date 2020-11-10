/* file: svm_train_cache_oneapi.h */
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

#ifndef __SVM_TRAIN_CACHE_ONEAPI_H__
#define __SVM_TRAIN_CACHE_ONEAPI_H__

#include "src/services/service_utils.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"
#include "src/algorithms/svm/oneapi/svm_helper_oneapi.h"

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
using namespace daal::services::internal::sycl;
using daal::data_management::internal::SyclHomogenNumericTable;

/**
 * Types of caches for kernel function values
 */
enum SVMCacheOneAPIType
{
    noCache,     /*!< No storage for caching kernel function values is provided */
    simpleCache, /*!< Storage for caching ALL kernel function values is provided */
    lruCache     /*!< Storage for caching PART of kernel function values is provided;
                         LRU algorithm is used to exclude values from cache */
};

/**
 * Common interface for cache that stores kernel function values
 */
template <typename algorithmFPType>
class SVMCacheOneAPIIface
{
public:
    virtual ~SVMCacheOneAPIIface() {}

    virtual services::Status compute(const NumericTablePtr & xTable, const services::internal::Buffer<uint32_t> & wsIndices, const size_t p) = 0;

    virtual const services::internal::Buffer<algorithmFPType> & getRowsBlock() const = 0;
    virtual services::Status copyLastToFirst()                                       = 0;

protected:
    SVMCacheOneAPIIface(const size_t blockSize, const size_t lineSize, const kernel_function::KernelIfacePtr & kernel)
        : _lineSize(lineSize), _blockSize(blockSize), _kernel(kernel)
    {}

    const size_t _lineSize;                        /*!< Number of elements in the cache line */
    const size_t _blockSize;                       /*!< Number of cache lines */
    const kernel_function::KernelIfacePtr _kernel; /*!< Kernel function */
};

template <SVMCacheOneAPIType cacheType, typename algorithmFPType>
class SVMCacheOneAPI
{};

template <typename algorithmFPType>
using SVMCacheOneAPIPtr = services::SharedPtr<SVMCacheOneAPIIface<algorithmFPType> >;

/**
 * No cache: kernel function values are not cached
 */
template <typename algorithmFPType>
class SVMCacheOneAPI<noCache, algorithmFPType> : public SVMCacheOneAPIIface<algorithmFPType>
{
    using Helper   = utils::internal::HelperSVM<algorithmFPType>;
    using super    = SVMCacheOneAPIIface<algorithmFPType>;
    using thisType = SVMCacheOneAPI<noCache, algorithmFPType>;
    using super::_kernel;
    using super::_lineSize;
    using super::_blockSize;

public:
    ~SVMCacheOneAPI() {}

    DAAL_NEW_DELETE();

    static SVMCacheOneAPIPtr<algorithmFPType> create(const size_t cacheSize, const size_t blockSize, const size_t lineSize,
                                                     const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel,
                                                     services::Status & status)
    {
        status.clear();
        services::SharedPtr<thisType> res = services::SharedPtr<thisType>(new thisType(blockSize, lineSize, xTable, kernel));
        if (!res)
        {
            status.add(ErrorMemoryAllocationFailed);
        }
        else
        {
            status = res->init(cacheSize, xTable);
            if (!status)
            {
                res.reset();
            }
        }
        return SVMCacheOneAPIPtr<algorithmFPType>(res);
    }

    const services::internal::Buffer<algorithmFPType> & getRowsBlock() const override { return _cacheBuff; }

    services::Status compute(const NumericTablePtr & xTable, const services::internal::Buffer<uint32_t> & wsIndices, const size_t p) override
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cacheCompute);

        services::Status status;
        BlockDescriptor<algorithmFPType> xBlock;

        DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(0, xTable->getNumberOfRows(), ReadWriteMode::readOnly, xBlock));
        const services::internal::Buffer<algorithmFPType> & xBuff = xBlock.getBuffer();

        size_t blockSize                                   = _blockSize;
        services::internal::Buffer<uint32_t> wsIndicesReal = wsIndices;
        if (_ifComputeSubKernel)
        {
            blockSize     = _blockSize / 2;
            wsIndicesReal = wsIndices.getSubBuffer(_nSelectRows, blockSize, status);
            DAAL_CHECK_STATUS_VAR(status);
            DAAL_CHECK_STATUS(status, initSubKernel(blockSize, xTable));
        }

        DAAL_CHECK_STATUS(status, Helper::copyDataByIndices(xBuff, wsIndicesReal, _xBlockBuff, blockSize, p));
        DAAL_CHECK_STATUS(status, xTable->releaseBlockOfRows(xBlock));

        DAAL_CHECK_STATUS(status, _kernel->computeNoThrow());
        return status;
    }

    services::Status copyLastToFirst() override
    {
        _nSelectRows        = _blockSize / 2;
        _ifComputeSubKernel = true;
        services::Status status;

        auto & context = services::internal::getDefaultContext();
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nSelectRows, _lineSize);
        context.copy(_cache, 0, _cache, _nSelectRows * _lineSize, _nSelectRows * _lineSize, status);
        return status;
    }

protected:
    SVMCacheOneAPI(const size_t blockSize, const size_t lineSize, const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel)
        : super(blockSize, lineSize, kernel), _nSelectRows(0), _ifComputeSubKernel(false)
    {}

    services::Status init(const size_t cacheSize, const NumericTablePtr & xTable)
    {
        services::Status status;
        auto & context = services::internal::getDefaultContext();

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _lineSize, _blockSize);
        _cache = context.allocate(TypeIds::id<algorithmFPType>(), _lineSize * _blockSize, status);
        DAAL_CHECK_STATUS_VAR(status);

        _cacheBuff      = _cache.get<algorithmFPType>();
        auto cacheTable = SyclHomogenNumericTable<algorithmFPType>::create(_cacheBuff, _lineSize, _blockSize, &status);

        const size_t p = xTable->getNumberOfColumns();

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _blockSize, p);
        _xBlock = context.allocate(TypeIds::id<algorithmFPType>(), _blockSize * p, status);
        DAAL_CHECK_STATUS_VAR(status);

        _xBlockBuff                    = _xBlock.get<algorithmFPType>();
        const NumericTablePtr xWSTable = SyclHomogenNumericTable<algorithmFPType>::create(_xBlockBuff, p, _blockSize, &status);

        DAAL_CHECK_STATUS_VAR(status);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        _kernel->getInput()->set(kernel_function::X, xWSTable);
        _kernel->getInput()->set(kernel_function::Y, xTable);

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, cacheTable);
        _kernel->setResult(shRes);

        return status;
    }

    services::Status initSubKernel(const size_t blockSize, const NumericTablePtr & xTable)
    {
        services::Status status;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _lineSize, blockSize);
        auto cacheHalf  = _cacheBuff.getSubBuffer(_lineSize * _nSelectRows, _lineSize * blockSize, status);
        auto cacheTable = SyclHomogenNumericTable<algorithmFPType>::create(cacheHalf, _lineSize, blockSize, &status);

        const size_t p = xTable->getNumberOfColumns();
        DAAL_CHECK_STATUS_VAR(status);

        const NumericTablePtr xWSTable = SyclHomogenNumericTable<algorithmFPType>::create(_xBlockBuff, p, blockSize, &status);

        DAAL_CHECK_STATUS_VAR(status);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        _kernel->getInput()->set(kernel_function::X, xWSTable);
        _kernel->getInput()->set(kernel_function::Y, xTable);

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, cacheTable);
        _kernel->setResult(shRes);

        return status;
    }

protected:
    size_t _nSelectRows;
    bool _ifComputeSubKernel;
    UniversalBuffer _cache;
    UniversalBuffer _xBlock;
    services::internal::Buffer<algorithmFPType> _xBlockBuff;
    services::internal::Buffer<algorithmFPType> _cacheBuff;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

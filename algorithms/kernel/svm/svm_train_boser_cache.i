/* file: svm_train_boser_cache.i */
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
//  SVM cache structure implementation
//--
*/

#ifndef __SVM_TRAIN_BOSER_CACHE_I__
#define __SVM_TRAIN_BOSER_CACHE_I__

#include "service_utils.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
using namespace daal::services::internal;

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

/**
 * Types of caches for kernel function values
 */
enum SVMCacheType
{
    noCache,        /*!< No storage for caching kernel function values is provided */
    simpleCache,    /*!< Storage for caching ALL kernel function values is provided */
    lruCache        /*!< Storage for caching PART of kernel function values is provided;
                         LRU algorithm is used to exclude values from cache */
};

template<typename algorithmFPType, CpuType cpu>
struct SVMCacheRowGetterIface
{
    virtual ~SVMCacheRowGetterIface() {}

    virtual algorithmFPType* getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize, size_t _lineSize,
                size_t cacheOffset, services::SharedPtr<kernel_function::KernelIface> kernel,
                algorithmFPType *_cache, services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable,
                size_t *shrinkingRowIndices) = 0;

    virtual void updateShrinkingRowIndices(size_t nActiveVectors, const char *I,
                size_t _nLines, size_t _lineSize, algorithmFPType *_cache, size_t *shrinkingRowIndices) = 0;
};

template<SVMCacheType cacheType, bool doShrinking, typename algorithmFPType, CpuType cpu>
struct SVMCacheRowGetter : public SVMCacheRowGetterIface<algorithmFPType, cpu> {};

template<typename algorithmFPType, CpuType cpu>
struct SVMCacheRowGetter<simpleCache, true, algorithmFPType, cpu>:
        public SVMCacheRowGetterIface<algorithmFPType, cpu>
{
    SVMCacheRowGetter(size_t _lineSize, services::SharedPtr<services::KernelErrorCollection> errors)
    {
        _tmp = (algorithmFPType *)daal::services::daal_malloc(_lineSize * sizeof(algorithmFPType));
        if (!_tmp) { errors->add(services::ErrorMemoryAllocationFailed); return; }
    }

    ~SVMCacheRowGetter()
    {
        daal::services::daal_free(_tmp);
    }

    algorithmFPType* getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize, size_t _lineSize,
                size_t cacheOffset, services::SharedPtr<kernel_function::KernelIface> kernel,
                algorithmFPType *_cache, services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable,
                size_t *shrinkingRowIndices)
    {
        return _cache + rowIndex * _lineSize + startColIndex;
    }

    void updateShrinkingRowIndices(size_t nActiveVectors, const char *I,
                size_t _nLines, size_t _lineSize, algorithmFPType *_cache, size_t *shrinkingRowIndices);
protected:
    algorithmFPType* _tmp;
};

template<typename algorithmFPType, CpuType cpu>
struct SVMCacheRowGetter<simpleCache, false, algorithmFPType, cpu>:
        public SVMCacheRowGetterIface<algorithmFPType, cpu>
{
    algorithmFPType* getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize, size_t _lineSize,
                size_t cacheOffset, services::SharedPtr<kernel_function::KernelIface> kernel,
                algorithmFPType *_cache, services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable,
                size_t *shrinkingRowIndices)
    {
        return _cache + rowIndex * _lineSize + startColIndex;
    }

    void updateShrinkingRowIndices(size_t nActiveVectors, const char *I,
                size_t _nLines, size_t _lineSize, algorithmFPType *_cache, size_t *shrinkingRowIndices)
    {}
};

template<typename algorithmFPType, CpuType cpu>
struct SVMCacheRowGetter<noCache, true, algorithmFPType, cpu>:
        public SVMCacheRowGetterIface<algorithmFPType, cpu>
{
    algorithmFPType* getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize, size_t _lineSize,
                size_t cacheOffset, services::SharedPtr<kernel_function::KernelIface> kernel,
                algorithmFPType *_cache, services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable,
                size_t *shrinkingRowIndices)
    {
        _cacheTable->setArray(_cache + cacheOffset);
        kernel->parameterBase->rowIndexY = shrinkingRowIndices[rowIndex];
        for (size_t i = 0; i < blockSize; i++)
        {
            kernel->parameterBase->rowIndexX       = shrinkingRowIndices[startColIndex + i];
            kernel->parameterBase->rowIndexResult  = i;
            kernel->computeNoThrow();
        }
        return _cache + cacheOffset;
    }

    void updateShrinkingRowIndices(size_t nActiveVectors, const char *I,
                size_t _nLines, size_t _lineSize, algorithmFPType *_cache, size_t *shrinkingRowIndices);
};

template<typename algorithmFPType, CpuType cpu>
struct SVMCacheRowGetter<noCache, false, algorithmFPType, cpu>:
        public SVMCacheRowGetterIface<algorithmFPType, cpu>
{
    algorithmFPType* getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize, size_t _lineSize,
                size_t cacheOffset, services::SharedPtr<kernel_function::KernelIface> kernel,
                algorithmFPType *_cache, services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable,
                size_t *shrinkingRowIndices)
    {
        _cacheTable->setArray(_cache + cacheOffset);
        kernel->parameterBase->rowIndexY = rowIndex;
        for (size_t i = 0; i < blockSize; i++)
        {
            kernel->parameterBase->rowIndexX       = startColIndex + i;
            kernel->parameterBase->rowIndexResult  = i;
            kernel->computeNoThrow();
        }
        return _cache + cacheOffset;
    }

    void updateShrinkingRowIndices(size_t nActiveVectors, const char *I,
                size_t _nLines, size_t _lineSize, algorithmFPType *_cache, size_t *shrinkingRowIndices)
    {}
};

/**
 * Common interface for cache that stores kernel function values
 */
template<typename algorithmFPType, CpuType cpu>
class SVMCacheIface
{
public:
    virtual ~SVMCacheIface() {}

    virtual size_t getDataRowIndex(size_t rowIndex) const = 0;
    /**
     * Get block of values from the row of the matrix Q (kernel(x[i], x[j]))
     * \param[in] rowIndex      Index of the requested row
     * \param[in] startColIndex Starting columns index of the requested block of values
     * \param[in] blockSize     Number of requested values
     * \return Block of values from the row of the matirx Q
     */
    virtual algorithmFPType *getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize) = 0;

    /**
     * Get blocks of values from the two rows of the matrix Q (kernel(x[i], x[j]))
     * \param[in] rowIndex1     Index of the first requested row
     * \param[in] rowIndex2     Index of the second requested row
     * \param[in] startColIndex Starting columns index of the requested blocks of values
     * \param[in] blockSize     Number of requested values in each block
     * \param[out] block1       Pointer to the first  block of values
     * \param[out] block2       Pointer to the second block of values
     */
    virtual void getTwoRowsBlock(size_t rowIndex1, size_t rowIndex2, size_t startColIndex, size_t blockSize,
                                 algorithmFPType **block1, algorithmFPType **block2) = 0;

    /**
     * Get type of the cache: noCache, simpleCache or lruCache
     * \return Type of the cache
     */
    virtual SVMCacheType getType() const = 0;

    /**
     * Move the indices of the shrunk feature vector to the end of the array
     *
     * \param[in] nActiveVectors Number of observations in a training data set that are used
     *                           in sequential minimum optimization at the current iteration
     * \param[in] I              Array of flags that describe the status of feature vectors
     */
    virtual void updateShrinkingRowIndices(size_t nActiveVectors, const char *I) = 0;
};

/**
 * Common implementation for cache that stores kernel function values
 */
template<typename algorithmFPType, CpuType cpu>
class SVMCacheImpl : public SVMCacheIface<algorithmFPType, cpu>
{
public:
    /**
     * Constructs cache
     *
     * \param[in] lineSize      Number of elements in the cache line
     * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
     * \param[in] kernel        Kernel function
     * \param[in] errors        Pointer to error collection associated with SVM training algorithm
     */
    SVMCacheImpl(size_t lineSize, bool doShrinking, services::SharedPtr<kernel_function::KernelIface> kernel,
                  services::SharedPtr<services::KernelErrorCollection> errors) :
        _lineSize(lineSize), doShrinking(doShrinking), shrinkingRowIndices(NULL),
        _kernel(kernel), _errors(errors)
    {
        if (doShrinking)
        {
            shrinkingRowIndices = (size_t *)daal::services::daal_malloc(_lineSize * sizeof(size_t));
            if (!shrinkingRowIndices) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }
            for (size_t i = 0; i < lineSize; i++)
            {
                shrinkingRowIndices[i] = i;
            }
        }
    }

    virtual ~SVMCacheImpl()
    {
        if (shrinkingRowIndices) { daal::services::daal_free(shrinkingRowIndices); }
    }

    virtual size_t getDataRowIndex(size_t rowIndex) const
    {
        if (doShrinking)
        {
            return shrinkingRowIndices[rowIndex];
        }
        else
        {
            return rowIndex;
        }
    }

    bool doShrinking;               /*!< Flag that enables use of the shrinking optimization technique */
    size_t *shrinkingRowIndices;    /*!< Array of input data row indices used with shrinking technique */
protected:
    algorithmFPType *_cache;
    size_t _lineSize;               /*!< Number of elements in the cache line */
    services::SharedPtr<kernel_function::KernelIface> _kernel;      /*!< Kernel function */
    services::SharedPtr<services::KernelErrorCollection> _errors;
    SVMCacheRowGetterIface<algorithmFPType, cpu> *rowGetter;
};

template<SVMCacheType cacheType, typename algorithmFPType, CpuType cpu>
class SVMCache {};

/**
 * Simple cache: all elements of kernel matrix fit into cache
 */
template<typename algorithmFPType, CpuType cpu>
class SVMCache<simpleCache, algorithmFPType, cpu> : public SVMCacheImpl<algorithmFPType, cpu>
{
    using SVMCacheImpl<algorithmFPType, cpu>::_cache;
    using SVMCacheImpl<algorithmFPType, cpu>::_kernel;
    using SVMCacheImpl<algorithmFPType, cpu>::_lineSize;
    using SVMCacheImpl<algorithmFPType, cpu>::shrinkingRowIndices;
    using SVMCacheImpl<algorithmFPType, cpu>::doShrinking;
    using SVMCacheImpl<algorithmFPType, cpu>::rowGetter;
public:
    /**
     * Constructs simple cache
     *
     * \param[in] lineSize      Number of elements in the cache line
     * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
     * \param[in] xTable        Input data set
     * \param[in] kernel        Kernel function
     * \param[in] errors        Pointer to error collection associated with SVM training algorithm
     */
    SVMCache(size_t cacheSize, size_t lineSize, bool doShrinking, NumericTablePtr xTable,
             services::SharedPtr<kernel_function::KernelIface> kernel,
             services::SharedPtr<services::KernelErrorCollection> errors) :
        SVMCacheImpl<algorithmFPType, cpu>(lineSize, doShrinking, kernel, errors), _nLines(lineSize)
    {
        _cache = (algorithmFPType *)daal::services::daal_malloc(_lineSize * _nLines * sizeof(algorithmFPType));
        if (!_cache) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }
        if (doShrinking)
        {
            rowGetter = new SVMCacheRowGetter<simpleCache, true,  algorithmFPType, cpu>(_lineSize, errors);
        }
        else
        {
            rowGetter = new SVMCacheRowGetter<simpleCache, false, algorithmFPType, cpu>();
        }
        _cacheTable = services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> >(new HomogenNumericTableCPU<algorithmFPType, cpu>(_cache,
                                                                                         _lineSize, _nLines));

        _kernel->parameterBase->computationMode = kernel_function::matrixMatrix;
        _kernel->inputBase->set(kernel_function::X, xTable);
        _kernel->inputBase->set(kernel_function::Y, xTable);

        services::SharedPtr<kernel_function::Result> shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, _cacheTable);
        _kernel->setResult(shRes);

        _kernel->computeNoThrow();
        if(_kernel->getErrors()->size() != 0) {errors->add(_kernel->getErrors()->getErrors()); return;}
    }

    /**
     * Get block of values from the row of the matrix Q (kernel(x[i], x[j]))
     * \param[in] rowIndex      Index of the requested row
     * \param[in] startColIndex Starting columns index of the requested block of values
     * \param[in] blockSize     Number of requested values
     * \return Block of values from the row of the matirx Q
     */
    algorithmFPType *getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize)
    {
        return rowGetter->getRowBlock(rowIndex, startColIndex, blockSize, _lineSize, 0, _kernel,
                                      _cache, _cacheTable, shrinkingRowIndices);
    }

    /**
     * Get blocks of values from the two rows of the matrix Q (kernel(x[i], x[j]))
     * \param[in] rowIndex1     Index of the first requested row
     * \param[in] rowIndex2     Index of the second requested row
     * \param[in] startColIndex Starting columns index of the requested blocks of values
     * \param[in] blockSize     Number of requested values in each block
     * \param[out] block1       Pointer to the first  block of values
     * \param[out] block2       Pointer to the second block of values
     */
    void getTwoRowsBlock(size_t rowIndex1, size_t rowIndex2, size_t startColIndex, size_t blockSize,
                         algorithmFPType **block1, algorithmFPType **block2)
    {
        *block1 = rowGetter->getRowBlock(rowIndex1, startColIndex, blockSize, _lineSize, 0, _kernel,
                                         _cache, _cacheTable, shrinkingRowIndices);
        *block2 = rowGetter->getRowBlock(rowIndex2, startColIndex, blockSize, _lineSize, 0, _kernel,
                                         _cache, _cacheTable, shrinkingRowIndices);
    }

    /**
     * Get type of the cache: noCache, simpleCache or lruCache
     * \return Type of the cache
     */
    SVMCacheType getType() const { return simpleCache; }

    /**
     * Move the indices of the shrunk feature vector to the end of the array
     *
     * \param[in] nActiveVectors Number of observations in a training data set that are used
     *                           in sequential minimum optimization at the current iteration
     * \param[in] I              Array of flags that describe the status of feature vectors
     */
    virtual void updateShrinkingRowIndices(size_t nActiveVectors, const char *I)
    {
        rowGetter->updateShrinkingRowIndices(nActiveVectors, I, _nLines, _lineSize, _cache, shrinkingRowIndices);
    }

    ~SVMCache()
    {
        daal::services::daal_free(_cache);
        delete rowGetter;
    }

protected:
    size_t _nLines;
    services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable;
};

/**
 * No cache: kernel function values are not cached
 */
template<typename algorithmFPType, CpuType cpu>
class SVMCache<noCache, algorithmFPType, cpu> : public SVMCacheImpl<algorithmFPType, cpu>
{
    using SVMCacheImpl<algorithmFPType, cpu>::_cache;
    using SVMCacheImpl<algorithmFPType, cpu>::_kernel;
    using SVMCacheImpl<algorithmFPType, cpu>::_lineSize;
    using SVMCacheImpl<algorithmFPType, cpu>::shrinkingRowIndices;
    using SVMCacheImpl<algorithmFPType, cpu>::doShrinking;
    using SVMCacheImpl<algorithmFPType, cpu>::rowGetter;
public:
    /**
     * Constructs the cache that doesn't cache kernel function values
     *
     * \param[in] lineSize      Number of elements in the cache line
     * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
     * \param[in] xTable        Input data set
     * \param[in] kernel        Kernel function
     * \param[in] errors        Pointer to error collection associated with SVM training algorithm
     */
    SVMCache(size_t cacheSize, size_t lineSize, bool doShrinking, NumericTablePtr xTable,
             services::SharedPtr<kernel_function::KernelIface> kernel,
             services::SharedPtr<services::KernelErrorCollection> errors) :
        SVMCacheImpl<algorithmFPType, cpu>(lineSize, doShrinking, kernel, errors)
    {
        _cache = (algorithmFPType *)daal::services::daal_malloc(cacheSize * sizeof(algorithmFPType));
        if (!_cache) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

        _cacheTable = services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> >(
            new HomogenNumericTableCPU<algorithmFPType, cpu>(NULL, 1, lineSize));
        services::SharedPtr<kernel_function::Result> result =
            services::SharedPtr<kernel_function::Result>(new kernel_function::Result());
        result->set(kernel_function::values, _cacheTable);
        _kernel->setResult(result);

        _kernel->inputBase->set(kernel_function::X, xTable);
        _kernel->inputBase->set(kernel_function::Y, xTable);
        _kernel->parameterBase->computationMode = kernel_function::vectorVector;
        if (doShrinking)
        {
            rowGetter = new SVMCacheRowGetter<noCache, true,  algorithmFPType, cpu>();
        }
        else
        {
            rowGetter = new SVMCacheRowGetter<noCache, false, algorithmFPType, cpu>();
        }
    }

    /**
     * Get block of values from the row of the matrix Q (kernel(x[i], x[j]))
     * \param[in] rowIndex      Index of the requested row
     * \param[in] startColIndex Starting columns index of the requested block of values
     * \param[in] blockSize     Number of requested values
     * \return Block of values from the row of the matirx Q
     */
    algorithmFPType *getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize)
    {
        return rowGetter->getRowBlock(rowIndex, startColIndex, blockSize, _lineSize, 0, _kernel,
                                      _cache, _cacheTable, shrinkingRowIndices);
    }

    /**
     * Get blocks of values from the two rows of the matrix Q (kernel(x[i], x[j]))
     * \param[in] rowIndex1     Index of the first requested row
     * \param[in] rowIndex2     Index of the second requested row
     * \param[in] startColIndex Starting columns index of the requested blocks of values
     * \param[in] blockSize     Number of requested values in each block
     * \param[out] block1       Pointer to the first  block of values
     * \param[out] block2       Pointer to the second block of values
     */
    void getTwoRowsBlock(size_t rowIndex1, size_t rowIndex2, size_t startColIndex, size_t blockSize,
                    algorithmFPType **block1, algorithmFPType **block2)
    {
        *block1 = rowGetter->getRowBlock(rowIndex1, startColIndex, blockSize, _lineSize, 0, _kernel,
                                         _cache, _cacheTable, shrinkingRowIndices);
        *block2 = rowGetter->getRowBlock(rowIndex2, startColIndex, blockSize, _lineSize, blockSize, _kernel,
                                         _cache, _cacheTable, shrinkingRowIndices);
    }

    /**
     * Get type of the cache: noCache, simpleCache or lruCache
     * \return Type of the cache
     */
    SVMCacheType getType() const { return noCache; }

    /**
     * Move the indices of the shrunk feature vector to the end of the array
     *
     * \param[in] nActiveVectors Number of observations in a training data set that are used
     *                           in sequential minimum optimization at the current iteration
     * \param[in] I              Array of flags that describe the status of feature vectors
     */
    virtual void updateShrinkingRowIndices(size_t nActiveVectors, const char *I)
    {
        rowGetter->updateShrinkingRowIndices(nActiveVectors, I, 0, _lineSize, _cache, shrinkingRowIndices);
    }

    ~SVMCache()
    {
        daal::services::daal_free(_cache);
        delete rowGetter;
    }
protected:
    services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable;
};

} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif

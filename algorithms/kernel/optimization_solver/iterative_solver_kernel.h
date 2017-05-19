/* file: iterative_solver_kernel.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Declaration of template function that calculate iterative solver.
//--


#ifndef __ITERATIVE_SOLVER_KERNEL_H__
#define __ITERATIVE_SOLVER_KERNEL_H__

#include "kernel.h"
#include "service_rng.h"
#include "service_math.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "data_management/data/memory_block.h"
#include "threading.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace iterative_solver
{
namespace internal
{
/**
 *  \brief Kernel for iterative_solver calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */

template<CpuType cpu, typename F>
void processByBlocks(size_t nRows, const F &processBlock, size_t minRowsNumInBlock = 1024, size_t blockStartThreshold = 5000)
{
    if(nRows < blockStartThreshold) // if number of rows is less that blockStartThreshold do sequential mode
        return processBlock(0, nRows);

    size_t nBlocks = nRows / minRowsNumInBlock;
    nBlocks += (nBlocks * minRowsNumInBlock != nRows);

    daal::threader_for(nBlocks, nBlocks, [=](size_t block)
    {
        const size_t nRowsInBlock = (block == nBlocks - 1) ? (nRows - block * minRowsNumInBlock) : minRowsNumInBlock;
        processBlock(block * minRowsNumInBlock, nRowsInBlock);
    } );
}

template<typename algorithmFPType, CpuType cpu>
class IterativeSolverKernel : public Kernel
{
protected:
    services::Status vectorNorm(NumericTable *vecNT, algorithmFPType& res)
    {
        res = 0;
        daal::tls<algorithmFPType *> normTls( [ = ]()-> algorithmFPType*
        {
            algorithmFPType *normPtr = (algorithmFPType *)daal_malloc(sizeof(algorithmFPType));
            *normPtr = 0;
            return normPtr;
        } );

        SafeStatus safeStat;
        processByBlocks<cpu>(vecNT->getNumberOfRows(), [=, &normTls, &safeStat](size_t startOffset, size_t nRowsInBlock)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> vecBD;
            algorithmFPType *vecLocal = vecBD.set(vecNT, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(vecBD);
            algorithmFPType *normPtr = normTls.local();
            DAAL_CHECK_THR(normPtr, services::ErrorMemoryAllocationFailed);
            PRAGMA_VECTOR_ALWAYS
            PRAGMA_IVDEP
            for(int j = 0; j < nRowsInBlock; j++)
            {
                *normPtr += vecLocal[j] * vecLocal[j];
            }
        },
        256);
        normTls.reduce( [ =, &res ](algorithmFPType * normPtr)-> void
        {
            res += *normPtr;
            daal_free( normPtr );
        });
        res = daal::internal::Math<algorithmFPType, cpu>::sSqrt(res); // change to sqNorm
        return safeStat.detach();
    }

    services::Status vectorNorm(const algorithmFPType *vec, size_t nElements, algorithmFPType& res)
    {
        res = 0;
        if(nElements < 5000)
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALIGNED
            for(size_t j = 0; j < nElements; j++)
            {
                res += vec[j] * vec[j];
            }
            res = daal::internal::Math<algorithmFPType, cpu>::sSqrt(res); // change to sqNorm
            return services::Status();
        }
        // algorithmFPType norm = 0;
        daal::tls<algorithmFPType *> normTls( [ = ]()-> algorithmFPType*
        {
            algorithmFPType *normPtr = (algorithmFPType *)daal_malloc(sizeof(algorithmFPType));
            *normPtr = 0;
            return normPtr;
        } );

        int _nRowsInBlock = 256;
        size_t nBlocks = nElements / _nRowsInBlock;
        nBlocks += (nBlocks * _nRowsInBlock != nElements);
        SafeStatus safeStat;
        daal::threader_for(nBlocks, nBlocks, [=, &normTls, &safeStat](size_t block)
        {
            algorithmFPType *normPtr = normTls.local();
            DAAL_CHECK_THR(normPtr, services::ErrorMemoryAllocationFailed);
            const algorithmFPType *vecLocal = &vec[block * _nRowsInBlock];
            int localArraySize = (block == nBlocks - 1) ? (int)(nElements - block * _nRowsInBlock) : _nRowsInBlock;
            PRAGMA_SIMD_ASSERT
            for(int j = 0; j < localArraySize; j++)
            {
                *normPtr += vecLocal[j] * vecLocal[j];
            }
        } );

        normTls.reduce( [ =, &res ](algorithmFPType * normPtr)-> void
        {
            res += *normPtr;
        });
        res = daal::internal::Math<algorithmFPType, cpu>::sSqrt(res); // change to sqNorm
        return safeStat.detach();
    }

    services::Status getRandom(int minVal, int maxVal, int *randomValue, int nRandomValues, size_t seed)
    {
        daal::internal::BaseRNGs<cpu> baseRng((int)seed);
        daal::internal::RNGs<int, cpu> rng;
        DAAL_CHECK(rng.uniform(nRandomValues, randomValue, baseRng, minVal, maxVal), ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }
};

template<typename algorithmFPType, CpuType cpu>
class RngTask
{
public:
    enum EMode
    {
        eUniform,
        eUniformWithoutReplacement
    };
    RngTask(const int *predefined, size_t size) : _predefined(predefined), _size(size), _rngChanged(false),
        _maxVal(0), _values(0), _brng(nullptr) {}
    ~RngTask()
    {
        delete _brng;
    }

    bool init(OptionalArgument *optionalArgument, int maxVal, int seed, size_t rngStateIdx)
    {
        _values.reset(_size);
        if(!_values.get())
            return false;

        _maxVal = maxVal;
        _brng = new BaseRNGType(seed);
        auto pOpt = optionalArgument;
        if(pOpt)
        {
            data_management::MemoryBlock *pState = dynamic_cast<data_management::MemoryBlock *>(pOpt->get(rngStateIdx).get());
            if(pState && pState->size())
                _brng->loadState(pState->get());
        }
        return true;
    }

    services::Status get(const int *& pValues, EMode eMode = eUniform)
    {
        if(_predefined)
        {
            pValues = _predefined;
            _predefined += _size;
            return services::Status();
        }
        const int errCode = (eMode == eUniform ? RNGsType().uniform((int)_size, _values.get(), *_brng, 0, _maxVal) :
            RNGsType().uniformWithoutReplacement((int)_size, _values.get(), *_brng, 0, _maxVal));
        DAAL_CHECK(errCode == 0, ErrorIncorrectErrorcodeFromGenerator);

        _rngChanged = true;
        pValues = _values.get();
        return Status();
    }

    services::Status save(OptionalArgument *optionalResult, size_t rngStateIdx) const
    {
        if(!_brng || !_rngChanged)
            return services::Status();

        auto pOpt = optionalResult;
        //pOpt should exist by now
        data_management::MemoryBlock *pState = dynamic_cast<data_management::MemoryBlock *>(pOpt->get(rngStateIdx).get());
        //pState should exist by now
        auto stateSize = _brng->getStateSize();
        if(!stateSize)
            return services::Status();

        pState->reserve(stateSize);
        DAAL_CHECK_MALLOC(pState->get());
        DAAL_CHECK(_brng->saveState(pState->get()) == 0, ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

protected:
    typedef daal::internal::RNGs <int, cpu> RNGsType;
    typedef daal::internal::BaseRNGs<cpu> BaseRNGType;
    BaseRNGType *_brng;
    const int *_predefined;
    size_t _size;
    bool _rngChanged;
    TArray<int, cpu> _values;
    int _maxVal;
};

} // namespace daal::internal
} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif

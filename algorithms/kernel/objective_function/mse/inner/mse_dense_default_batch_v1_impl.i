/* file: mse_dense_default_batch_v1_impl.i */
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

/*
//++
//  Implementation of mse algorithm
//--
*/
#include "service_math.h"

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
/**
 *  \brief Kernel for mse objective function calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
inline services::Status I1MSEKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataNT, NumericTable * dependentVariablesNT,
                                                                           NumericTable * argumentNT, NumericTable * valueNT,
                                                                           NumericTable * hessianNT, NumericTable * gradientNT,
                                                                           NumericTable * nonSmoothTermValue, NumericTable * proximalProjection,
                                                                           NumericTable * lipschitzConstant, interface1::Parameter * parameter)
{
    const size_t nDataRows = dataNT->getNumberOfRows();

    if (proximalProjection)
    {
        const size_t nBeta = argumentNT->getNumberOfRows();
        DAAL_ASSERT(proximalProjection->getNumberOfRows() == nBeta);
        WriteRows<algorithmFPType, cpu> prox(proximalProjection, 0, nBeta);
        ReadRows<algorithmFPType, cpu> beta(argumentNT, 0, nBeta);

        algorithmFPType * p       = prox.get();
        const algorithmFPType * b = beta.get();

        for (int i = 0; i < nBeta; i++)
        {
            p[i] = b[i];
        }
        return services::Status();
    }

    if (lipschitzConstant)
    {
        const size_t n = dataNT->getNumberOfRows();
        const size_t p = dataNT->getNumberOfColumns();
        ReadRows<algorithmFPType, cpu> xPtr(dataNT, 0, n);
        const algorithmFPType * x = xPtr.get();
        DAAL_ASSERT(lipschitzConstant->getNumberOfRows() == 1);
        WriteRows<algorithmFPType, cpu> lipschitzConstantPtr(lipschitzConstant, 0, 1);
        algorithmFPType & c = *lipschitzConstantPtr.get();

        const size_t blockSize = 256;
        size_t nBlocks         = n / blockSize;
        nBlocks += (nBlocks * blockSize != n);
        algorithmFPType globalMaxNorm = 0;

        TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> > tlsData(lipschitzConstant->getNumberOfRows());
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            algorithmFPType & _maxNorm = *tlsData.local();
            const size_t startRow      = iBlock * blockSize;
            const size_t finishRow     = (iBlock + 1 == nBlocks ? n : (iBlock + 1) * blockSize);
            algorithmFPType curentNorm = 0;
            for (size_t i = startRow; i < finishRow; i++)
            {
                curentNorm = 0;
                for (int j = 0; j < p; j++)
                {
                    curentNorm += x[i * p + j] * x[i * p + j];
                }
                if (curentNorm > _maxNorm)
                {
                    _maxNorm = curentNorm;
                }
            }
        });
        tlsData.reduce([&](algorithmFPType * maxNorm) {
            if (globalMaxNorm < *maxNorm)
            {
                globalMaxNorm = *maxNorm;
            }
        });

        algorithmFPType lipschitz = (globalMaxNorm + 1);
        c                         = 2 * lipschitz;
        return services::Status();
    }

    if (nonSmoothTermValue)
    {
        WriteRows<algorithmFPType, cpu> vr(nonSmoothTermValue, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(vr);
        algorithmFPType & v = *vr.get();
        v                   = 0;
        return services::Status();
    }

    if (parameter->batchIndices.get() != NULL && parameter->batchIndices->getNumberOfColumns() != nDataRows)
    {
        I1MSETaskSample<algorithmFPType, cpu> task(dataNT, dependentVariablesNT, argumentNT, valueNT, hessianNT, gradientNT, parameter,
                                                   blockSizeDefault);
        return run(task);
    }
    I1MSETaskAll<algorithmFPType, cpu> task(dataNT, dependentVariablesNT, argumentNT, valueNT, hessianNT, gradientNT, parameter, blockSizeDefault);
    return run(task);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status I1MSEKernel<algorithmFPType, method, cpu>::run(I1MSETask<algorithmFPType, cpu> & task)
{
    algorithmFPType * argumentArray = nullptr;
    Status s                        = task.init(argumentArray);
    if (!s) return s;
    algorithmFPType *dataBlock = nullptr, *dependentVariablesBlock = nullptr;
    algorithmFPType *value = nullptr, *gradient = nullptr, *hessian = nullptr;
    DAAL_CHECK_STATUS(s, task.getResultValues(value, gradient, hessian));
    task.setResultValuesToZero(value, gradient, hessian);

    size_t blockSize = blockSizeDefault;
    size_t nBlocks   = task.batchSize / blockSizeDefault;
    nBlocks += (nBlocks * blockSizeDefault != task.batchSize);
    if (nBlocks == 1)
    {
        blockSize = task.batchSize;
    }

    for (size_t block = 0; s.ok() && (block < nBlocks); block++)
    {
        if (block == nBlocks - 1) blockSize = task.batchSize - block * blockSizeDefault;

        s = task.getCurrentBlock(block * blockSizeDefault, blockSize, dataBlock, dependentVariablesBlock);
        if (s) computeMSE(blockSize, task, dataBlock, argumentArray, dependentVariablesBlock, value, gradient, hessian);
        task.releaseCurrentBlock();
    }

    if (s) normalizeResults(task, value, gradient, hessian);
    task.releaseResultValues();
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
inline void I1MSEKernel<algorithmFPType, method, cpu>::computeMSE(size_t blockSize, I1MSETask<algorithmFPType, cpu> & task, algorithmFPType * data,
                                                                  algorithmFPType * argumentArray, algorithmFPType * dependentVariablesArray,
                                                                  algorithmFPType * value, algorithmFPType * gradient, algorithmFPType * hessian)
{
    char trans                   = 'T';
    algorithmFPType one          = 1.0;
    algorithmFPType zero         = 0.0;
    DAAL_INT n                   = (DAAL_INT)blockSize;
    size_t nTheta                = task.nTheta;
    DAAL_INT dim                 = (DAAL_INT)nTheta;
    DAAL_INT ione                = 1;
    algorithmFPType theta0       = argumentArray[0];
    algorithmFPType * theta      = &argumentArray[1];
    algorithmFPType * xMultTheta = task.xMultTheta.get();

    if (task.gradientFlag || task.valueFlag)
    {
        Blas<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &one, data, &dim, theta, &ione, &zero, xMultTheta, &ione);

        for (size_t i = 0; i < blockSize; i++)
        {
            xMultTheta[i] = xMultTheta[i] + theta0 - dependentVariablesArray[i];
        }
    }

    if (task.gradientFlag)
    {
        for (size_t i = 0; i < blockSize; i++)
        {
            gradient[0] += xMultTheta[i];
            for (size_t j = 0; j < nTheta; j++)
            {
                gradient[j + 1] += xMultTheta[i] * data[i * nTheta + j];
            }
        }
    }

    if (task.valueFlag)
    {
        for (size_t i = 0; i < blockSize; i++)
        {
            value[0] += xMultTheta[i] * xMultTheta[i];
        }
    }

    if (task.hessianFlag)
    {
        char uplo             = 'U';
        char notrans          = 'N';
        DAAL_INT argumentSize = dim + 1;

        Blas<algorithmFPType, cpu>::xsyrk(&uplo, &notrans, &dim, &n, &one, data, &dim, &one, hessian + argumentSize + 1, &argumentSize);

        for (size_t i = 0; i < blockSize; i++)
        {
            for (size_t j = 0; j < nTheta; j++)
            {
                hessian[j + 1] += data[i * nTheta + j];
            }
        }

        for (size_t i = 0; i < argumentSize; i++)
        {
            for (size_t j = 1; j < i; j++)
            {
                hessian[j * argumentSize + i] = hessian[i * argumentSize + j];
            }
            hessian[i * argumentSize] = hessian[i];
        }
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void I1MSEKernel<algorithmFPType, method, cpu>::normalizeResults(I1MSETask<algorithmFPType, cpu> & task, algorithmFPType * value,
                                                                 algorithmFPType * gradient, algorithmFPType * hessian)
{
    size_t argumentSize          = task.argumentSize;
    const algorithmFPType one    = 1.0;
    algorithmFPType batchSizeInv = (algorithmFPType)one / task.batchSize;
    if (task.valueFlag)
    {
        value[0] /= (algorithmFPType)(2 * task.batchSize);
    }

    if (task.gradientFlag)
    {
        for (size_t j = 0; j < argumentSize; j++)
        {
            gradient[j] *= batchSizeInv;
        }
    }

    if (task.hessianFlag)
    {
        hessian[0] = one;
        for (size_t j = 1; j < argumentSize * argumentSize; j++)
        {
            hessian[j] *= batchSizeInv;
        }
    }
}

} // namespace internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

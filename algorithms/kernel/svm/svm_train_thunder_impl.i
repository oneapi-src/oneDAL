/* file: svm_train_thunder_impl.i */
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
//  SVM training algorithm implementation
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Zeyi Wen, Jiashuai Shi, Bingsheng He
//     ThunderSVM: A Fast SVM Library on GPUs and CPUs,
//     Journal of Machine Learning Research, 19, 1-5 (2018)
//  2. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  3. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  4. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_THUNDER_I__
#define __SVM_TRAIN_THUNDER_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_data_utils.h"
#include "externals/service_ittnotify.h"

#include "algorithms/kernel/svm/svm_train_thunder_workset.h"
#include "algorithms/kernel/svm/svm_train_thunder_cache.h"

#include <set>

#define PRINT_N 8

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
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, ParameterType, cpu>::compute(const NumericTablePtr & xTable, NumericTable & yTable,
                                                                                     daal::algorithms::Model * r, const ParameterType * svmPar)
{
    printf("SVMTrainImpl:compute\n");
    services::Status status;

    const algorithmFPType C(svmPar->C);
    const algorithmFPType eps(svmPar->accuracyThreshold);
    const algorithmFPType tau(svmPar->tau);
    const size_t maxIterations(svmPar->maxIterations);
    const size_t cacheSize(svmPar->cacheSize);
    kernel_function::KernelIfacePtr kernel = svmPar->kernel->clone();

    const size_t nVectors  = xTable->getNumberOfRows();
    const size_t nFeatures = xTable->getNumberOfColumns();
    // ai = 0

    TArrayCalloc<algorithmFPType, cpu> alphaTArray(nVectors);
    DAAL_CHECK_MALLOC(alphaTArray.get());
    algorithmFPType * alpha = alphaTArray.get();

    ReadColumns<algorithmFPType, cpu> mtY(yTable, 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtY);
    const algorithmFPType * y = mtY.get();

    // gradi = -yi
    TArray<algorithmFPType, cpu> gradTArray(nVectors);
    DAAL_CHECK_MALLOC(gradTArray.get());
    algorithmFPType * grad = gradTArray.get();

    for (size_t i = 0; i < nVectors; i++)
    {
        grad[i] = -y[i];
    }

    TaskWorkingSet<algorithmFPType, cpu> workSet(nVectors);
    DAAL_CHECK_STATUS(status, workSet.init());
    const size_t nWS = workSet.getSize();
    printf("%lu\n", nWS);
    const size_t innerMaxIterations(nWS * cInnerIterations);

    algorithmFPType diff     = algorithmFPType(0);
    algorithmFPType diffPrev = algorithmFPType(0);

    size_t innerIteration = 0;
    size_t sameLocalDiff  = 0;

    TArray<algorithmFPType, cpu> buffer(nWS * 5 + nWS * nWS);
    DAAL_CHECK_MALLOC(buffer.get());

    TArray<algorithmFPType, cpu> deltaAlpha(nWS);
    DAAL_CHECK_MALLOC(deltaAlpha.get());

    SVMCachePtr<thunder, algorithmFPType, cpu> cachePtr;

    TArray<char, cpu> I(nVectors);
    DAAL_CHECK_MALLOC(I.get());

    if (cacheSize >= nVectors * nVectors * sizeof(algorithmFPType))
    {
        // TODO: support the simple cache for thunder method
        cachePtr = SVMCache<thunder, noCache, algorithmFPType, cpu>::create(cacheSize, nWS, nVectors, xTable, kernel, status);
    }
    else
    {
        cachePtr = SVMCache<thunder, noCache, algorithmFPType, cpu>::create(cacheSize, nWS, nVectors, xTable, kernel, status);
    }

    std::set<uint32_t> setws;

    size_t iter = 0;
    for (; iter < 1; iter++)
    {
        if (iter != 0)
        {
            // DAAL_CHECK_STATUS(status, workSet.copyLastToFirst());
            // DAAL_CHECK_STATUS(status, cachePtr->copyLastToFirst());
        }

        DAAL_CHECK_STATUS(status, workSet.select(y, alpha, grad, C));
        const uint32_t * wsIndices = workSet.getIndices();
        {
            for (int i = 0; i < PRINT_N; i++)
            {
                printf("%d, ", (int)wsIndices[i]);
            }
            printf(" ... ");
            for (int i = nWS - PRINT_N; i < nWS; i++)
            {
                printf("%d, ", (int)wsIndices[i]);
            }
            printf("\n");
            for (int i = 0; i < nWS; i++)
            {
                setws.insert(wsIndices[i]);
            }
            DAAL_ASSERT(setws.size() == nWS);
        }

        DAAL_CHECK_STATUS(status, cachePtr->compute(wsIndices));
        const algorithmFPType * kernelWS = cachePtr->getRowsBlock();

        SMOBlockSolver(y, grad, wsIndices, kernelWS, nVectors, nWS, C, eps, tau, buffer.get(), I.get(), alpha, deltaAlpha.get(), diff);
    }

    return status;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, ParameterType, cpu>::SMOBlockSolver(
    const algorithmFPType * y, const algorithmFPType * grad, const uint32_t * wsIndices, const algorithmFPType * kernelWS, const size_t nVectors,
    const size_t nWS, const double C, const double eps, const double tau, algorithmFPType * buffer, char * I, algorithmFPType * alpha,
    algorithmFPType * deltaAlpha, algorithmFPType & localDiff) const
{
    DAAL_ITTNOTIFY_SCOPED_TASK(SMOBlockSolver);
    services::Status status;

    const size_t innerMaxIterations(nWS * cInnerIterations);

    algorithmFPType * alphaLocal    = buffer + nWS * 0;
    algorithmFPType * yLocal        = buffer + nWS * 1;
    algorithmFPType * gradLocal     = buffer + nWS * 2;
    algorithmFPType * kdLocal       = buffer + nWS * 3;
    algorithmFPType * oldAlphaLocal = buffer + nWS * 4;
    algorithmFPType * kernelLocal   = buffer + nWS * 5;

    /* Gather data to local buffers */
    for (size_t i = 0; i < nWS; i++)
    {
        const size_t wsIndex = wsIndices[i];
        yLocal[i]            = y[wsIndex];
        gradLocal[i]         = grad[wsIndex];
        oldAlphaLocal[i]     = alpha[wsIndex];
        alphaLocal[i]        = alpha[wsIndex];
        kdLocal[i]           = kernelWS[i * nVectors + wsIndices[i]];

        char Ii = free;
        Ii |= isUpper(yLocal[i], alphaLocal[i], C) ? up : free;
        Ii |= isLower(yLocal[i], alphaLocal[i], C) ? low : free;
        I[i] = Ii;
        for (size_t j = 0; j < nWS; j++)
        {
            kernelLocal[i + j * nWS] = kernelWS[i * nVectors + wsIndices[j]];
        }
    }

    algorithmFPType delta    = algorithmFPType(0);
    algorithmFPType localEps = algorithmFPType(0);
    localDiff                = algorithmFPType(0);
    size_t Bi                = 0;
    size_t Bj                = 0;
    size_t iter              = 0;

    for (; iter < 1; iter++)
    {
        algorithmFPType GMin = MaxVal<algorithmFPType>::get();

        /* Find i index of the working set (Bi) */
        for (size_t i = 0; i < nWS; i++)
        {
            algorithmFPType objFunc = grad[i];
            if ((I[i] & up) && objFunc < GMin)
            {
                GMin = objFunc;
                Bi   = i;
            }
        }

        // for (size_t i = 0; i < nWS; i++)
        // {
        //     algorithmFPType objFunc = grad[i];
        //     if ((I[i] & low) && objFunc > maxGrad)
        //     {
        //         maxGrad = objFunc;
        //     }
        // }

        algorithmFPType GMax  = -MaxVal<algorithmFPType>::get();
        algorithmFPType GMax2 = MaxVal<algorithmFPType>::get();

        const algorithmFPType zero(0.0);
        const algorithmFPType two(2.0);

        for (size_t i = 0; i < nWS; i++)
        {
            algorithmFPType ygrad = gradLocal[i];
            if (!(I[i] & low)) continue;
            if (ygrad <= GMax2)
            {
                GMax2 = ygrad;
            }
            // if (ygrad >= GMax)
            // {
            //     continue;
            // }

            algorithmFPType b = ygrad - GMin;

            const algorithmFPType Kii   = kdLocal[i];
            const algorithmFPType KBiBi = kdLocal[Bi];
            const algorithmFPType KiBi  = kernelLocal[Bi * nWS + i];

            const algorithmFPType a = services::internal::max<cpu, algorithmFPType>(Kii + KBiBi - two * KiBi, tau);

            const algorithmFPType dt      = b / a;
            const algorithmFPType objFunc = -b * dt;
            if (objFunc < GMax)
            {
                GMax  = objFunc;
                Bj    = i;
                delta = dt;
            }
        }

        localDiff = GMax2 - GMin;

        if (iter == 0)
        {
            localEps = services::internal::max<cpu, algorithmFPType>(eps, localDiff * (algorithmFPType)1e-1);
        }
        if (localDiff < localEps)
        {
            break;
        }

        /* Update coefficients */
        const algorithmFPType alphaBiDelta = (yLocal[Bi] > 0.0f) ? C - alphaLocal[Bi] : alphaLocal[Bi];
        const algorithmFPType alphaBjDelta =
            services::internal::min<cpu, algorithmFPType>((yLocal[Bj] > 0.0f) ? alphaLocal[Bj] : C - alphaLocal[Bj], delta);
        delta = services::internal::min<cpu, algorithmFPType>(alphaBiDelta, alphaBjDelta);

        /* Update alpha */
        alphaLocal[Bi] += delta * yLocal[Bi];
        alphaLocal[Bj] -= delta * yLocal[Bj];

        /* Update up/low sets */
        char IBi = free;
        IBi |= isUpper(yLocal[Bi], alphaLocal[Bi], C) ? up : free;
        IBi |= isLower(yLocal[Bi], alphaLocal[Bi], C) ? low : free;
        I[Bi] = IBi;

        char IBj = free;
        IBj |= isUpper(yLocal[Bj], alphaLocal[Bj], C) ? up : free;
        IBj |= isLower(yLocal[Bj], alphaLocal[Bj], C) ? low : free;
        I[Bj] = IBj;

        /* Update gradient */
        for (size_t i = 0; i < nWS; i++)
        {
            const algorithmFPType KiBi = kernelLocal[Bi * nWS + i];
            const algorithmFPType KiBj = kernelLocal[Bj * nWS + i];
            gradLocal[i] += delta * (KiBi + KiBj);
        }
    }

    /* Compute diff and scatter to alpha vector */
    for (size_t i = 0; i < nWS; i++)
    {
        deltaAlpha[i]       = (alphaLocal[i] - oldAlphaLocal[i]) * yLocal[i];
        alpha[wsIndices[i]] = alphaLocal[i];
    }
    return status;
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

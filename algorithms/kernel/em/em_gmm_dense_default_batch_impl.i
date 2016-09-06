/* file: em_gmm_dense_default_batch_impl.i */
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

//++
//  Implementation of em algorithm
//--


#include "service_math.h"
#include "service_blas.h"
#include "service_lapack.h"
#include "service_stat.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_micro_table.h"
#include "em_gmm_dense_default_batch_kernel.h"
#include "threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::data_feature_utils::internal;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace internal
{

template <CpuType cpu, typename algorithmFPType>
void get2PIvalue(algorithmFPType *pi2);

template<typename algorithmFPType, CpuType cpu>
void performBlockEM(
    algorithmFPType *dataBlock,
    size_t nComponents, size_t nVectorsInCurrentBlock, size_t nFeatures, size_t blockSizeDeafult,
    algorithmFPType *localBuffer,
    algorithmFPType *resultBuffer,
    algorithmFPType *logAlpha,
    algorithmFPType *means,
    algorithmFPType *invSigma,
    algorithmFPType *logSqrtInvDetSigma,
    algorithmFPType *returnLogLikelyhood,
    Error *error);

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
        const size_t nr, NumericTable *r[], const Parameter *par)
{
    setVariables(a[0], nr, par);
    allocateMemory();
    if(!this->_errors->isEmpty()) {return;}

    iterCounter = 0;
    double diff = 2 * threshold + 1;
    double oldLogLikelyhood = 0;

    getInitValues(const_cast<NumericTable **>(a));
    if(!this->_errors->isEmpty()) {deallocate(); return;}

    blockSizeDeafult = 512;

    nBlocks = nVectors / blockSizeDeafult;
    nBlocks += (nBlocks * blockSizeDeafult != nVectors);

    if(nBlocks == 1)
    {
        blockSizeDeafult = nVectors;
    }

    algorithmFPType *logLikelyhoodLocalArray = (algorithmFPType *) daal_malloc(nBlocks * sizeof(algorithmFPType));
    if(!logLikelyhoodLocalArray) {this->_errors->add(ErrorMemoryAllocationFailed); return;}

    memorySizeForOneThread = blockSizeDeafult * nFeatures        + /* x_mu   */
                             blockSizeDeafult * nFeatures        + /* Ax_mu  */
                             blockSizeDeafult * nComponents      + /* p      */
                             blockSizeDeafult;                     /* rowSum */

    memorySizeForOneBlockResult = nComponents                         + /* wSums        */
                                  nComponents * nFeatures             + /* partialMeans */
                                  nComponents * nFeatures * nFeatures;  /* partialCP    */

    algorithmFPType *blockResultsMemory =
        (algorithmFPType *) daal_malloc(nBlocks * memorySizeForOneBlockResult * sizeof(algorithmFPType));
    if(!blockResultsMemory)
    {
        deallocate();
        daal_free(logLikelyhoodLocalArray);
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    daal::tls<algorithmFPType *> threadBuffer( [ = ]()-> algorithmFPType*
    {
        return (algorithmFPType *) daal_malloc(memorySizeForOneThread * sizeof(algorithmFPType));
    }
                                             );

    while (diff > threshold && iterCounter < maxIterations)
    {
        daal::tls<Error *> threadLocalError( [ = ]()-> Error* { return new Error(); } );

        computeSigmaValues(iterCounter);
        if(!this->_errors->isEmpty())
        {
            deallocate();
            daal_free(logLikelyhoodLocalArray);
            return;
        }

        logLikelyhood = 0;

        daal::threader_for( nBlocks, nBlocks, [ =, &threadBuffer, &threadLocalError ](size_t iBlock)
        {
            Error *localError = threadLocalError.local();
            size_t jn = blockSizeDeafult;
            if( iBlock == nBlocks - 1 )
            {
                jn = nVectors - iBlock * blockSizeDeafult;
            }
            size_t j0 = iBlock * blockSizeDeafult;

            size_t nVectorsInCurrentBlock = jn;

            algorithmFPType *localBuffer =  threadBuffer.local();
            if(!localBuffer) {localError->setId(ErrorMemoryAllocationFailed); return;}

            algorithmFPType *dataBlock;
            BlockMicroTable<algorithmFPType, readOnly, cpu> dataTable(a[0]);
            size_t read = dataTable.getBlockOfRows(j0, jn, &dataBlock);
            if(read != jn) {localError->setId(ErrorMemoryAllocationFailed); dataTable.release(); return;}

            performBlockEM<algorithmFPType, cpu>(
                dataBlock,
                nComponents, nVectorsInCurrentBlock, nFeatures, blockSizeDeafult,
                localBuffer,
                &blockResultsMemory[iBlock * memorySizeForOneBlockResult],
                logAlpha,
                means,
                sigma,
                logSqrtInvDetSigma,
                &logLikelyhoodLocalArray[iBlock],
                localError);
            if(localError->id() != NoErrorMessageFound) {dataTable.release(); return;}

            dataTable.release();
        }
                          );
        threadLocalError.reduce( [ = ](Error * e)-> void
        {
            if(e->id() != NoErrorMessageFound)
            {
                SharedPtr<Error> eCopy = SharedPtr<Error>(new Error(*e));
                this->_errors->add(eCopy);
            }
            delete e;
        });
        if(!this->_errors->isEmpty())
        {
            threadBuffer.reduce( [ = ](algorithmFPType * v)-> void {daal_free( v ); });
            daal_free(blockResultsMemory);
            daal_free(logLikelyhoodLocalArray);
            deallocate();
            return;
        }

        for(size_t iBlock = 0; iBlock < nBlocks; iBlock++)
        {
            logLikelyhood += logLikelyhoodLocalArray[iBlock];
        }
        logLikelyhood -= logLikelyhoodCorrection;

        stepM_merge(blockResultsMemory);
        if(!this->_errors->isEmpty())
        {
            threadBuffer.reduce( [ = ](algorithmFPType * v)-> void {daal_free( v ); });
            daal_free(blockResultsMemory);
            daal_free(logLikelyhoodLocalArray);
            deallocate();
            return;
        }

        if(iterCounter > 0)
        {
            diff = logLikelyhood - oldLogLikelyhood;
        }
        oldLogLikelyhood = logLikelyhood;

        iterCounter++;
    }
    threadBuffer.reduce( [ = ](algorithmFPType * v)-> void {daal_free( v ); });

    daal_free(blockResultsMemory);
    daal_free(logLikelyhoodLocalArray);

    writeResult(nr, r);
    if(!this->_errors->isEmpty()) {deallocate(); return;}

    deallocate();
}

template<typename algorithmFPType, CpuType cpu>
struct task
{
    algorithmFPType *x_mu;
    algorithmFPType *Ax_mu;
    algorithmFPType *p;
    algorithmFPType *w;
    algorithmFPType *wTransposed;
    algorithmFPType *rowSum;
    algorithmFPType *rowSumInv;
    algorithmFPType *wSums;
    algorithmFPType *partialMeans;
    algorithmFPType *partialCP;
    algorithmFPType *logAlpha;
    algorithmFPType *means;
    algorithmFPType *invSigma;
    algorithmFPType *logSqrtInvDetSigma;
    algorithmFPType partLogLikelyhood;
};

template<typename algorithmFPType, CpuType cpu>
ErrorID stepE(algorithmFPType *dataBlock,  const size_t nComponents, const size_t nVectorsInCurrentBlock, const size_t nFeatures,
              task<algorithmFPType, cpu> &t, Error *returnError)
{
    typedef Blas<algorithmFPType, cpu> blas;

    for(size_t k = 0; k < nComponents; k++)
    {
        algorithmFPType *curMean = &t.means[k * nFeatures];

        for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            for(size_t j = 0; j < nFeatures; j++)
            {
                t.x_mu[i * nFeatures + j] = dataBlock[i * nFeatures + j] - curMean[j];
            }
        }

        char side = 'L';
        char uplo = 'U';
        MKL_INT m = nFeatures;
        MKL_INT n = nVectorsInCurrentBlock;
        algorithmFPType alphaCoeff = 1.0;
        algorithmFPType betaCoeff = 0.0;
        MKL_INT lda = m;

        blas::xsymm(&side, &uplo, &m, &n, &alphaCoeff, &t.invSigma[k * nFeatures * nFeatures], &lda, t.x_mu, &m, &betaCoeff, t.Ax_mu, &lda);

        algorithmFPType addition = t.logAlpha[k] + t.logSqrtInvDetSigma[k];
        for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            t.p[k * nVectorsInCurrentBlock + i] = 0.0;
            for(size_t j = 0; j < nFeatures; j++)
            {
                t.p[k * nVectorsInCurrentBlock + i] += t.x_mu[i * nFeatures + j] * t.Ax_mu[i * nFeatures + j];
            }
            t.p[k * nVectorsInCurrentBlock + i] = addition + -0.5 * t.p[k * nVectorsInCurrentBlock + i];
        }
    }

    t.partLogLikelyhood = 0;
    algorithmFPType *maxInRow = t.rowSum;
    for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        maxInRow[i] = t.p[i];
    }

    for(size_t k = 1; k < nComponents; k++)
    {
        for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            if(t.p[k * nVectorsInCurrentBlock + i] > maxInRow[i])
            {
                maxInRow[i] = t.p[k * nVectorsInCurrentBlock + i];
            }
        }
    }

    for(size_t k = 0; k < nComponents; k++)
    {
        for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            t.p[k * nVectorsInCurrentBlock + i] -= maxInRow[i];
        }
    }

    for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        t.partLogLikelyhood += maxInRow[i];
        maxInRow[i] = 0; // same memory as t.rowSum, set to zero before computing row sum
    }

    daal::internal::Math<algorithmFPType,cpu>::vExp(nVectorsInCurrentBlock * nComponents, t.p, t.p);

    for(size_t k = 0; k < nComponents; k++)
    {
        for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            t.rowSum[i] += t.p[k * nVectorsInCurrentBlock + i];
        }
    }

    t.rowSumInv = t.rowSum;
    algorithmFPType one = 1.0;
    for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        t.rowSumInv[i] = one / t.rowSum[i];
    }


    for(size_t k = 0; k < nComponents; k++)
    {
        for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            t.p[k * nVectorsInCurrentBlock + i] = t.p[k * nVectorsInCurrentBlock + i] * t.rowSumInv[i];
        }
    }
    t.w = t.p;
    return (ErrorID)0;

}

template<typename algorithmFPType, CpuType cpu>
double computePartialLogLikelyhood(const size_t nVectorsInCurrentBlock, task<algorithmFPType, cpu> &t)
{
    algorithmFPType *logRowSumInv = t.rowSumInv;
    daal::internal::Math<algorithmFPType,cpu>::vLog(nVectorsInCurrentBlock, t.rowSumInv, logRowSumInv);

    algorithmFPType loglikPartial = t.partLogLikelyhood;
    for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        loglikPartial -= logRowSumInv[i];
    }
    return loglikPartial;
}

template<typename algorithmFPType, CpuType cpu>
void stepM_partial(algorithmFPType *dataBlock, const size_t nComponents, const size_t nFeatures, const size_t nVectorsInCurrentBlock,
                   task<algorithmFPType, cpu> &t, Error *returnError)
{
    __int64 matCompMethod = __DAAL_VSL_SS_METHOD_FAST;
    for(size_t k = 0; k < nComponents; k++)
    {
        t.wSums[k] = 0;

        int errcode = Statistics<algorithmFPType, cpu>::xxcp_weight(dataBlock, nFeatures, nVectorsInCurrentBlock,
                                               &t.w[k * nVectorsInCurrentBlock],
                                               &t.wSums[k],
                                               &t.partialMeans[k * nFeatures],
                                               &t.partialCP[k * nFeatures * nFeatures],
                                               matCompMethod);
        if(errcode)
        {
            returnError->setId(ErrorEMCovariance);
            returnError->addIntDetail(Component, k);
            return;
        }
    }

    return;
}

template<typename algorithmFPType, CpuType cpu>
void performBlockEM(
    algorithmFPType *dataBlock,
    size_t nComponents, size_t nVectorsInCurrentBlock, size_t nFeatures, size_t blockSizeDeafult,
    algorithmFPType *localBuffer,
    algorithmFPType *resultBuffer,
    algorithmFPType *logAlpha,
    algorithmFPType *means,
    algorithmFPType *invSigma,
    algorithmFPType *logSqrtInvDetSigma,
    algorithmFPType *returnLogLikelyhood,
    Error *returnError)
{
    task<algorithmFPType, cpu> t;
    t.x_mu                = localBuffer;
    t.Ax_mu               = &t.x_mu       [blockSizeDeafult * nFeatures  ];
    t.p                   = &t.Ax_mu      [blockSizeDeafult * nFeatures  ];
    t.rowSum              = &t.p          [blockSizeDeafult * nComponents];

    t.wSums        = resultBuffer;
    t.partialMeans = &t.wSums[nComponents];
    t.partialCP    = &t.partialMeans[nComponents * nFeatures];

    t.logAlpha           = logAlpha;
    t.means              = means;
    t.invSigma           = invSigma;
    t.logSqrtInvDetSigma = logSqrtInvDetSigma;

    stepE(dataBlock, nComponents, nVectorsInCurrentBlock, nFeatures, t, returnError);

    *returnLogLikelyhood = computePartialLogLikelyhood(nVectorsInCurrentBlock, t);

    stepM_partial(dataBlock, nComponents, nFeatures, nVectorsInCurrentBlock, t, returnError);
    if(returnError->id() != NoErrorMessageFound) {return;}

    return ;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::stepM_merge(algorithmFPType *localResultFull)
{
    for(size_t i = 0; i < nComponents; i++) { alpha[i] = 0; }
    for(size_t i = 0; i < nComponents * nFeatures; i++) { means[i] = 0; }
    for(size_t i = 0; i < nComponents * nFeatures * nFeatures; i++) { sigma[i] = 0; }

    algorithmFPType denominator;
    for(size_t k = 0; k < nComponents; k ++)
    {
        for(size_t iBlock = 0; iBlock < nBlocks; iBlock++)
        {
            algorithmFPType *localResult  = &localResultFull[iBlock * memorySizeForOneBlockResult];

            algorithmFPType *wSums        = localResult;
            algorithmFPType *partialMeans = &wSums       [nComponents];
            algorithmFPType *partialCP    = &partialMeans[nComponents * nFeatures];

            if(wSums[k] > MinVal<algorithmFPType, cpu>::get())
            {
                stepM_merge_inner(&sigma[k * nFeatures * nFeatures], &partialCP[k * nFeatures * nFeatures],
                                  &means[k * nFeatures], &partialMeans[k * nFeatures],
                                  alpha[k], wSums[k]);
            }
        }
        if(alpha[k] < MinVal<algorithmFPType, cpu>::get())
        {
            this->_errors->add(ErrorEMEmptyComponent).addIntDetail(Component, k);
            return;
        }
        denominator = 1.0 / alpha[k];
        alpha[k] /= nVectors;
        for(size_t i = 0; i < nFeatures; i++)
        {
            for(size_t j = 0; j < i; j++)
            {
                sigma[k * nFeatures * nFeatures + i * nFeatures + j] *= denominator;
                sigma[k * nFeatures * nFeatures + j * nFeatures + i] = sigma[k * nFeatures * nFeatures + i * nFeatures + j];
            }
            sigma[k * nFeatures * nFeatures + i * nFeatures + i] *= denominator;
        }
    }
}


template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::allocateMemory()
{
    buffer = (algorithmFPType *)daal_malloc((
                 nComponents                                   + /* alpha           */
                 nComponents * nFeatures                       + /* means           */
                 nComponents * nFeatures * nFeatures           + /* sigma           */
                 nComponents                                     /* logSqrtInvDetSigma */
             ) * sizeof(algorithmFPType));

    if(!buffer) {this->_errors->add(ErrorMemoryAllocationFailed); return;}

    alpha           = buffer;
    means           = &alpha[nComponents];
    sigma           = &means[nComponents * nFeatures];
    logSqrtInvDetSigma = &sigma[nComponents * nFeatures * nFeatures];
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::deallocate()
{
    daal_free(buffer);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::setVariables(const NumericTable *dataTable, const size_t nr, const Parameter *par)
{
    nFeatures = dataTable->getNumberOfColumns();
    nVectors = dataTable->getNumberOfRows();
    nBlocks = 1;

    algorithmFPType PI2;
    get2PIvalue<cpu, algorithmFPType>(&PI2);
    logLikelyhoodCorrection = 0.5 * nVectors * nFeatures * daal::internal::Math<algorithmFPType,cpu>::sLog(PI2);

    threshold = par->accuracyThreshold;
    maxIterations = par->maxIterations;
    nComponents = par->nComponents;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::stepM_merge_inner(
    algorithmFPType *cp_n, algorithmFPType *cp_m,
    algorithmFPType *mean_n, algorithmFPType *mean_m,
    algorithmFPType &W_n, algorithmFPType &W_m)
{
    algorithmFPType *Cov_n = cp_n;
    algorithmFPType *Cov_m = cp_m;
    algorithmFPType one_Wnm;

    if(W_n == (algorithmFPType)0.0)
    {
        one_Wnm = 0.0;
    }
    else
    {
        one_Wnm = (W_n * W_m) / (W_n + W_m);
    }

    for(size_t i = 0; i < nFeatures; i++)
    {
PRAGMA_VECTOR_UNALIGNED
        for(size_t j = 0; j <= i; j++)
        {
            Cov_n[i * nFeatures + j] =
                Cov_n[i * nFeatures + j] + Cov_m[i * nFeatures + j] +
                one_Wnm * (
                    + mean_n[i] * mean_n[j] + mean_m[i] * mean_m[j]
                    - mean_n[i] * mean_m[j] - mean_m[i] * mean_n[j] );
        }
    }

    algorithmFPType *mom_n = mean_n;
    algorithmFPType *mom_m = mean_m;
    one_Wnm = 1.0 / (W_n + W_m);

PRAGMA_VECTOR_UNALIGNED
    for(size_t j = 0; j < nFeatures; j++)
    {
        mom_n[j] = (W_n * mom_n[j] + W_m * mom_m[j]) * one_Wnm;
    }
    W_n += W_m;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::regularizeCovarianceMatrix(algorithmFPType *cov, Error *error)
{
    char jobz  = 'N';
    char uplo  = 'L';

    MKL_INT lwork = 2 * nFeatures  + 1;
    MKL_INT liwork = 1;
    MKL_INT info = 0;

    algorithmFPType *work = (algorithmFPType *)daal_malloc(lwork  * sizeof(algorithmFPType));
    MKL_INT *iwork = (MKL_INT *)daal_malloc(liwork * sizeof(MKL_INT));
    algorithmFPType *eigenvalues = (algorithmFPType *)daal_malloc(nFeatures * sizeof(algorithmFPType));
    algorithmFPType *diagValues = (algorithmFPType *)daal_malloc(nFeatures * sizeof(algorithmFPType));
    if(work == 0 || iwork == 0 || eigenvalues == 0 || diagValues == 0)
    {
        daal_free(work);
        daal_free(iwork);
        daal_free(eigenvalues);
        daal_free(diagValues);
        error->setId(ErrorMemoryAllocationFailed); return;
    }

    for(size_t i = 0; i < nFeatures; i++)
    {
        diagValues[i] = cov[i * nFeatures + i];
    }

    Lapack<algorithmFPType, cpu>::xsyevd(&jobz, &uplo, (MKL_INT *)(&nFeatures), cov, (MKL_INT *)(&nFeatures), eigenvalues,
                                         work, &lwork, iwork, &liwork, &info);
    if (info != 0)
    {
        daal_free(work);
        daal_free(iwork);
        daal_free(eigenvalues);
        if(info < 0) {error->setId(ErrorIncorrectInternalFunctionParameter); return;}
        error->setId(ErrorEMIllConditionedCovarianceMatrix); return;
    }

    daal_free(iwork);
    daal_free(work);

    for(size_t i = 0 ; i < nFeatures ; i++ )
    {
        cov[i * nFeatures + i] = diagValues[i];
        for(int j = i + 1; j < nFeatures; j++)
        {
            cov[i * nFeatures + j] = cov[j * nFeatures + i];
        }
    }

    if(eigenvalues[0] <= 0.0)
    {
        size_t i = 0;
        while(i < nFeatures && eigenvalues[i] < 0)
        {
            i++;
        }
        if ( i == nFeatures )
        {
            daal_free(diagValues);
            daal_free(eigenvalues);
            error->setId(ErrorEMNegativeDefinedCovarianceMartix);
            return;
        }
    }

    algorithmFPType EIGENVALUE_THRESHOLD = 1000 * MinVal<algorithmFPType, cpu>::get();
    size_t i = 0;
    for( i = 0; i < nFeatures; i++ )
    {
        if( eigenvalues[i] > EIGENVALUE_THRESHOLD )
        {
            break;
        }
    }
    if(i == nFeatures)
    {
        daal_free(diagValues);
        daal_free(eigenvalues);
        error->setId(ErrorEMIllConditionedCovarianceMatrix);
        return;
    }

    //get maximum
    algorithmFPType C001 = 0.01;
    algorithmFPType cur_eigenvalue;
    algorithmFPType a = eigenvalues[i] * C001;
    algorithmFPType b = -eigenvalues[0] * (1 + C001);
    cur_eigenvalue = (a > b) ? a : b;
    for(size_t j = 0 ; j < nFeatures ; j++ )
    {
        cov[j * nFeatures + j] += cur_eigenvalue;
    }

    daal_free(diagValues);
    daal_free(eigenvalues);
    return;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::computeSigmaValues(size_t iteration)
{
    typedef Lapack<algorithmFPType, cpu> lapack;

    algorithmFPType *invSigma = sigma; //one place for both arrays

    algorithmFPType *sqrtInvDetSigma = logSqrtInvDetSigma;

    daal::tls<Error *> threadLocalError( [ = ]()-> Error* { return new Error(); } );
    daal::tls<algorithmFPType *> sigma_buff( [ = ]()-> algorithmFPType*
    {
        return (algorithmFPType *) daal_malloc(nFeatures *nFeatures * sizeof(algorithmFPType));
    } );


    daal::threader_for( nComponents, nComponents, [ =, &sigma_buff, &threadLocalError ](size_t iComp)
    {
        Error *localError = threadLocalError.local();
        char uplo = 'U';
        MKL_INT info;
        MKL_INT lda = nFeatures;
        algorithmFPType *pInvSigma = NULL;
        pInvSigma = &invSigma[iComp * nFeatures * nFeatures];
        algorithmFPType *pSigma = pInvSigma;

        algorithmFPType *sigmaTmpBuff = sigma_buff.local();
        for(size_t i = 0; i < nFeatures * nFeatures; i++)
        {
            sigmaTmpBuff[i] = pSigma[i];
        }
        lapack::xpotrf(&uplo, &nFeatures, pInvSigma, &lda, &info);
        if (info != 0)
        {
            if(info < 0)
            {
                localError->setId(ErrorIncorrectInternalFunctionParameter);
                localError->addIntDetail(Component, iComp);
                return;
            }
            for(size_t i = 0; i < nFeatures * nFeatures; i++)
            {
                pSigma[i] = sigmaTmpBuff[i];
            }
            regularizeCovarianceMatrix(pSigma, localError);
            if(localError->id() != NoErrorMessageFound)
            {
                localError->addIntDetail(Component, iComp);
                return;
            }

            lapack::xpotrf(&uplo, &nFeatures, pInvSigma, &lda, &info);
            if(info != 0)
            {
                if(info < 0) { localError->setId(ErrorIncorrectInternalFunctionParameter); }
                else { localError->setId(ErrorEMIllConditionedCovarianceMatrix); }
                localError->addIntDetail(Component, iComp);
                localError->addIntDetail(Minor, info);
                return;
            }
        }
        algorithmFPType sqrtDetSigma = 1;
        for(size_t j = 0; j < nFeatures; j++)
        {
            sqrtDetSigma *= pInvSigma[j * nFeatures + j];
        }
        sqrtInvDetSigma[iComp] = 1.0 / sqrtDetSigma;

        lapack::xpotri(&uplo, &nFeatures, pInvSigma, &lda, &info);
        if (info != 0)
        {
            if(info < 0) {localError->setId(ErrorIncorrectInternalFunctionParameter);}
            else {localError->setId(ErrorEMMatrixInverse); }
            localError->addIntDetail(Component, iComp);
            localError->addIntDetail(Minor, info);
            return;
        }
    }
                      );

    threadLocalError.reduce( [ = ](Error * e)-> void
    {
        if(e->id() != NoErrorMessageFound)
        {
            SharedPtr<Error> eCopy = SharedPtr<Error>(new Error(*e));
            eCopy->addIntDetail(Iteration, iteration + 1);
            this->_errors->add(eCopy);
        }
        delete e;
    });
    sigma_buff.reduce( [ = ](algorithmFPType * v)-> void {daal_free( v ); });

    daal::internal::Math<algorithmFPType,cpu>::vLog(nComponents, sqrtInvDetSigma, logSqrtInvDetSigma);

    logAlpha = alpha;
    daal::internal::Math<algorithmFPType,cpu>::vLog(nComponents, alpha, logAlpha);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::writeResult(const size_t nr, NumericTable *r[])
{

    writeArrayToNumericTable(r[0], alpha, nComponents, 1);
    if(!this->_errors->isEmpty()) { return; }

    writeArrayToNumericTable(r[1], means, nFeatures, nComponents);
    if(!this->_errors->isEmpty()) { return; }

    writeArrayToNumericTable(r[2], &logLikelyhood, 1, 1);
    if(!this->_errors->isEmpty()) { return; }

    algorithmFPType iterCounterFPtype = iterCounter;
    writeArrayToNumericTable(r[3], &iterCounterFPtype, 1, 1);
    if(!this->_errors->isEmpty()) { return; }

    for (size_t i = 0; i < nComponents; i++)
    {
        writeArrayToNumericTable(r[4 + i], &sigma[i * nFeatures * nFeatures], nFeatures, nFeatures);
        if(!this->_errors->isEmpty()) { return; }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::writeArrayToNumericTable(NumericTable *nt, algorithmFPType *array,
        size_t nColsArr, size_t nRowsArr)
{
    size_t nCols = nt->getNumberOfColumns();
    size_t nRows = nt->getNumberOfRows();

    algorithmFPType *d;
    BlockMicroTable<algorithmFPType, writeOnly, cpu> dataTable(nt);
    size_t read = dataTable.getBlockOfRows(0, nRows, &d);
    if(read != nRows)
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        dataTable.release();
        return;
    }

    for(size_t i = 0; i < nRows * nCols; i++)
    {
        d[i] = array[i];
    }

    dataTable.release();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::getArrayFromNumericTable(const NumericTable *ntConst,
        algorithmFPType *array, size_t nColsArr, size_t nRowsArr)
{
    NumericTable *nt = const_cast<NumericTable *>(ntConst);
    size_t nCols = nt->getNumberOfColumns();
    size_t nRows = nt->getNumberOfRows();

    algorithmFPType *d;
    BlockMicroTable<algorithmFPType, readOnly, cpu> dataTable(nt);
    size_t read = dataTable.getBlockOfRows(0, nRows, &d);
    if(read != nRows)
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        dataTable.release();
        return;
    }

    for(size_t i = 0; i < nRows * nCols; i++)
    {
        array[i] = d[i];
    }

    dataTable.release();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::getInitValues(NumericTable **a)
{
    getArrayFromNumericTable(a[1], alpha, nComponents, 1);
    if(!this->_errors->isEmpty()) { return; }

    getArrayFromNumericTable(a[2], means, nFeatures, nComponents);
    if(!this->_errors->isEmpty()) { return; }

    for (size_t i = 0; i < nComponents; i++)
    {
        getArrayFromNumericTable(a[3 + i], &sigma[i * nFeatures * nFeatures], nFeatures, nFeatures);
        if(!this->_errors->isEmpty()) { return; }
    }
}

template <CpuType cpu, typename algorithmFPType>
struct get2PIvalueImpl
{
    static void get(algorithmFPType *pi2)
    {
        *pi2 = (algorithmFPType) (2.0 * 3.1415926535897932384626433);
    };
};

template <CpuType cpu>
struct get2PIvalueImpl<cpu, float>
{
    static void get(float *pi2)
    {
        unsigned int pi2_hex = 0x40C90FDB;
        float *pi2_decimal = reinterpret_cast<float *>(&pi2_hex);
        *pi2 = (float) * pi2_decimal;
    };
};

template <CpuType cpu>
struct get2PIvalueImpl<cpu, double>
{
    static void get(double *pi2)
    {
        DAAL_UINT64 pi2_hex = 0x401921FB54442D18;
        double *pi2_decimal = reinterpret_cast<double *>(&pi2_hex);
        *pi2 = (double) * pi2_decimal;
    };
};

template <CpuType cpu, typename algorithmFPType>
void get2PIvalue(algorithmFPType *pi2) { get2PIvalueImpl<cpu, algorithmFPType>::get(pi2); }

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

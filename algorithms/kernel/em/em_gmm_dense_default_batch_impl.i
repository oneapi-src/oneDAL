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
#include "service_lapack.h"
#include "service_stat.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "em_gmm_dense_default_batch_kernel.h"
#include "em_gmm_dense_default_batch_task.h"
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

/**
 * Function computes expectation-maximization algorithm for Gaussian Mixture Model.
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::compute(
    NumericTable *dataTable,
    NumericTable *initialWeights,
    NumericTable *initialMeans,
    NumericTable **initialCovariances,
    NumericTable *resultWeights,
    NumericTable *resultMeans,
    NumericTable **resultCovariances,
    NumericTable *resultNIterations,
    NumericTable *resultGoalFunction,
    const Parameter *par)
{
    setVariables(dataTable, par);

    WriteRows<algorithmFPType, cpu, NumericTable> weightsBD(resultWeights, 0, 1);
    alpha = weightsBD.get();
    DAAL_CHECK(alpha, ErrorMemoryAllocationFailed);

    WriteRows<algorithmFPType, cpu, NumericTable> meansBD(resultMeans, 0, nFeatures);
    means = meansBD.get();
    DAAL_CHECK(means, ErrorMemoryAllocationFailed);

    TNewSmartPtr<WriteRows<algorithmFPType, cpu, NumericTable>, cpu> covsPtr(nComponents);
    WriteRows<algorithmFPType, cpu, NumericTable> *covsNt = covsPtr.get();

    GmmModel<algorithmFPType, cpu> *covs  = nullptr;
    if(par->covarianceStorage == diagonal)
    {
        covs = new GmmModelDiag<algorithmFPType, cpu>(nFeatures, nComponents);
    }
    else
    {
        covs = new GmmModelFull<algorithmFPType, cpu>(nFeatures, nComponents);
    }
    SharedPtr<GmmModel<algorithmFPType, cpu> > covsShPtr(covs);
    DAAL_CHECK(covs, ErrorMemoryAllocationFailed);

    algorithmFPType **sigma = covs->getSigma();
    for(size_t i = 0; i < nComponents; i++)
    {
        covsNt[i].set(resultCovariances[i], 0, covs->getNumberOfRowsInCov());
        sigma[i] = covsNt[i].get();
        DAAL_CHECK(sigma[i], ErrorMemoryAllocationFailed);
    }

    WriteRows<int, cpu, NumericTable> nIterationsBD(resultNIterations, 0, 1);
    int *iterCounterArray = nIterationsBD.get();
    DAAL_CHECK(iterCounterArray, ErrorMemoryAllocationFailed);
    int &iterCounter = iterCounterArray[0];
    iterCounter = 0;

    WriteRows<algorithmFPType, cpu, NumericTable> goalFunctionBD(resultGoalFunction, 0, 1);
    algorithmFPType *logLikelyhoodArray = goalFunctionBD.get();
    DAAL_CHECK(logLikelyhoodArray, ErrorMemoryAllocationFailed);
    algorithmFPType &logLikelyhood = logLikelyhoodArray[0];

    double diff = 2 * threshold + 1;
    double oldLogLikelyhood = 0;

    getInitValues(initialWeights, initialMeans, initialCovariances, covs);
    if(!this->_errors->isEmpty()) {return;}

    blockSizeDefault = 512;

    nBlocks = nVectors / blockSizeDefault;
    nBlocks += (nBlocks * blockSizeDefault != nVectors);
    covs->setCovRegularizer(par->regularizationFactor);

    if(nBlocks == 1)
    {
        blockSizeDefault = nVectors;
    }

    TSmartPtr<algorithmFPType, cpu> logLikelyhoodLocalArrayPtr(nBlocks);
    algorithmFPType *logLikelyhoodLocalArray = logLikelyhoodLocalArrayPtr.get();
    logLikelyhoodLocalArray = logLikelyhoodLocalArrayPtr.get();
    DAAL_CHECK(logLikelyhoodLocalArray, ErrorMemoryAllocationFailed);

    daal::tls<Task<algorithmFPType, cpu> *> threadBuffer( [ = ]()-> Task<algorithmFPType, cpu> *
    {
        return new Task<algorithmFPType, cpu>(dataTable, blockSizeDefault, nFeatures, nComponents, alpha, means, covs);
    }
                                                        );

    daal::tls<Error *> threadLocalError( [ = ]()-> Error* { return new Error(); } );
    while (diff > threshold && iterCounter < maxIterations)
    {
        covs->computeSigmaInverse(iterCounter, this->_errors.get());
        if(!this->_errors->isEmpty()) { return; }

        algorithmFPType *sqrtInvDetSigma = covs->getLogSqrtInvDetSigma();
        Math<algorithmFPType, cpu>::vLog(nComponents, sqrtInvDetSigma, covs->getLogSqrtInvDetSigma());

        logAlpha = alpha;
        Math<algorithmFPType, cpu>::vLog(nComponents, alpha, logAlpha);

        logLikelyhood = 0;
        daal::threader_for( nBlocks, nBlocks, [ =, &threadBuffer, &threadLocalError](size_t iBlock)
        {
            size_t j0 = iBlock * blockSizeDefault;
            size_t nVectorsInCurrentBlock = blockSizeDefault;
            if( iBlock == nBlocks - 1 )
            {
                nVectorsInCurrentBlock = nVectors - iBlock * blockSizeDefault;
            }

            Error *localError = threadLocalError.local();
            Task<algorithmFPType, cpu> *tPtr = threadBuffer.local();
            if(!tPtr) {localError->setId(ErrorMemoryAllocationFailed); return;}
            Task<algorithmFPType, cpu> &t = *tPtr;

            t.next(j0, nVectorsInCurrentBlock, localError);
            if(localError->id() != NoErrorMessageFound) {return;}

            stepE(nVectorsInCurrentBlock, t);

            logLikelyhoodLocalArray[iBlock] = computePartialLogLikelyhood(nVectorsInCurrentBlock, t);

            stepM_partial(nVectorsInCurrentBlock, t, localError);
            if(localError->id() != NoErrorMessageFound) {return;}
        }
                          );
        threadLocalError.reduce( [ = ](Error * e)-> void
        {
            if(e->id() != NoErrorMessageFound) { this->_errors->add(SharedPtr<Error>(new Error(*e)));}
        });
        if(!this->_errors->isEmpty()) {break;}

        for(size_t i = 0; i < nComponents; i++) {alpha[i] = 0;}
        for(size_t i = 0; i < nComponents * nFeatures; i++) {means[i] = 0;}
        covs->setToZero();
        size_t nElementsOnOneCov = covs->getOneCovSize();
        threadBuffer.reduce( [ = ]( Task<algorithmFPType, cpu> *e)-> void
        {
            for(size_t k = 0; k < nComponents; k++)
            {
                if(e->mergedWSums[k] > MinVal<algorithmFPType, cpu>::get())
                {
                    stepM_mergePartialSums(
                        sigma[k], &e->mergedPartialCP[k * nElementsOnOneCov],
                        &means[k * nFeatures], &e->mergedPartialMeans[k * nFeatures],
                        alpha[k], e->mergedWSums[k],
                        nFeatures, covs
                    );
                }
            }
            e->setMergedToZero();
        });

        for(size_t iBlock = 0; iBlock < nBlocks; iBlock++)
        {
            logLikelyhood += logLikelyhoodLocalArray[iBlock];
        }
        logLikelyhood -= logLikelyhoodCorrection;

        stepM_merge(iterCounter, covs);
        if(!this->_errors->isEmpty()) {break;}

        if(iterCounter > 0)
        {
            diff = logLikelyhood - oldLogLikelyhood;
        }
        oldLogLikelyhood = logLikelyhood;

        iterCounter++;
    }
    threadLocalError.reduce( [ = ](Error * e)-> void { delete e; });
    threadBuffer.reduce( [ = ](Task<algorithmFPType, cpu> *v)-> void {delete( v ); });
}

/**
 * Function computes t.w values that stores weight of each data point belongs to each cluster.
 * t.s is computed by numeric stable log-sum-exp trick.
 */
template<typename algorithmFPType, CpuType cpu>
void stepE(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> &t)
{
    const size_t nComponents = t.nComponents;
    const size_t nFeatures = t.nFeatures;

    for(size_t k = 0; k < nComponents; k++)
    {
        algorithmFPType *curMean = &t.means[k * nFeatures];

        for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            for(size_t j = 0; j < nFeatures; j++)
            {
                t.x_mu[i * nFeatures + j] = t.dataBlock[i * nFeatures + j] - curMean[j];
            }
        }

        t.covs->multiplyByInverseMatrix(nVectorsInCurrentBlock, k, t.x_mu, t.Ax_mu);

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

    Math<algorithmFPType, cpu>::vExp(nVectorsInCurrentBlock * nComponents, t.p, t.p);

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
}

/**
 * Function computes log likelihood value of for data block. This value contains only logarithm of inversed
 * weighted sum of probabilities of belonging the point to each cluster. To get full log likelihood value,
 * all these values will be summed.
 */

template<typename algorithmFPType, CpuType cpu>
double computePartialLogLikelyhood(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> &t)
{
    algorithmFPType *logRowSumInv = t.rowSumInv;
    Math<algorithmFPType, cpu>::vLog(nVectorsInCurrentBlock, t.rowSumInv, logRowSumInv);

    algorithmFPType loglikPartial = t.partLogLikelyhood;
    for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
    {
        loglikPartial -= logRowSumInv[i];
    }
    return loglikPartial;
}

/**
 * Function computes from block of data points and weights:
 * 1) sum of weights
 * 2) weighted mean
 * 3) weighted cross product
 * After this computation for current block the results are merged to thread local values
 */
template<typename algorithmFPType, CpuType cpu>
void stepM_partial(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> &t, Error *returnError)
{

    const size_t nFeatures = t.nFeatures;
    const size_t nElementsOnOneCov = t.covs->getOneCovSize();

    for(size_t k = 0; k < t.nComponents; k++)
    {
        t.wSums[k] = 0;

        int errcode = t.covs->computeThreadPartialResults(const_cast<algorithmFPType *>(t.dataBlock), &t.w[k * nVectorsInCurrentBlock],
                      t.nFeatures, nVectorsInCurrentBlock,
                      &t.wSums[k],
                      &t.partialMeans[k * nFeatures],
                      &t.partialCP[k * nElementsOnOneCov]);
        if(errcode)
        {
            returnError->setId(ErrorEMCovariance);
            returnError->addIntDetail(Component, k);
            return;
        }

        if(t.wSums[k] > MinVal<algorithmFPType, cpu>::get())
        {
            stepM_mergePartialSums(
                &t.mergedPartialCP[k * nElementsOnOneCov], &t.partialCP[k * nElementsOnOneCov],
                &t.mergedPartialMeans[k * nFeatures], &t.partialMeans[k * nFeatures],
                t.mergedWSums[k], t.wSums[k],
                nFeatures, t.covs
            );
        }
    }

    return;
}

/**
 * Function scales merged values of to get result
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::stepM_merge(size_t iteration, GmmModel<algorithmFPType, cpu> *covs)
{
    algorithmFPType denominator;
    for(size_t k = 0; k < nComponents; k ++)
    {
        if(alpha[k] < MinVal<algorithmFPType, cpu>::get())
        {
            this->_errors->add(ErrorEMEmptyComponent).addIntDetail(Component, k).addIntDetail(Iteration, iteration + 1);
            return;
        }
        covs->finalize(k, alpha[k]);
        alpha[k] /= nVectors;
    }
}

/**
 * Sets constants and values
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::setVariables(const NumericTable *dataTable, const Parameter *par)
{
    nFeatures = dataTable->getNumberOfColumns();
    nVectors = dataTable->getNumberOfRows();
    nBlocks = 1;

    algorithmFPType pi = 3.1415926535897932384626433;
    logLikelyhoodCorrection = 0.5 * nVectors * nFeatures * Math<algorithmFPType, cpu>::sLog(2 * pi);

    threshold = par->accuracyThreshold;
    maxIterations = par->maxIterations;
    nComponents = par->nComponents;
}

/**
 * Function merges partial cross products, sum of weights and means
 */
template<typename algorithmFPType, CpuType cpu>
void stepM_mergePartialSums(
    algorithmFPType *cp_n, algorithmFPType *cp_m,
    algorithmFPType *mean_n, algorithmFPType *mean_m,
    algorithmFPType &w_n, algorithmFPType &w_m,
    size_t nFeatures, GmmModel<algorithmFPType, cpu> *covs)
{
    covs->stepM_mergeCovs(
        cp_n, cp_m,
        mean_n, mean_m,
        w_n, w_m,
        nFeatures
    );

    algorithmFPType *mom_n = mean_n;
    algorithmFPType *mom_m = mean_m;
    algorithmFPType one_Wnm = 1.0 / (w_n + w_m);

    PRAGMA_VECTOR_UNALIGNED
    for(size_t j = 0; j < nFeatures; j++)
    {
        mom_n[j] = (w_n * mom_n[j] + w_m * mom_m[j]) * one_Wnm;
    }
    w_n += w_m;
}
/**
 * Function merges partial cross products, sum of weights and means
 */
template<typename algorithmFPType, CpuType cpu>
void GmmModelFull<algorithmFPType, cpu>::stepM_mergeCovs(
    algorithmFPType *cp_n, algorithmFPType *cp_m,
    algorithmFPType *mean_n, algorithmFPType *mean_m,
    algorithmFPType &w_n, algorithmFPType &w_m,
    size_t nFeatures)
{
    algorithmFPType one_Wnm = (w_n == (algorithmFPType)0.0) ? 0.0 : (w_n * w_m) / (w_n + w_m);

    for(size_t i = 0; i < nFeatures; i++)
    {
        PRAGMA_VECTOR_UNALIGNED
        for(size_t j = 0; j <= i; j++)
        {
            cp_n[i * nFeatures + j] =
                cp_n[i * nFeatures + j] + cp_m[i * nFeatures + j] +
                one_Wnm * (
                    + mean_n[i] * mean_n[j] + mean_m[i] * mean_m[j]
                    - mean_n[i] * mean_m[j] - mean_m[i] * mean_n[j] );
        }
    }
}

/**
 * Function merges partial cross products, sum of weights and means
 */
template<typename algorithmFPType, CpuType cpu>
void GmmModelDiag<algorithmFPType, cpu>::stepM_mergeCovs(
    algorithmFPType *cp_n, algorithmFPType *cp_m,
    algorithmFPType *mean_n, algorithmFPType *mean_m,
    algorithmFPType &w_n, algorithmFPType &w_m,
    size_t nFeatures)
{
    algorithmFPType one_Wnm = (w_n == (algorithmFPType)0.0) ? 0.0 : (w_n * w_m) / (w_n + w_m);

    for(size_t i = 0; i < nFeatures; i++)
    {
        cp_n[i] =
            cp_n[i] + cp_m[i] +
            one_Wnm * (
                + mean_n[i] * mean_n[i] + mean_m[i] * mean_m[i]
                - mean_n[i] * mean_m[i] - mean_m[i] * mean_n[i] );
    }
}


/**
 * Function for regularization ill-conditioned covariance matrices. It adds value based on eigen values of matrix to diagonal.
 */
template<typename algorithmFPType, CpuType cpu>
void GmmModelFull<algorithmFPType, cpu>::regularizeCovarianceMatrix(algorithmFPType *cov, Error *error)
{
    char jobz  = 'N';
    char uplo  = 'L';

    DAAL_INT lwork = 2 * nFeatures  + 1;
    DAAL_INT liwork = 1;
    DAAL_INT info = 0;

    TSmartPtr<algorithmFPType, cpu> eigenvaluesPtr(nFeatures);
    TSmartPtr<algorithmFPType, cpu> diagValuesPtr(nFeatures);
    algorithmFPType *eigenvalues = eigenvaluesPtr.get();
    algorithmFPType *diagValues = diagValuesPtr.get();

    TSmartPtr<algorithmFPType, cpu> workPtr(lwork);
    TSmartPtr<DAAL_INT, cpu> iworkPtr(liwork);
    algorithmFPType *work = workPtr.get();
    DAAL_INT *iwork = iworkPtr.get();
    if(work == 0 || iwork == 0 || eigenvalues == 0 || diagValues == 0)
    {
        error->setId(ErrorMemoryAllocationFailed); return;
    }

    for(size_t i = 0; i < nFeatures; i++)
    {
        diagValues[i] = cov[i * nFeatures + i];
    }

    DAAL_INT p = nFeatures;
    Lapack<algorithmFPType, cpu>::xxsyevd(&jobz, &uplo, &p, cov, &p, eigenvalues, work, &lwork, iwork, &liwork, &info);
    if (info != 0)
    {
        if(info < 0) {error->setId(ErrorIncorrectInternalFunctionParameter); return;}
        error->setId(ErrorEMIllConditionedCovarianceMatrix); return;
    }

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
            error->setId(ErrorEMNegativeDefinedCovarianceMartix);
            return;
        }
    }

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
        error->setId(ErrorEMIllConditionedCovarianceMatrix);
        return;
    }

    //get maximum
    algorithmFPType a = eigenvalues[i] * covRegularizer;
    algorithmFPType b = -eigenvalues[0] * (1.0 + covRegularizer);
    algorithmFPType cur_eigenvalue = (a > b) ? a : b;
    for(size_t j = 0 ; j < nFeatures ; j++ )
    {
        cov[j * nFeatures + j] += cur_eigenvalue;
    }

    return;
}

/**
 * Ties to inverse covariance matrices. In case of ill-conditioned matrix try to regularize.
 */
template<typename algorithmFPType, CpuType cpu>
void GmmModelFull<algorithmFPType, cpu>::computeSigmaInverse(size_t iteration, KernelErrorCollection *errorCollection)
{
    typedef Lapack<algorithmFPType, cpu> lapack;

    algorithmFPType **invSigma = sigma; //one place for both arrays

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
        DAAL_INT info;
        DAAL_INT lda = nFeatures;
        DAAL_INT nFeaturesLong = nFeatures;
        algorithmFPType *pInvSigma = NULL;
        pInvSigma = invSigma[iComp];
        algorithmFPType *pSigma = pInvSigma;

        algorithmFPType *sigmaTmpBuff = sigma_buff.local();
        for(size_t i = 0; i < nFeatures * nFeatures; i++)
        {
            sigmaTmpBuff[i] = pSigma[i];
        }
        lapack::xxpotrf(&uplo, &nFeaturesLong, pInvSigma, &lda, &info);
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

            lapack::xxpotrf(&uplo, &nFeaturesLong, pInvSigma, &lda, &info);
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

        lapack::xxpotri(&uplo, &nFeaturesLong, pInvSigma, &lda, &info);
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
            errorCollection->add(eCopy);
        }
        delete e;
    });
    sigma_buff.reduce( [ = ](algorithmFPType * v)-> void {daal_free( v ); });
}

/**
 * Read initial values and copy them to work arrays.
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void EMKernel<algorithmFPType, method, cpu>::getInitValues(NumericTable *initialWeights, NumericTable *initialMeans, NumericTable **initialCovariances, GmmModel<algorithmFPType, cpu> *covs)
{
    ReadRows<algorithmFPType, cpu, NumericTable> bd;

    bd.set(initialWeights, 0, 1);
    const algorithmFPType *initialWeightsArray = bd.get();
    DAAL_CHECK(initialWeightsArray, ErrorMemoryAllocationFailed);
    if(initialWeightsArray != alpha)
    {
        size_t nCopy = nComponents * sizeof(algorithmFPType);
        daal_memcpy_s(alpha, nCopy, initialWeightsArray, nCopy);
    }

    bd.set(initialMeans, 0, nComponents);
    const algorithmFPType *initialMeansArray = bd.get();
    DAAL_CHECK(initialMeansArray, ErrorMemoryAllocationFailed);
    if(initialMeansArray != means)
    {
        size_t nCopy = nComponents * nFeatures * sizeof(algorithmFPType);
        daal_memcpy_s(means, nCopy, initialMeansArray, nCopy);
    }

    size_t nCopy = covs->getOneCovSize() * sizeof(algorithmFPType);
    algorithmFPType **sigma = covs->getSigma();
    for(size_t i = 0; i < nComponents; i++)
    {
        bd.set(initialCovariances[i], 0, covs->getNumberOfRowsInCov());
        const algorithmFPType *initCov = bd.get();
        DAAL_CHECK(initCov, ErrorMemoryAllocationFailed);
        if(initCov != sigma[i])
        {
            daal_memcpy_s(sigma[i], nCopy, initCov, nCopy);
        }
    }
}

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

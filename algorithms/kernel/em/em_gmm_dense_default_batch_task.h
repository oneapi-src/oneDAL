/* file: em_gmm_dense_default_batch_task.h */
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
//  Implementation of task for the em algorithm
//--

#ifndef __EM_GMM_DENSE_DEFAULT_BATCH_TASK_H__
#define __EM_GMM_DENSE_DEFAULT_BATCH_TASK_H__

#include "service_memory.h"
#include "service_data_utils.h"
#include "numeric_table.h"
#include "service_numeric_table.h"
#include "service_blas.h"
#include "service_stat.h"
#include "service_math.h"
#include "service_sort.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::data_feature_utils::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
class GmmModel
{
public:
    DAAL_NEW_DELETE();
    GmmModel(size_t _nFeatures, size_t _nComponents) :
        nFeatures(_nFeatures),
        nComponents(_nComponents),
        sigma(nullptr),
        sigmaPtr(_nComponents),
        logSqrtInvDetSigmaPtr(_nComponents)
    {
        sigma = sigmaPtr.get();
        logSqrtInvDetSigma = logSqrtInvDetSigmaPtr.get();
        EIGENVALUE_THRESHOLD = 1000 * MinVal<algorithmFPType, cpu>::get();
    }
    virtual ~GmmModel() {}
    algorithmFPType **getSigma() {return sigma;}
    algorithmFPType *getLogSqrtInvDetSigma() {return logSqrtInvDetSigma;}

    void setToZero()
    {
        size_t nElements = getOneCovSize();
        for(size_t i = 0; i < nComponents; i++)
        {
            for(size_t j = 0; j < nElements; j++)
            {
                sigma[i][j] = 0;
            }
        }
    }

    virtual size_t getOneCovSize() = 0;
    virtual size_t getNumberOfRowsInCov() = 0;
    virtual void multiplyByInverseMatrix(size_t nVectorsInCurrentBlock, size_t k, algorithmFPType *X, algorithmFPType *CovX) = 0;
    virtual void computeSigmaInverse(size_t iteration, KernelErrorCollection *errorCollection) = 0;
    virtual int computeThreadPartialResults(algorithmFPType *data, algorithmFPType *weights, size_t nFeatures, size_t nElements,
                                            algorithmFPType *sumOfWeights,
                                            algorithmFPType *partialMean,
                                            algorithmFPType *partialCovs) = 0;
    virtual void stepM_mergeCovs(
        algorithmFPType *cp_n, algorithmFPType *cp_m,
        algorithmFPType *mean_n, algorithmFPType *mean_m,
        algorithmFPType &w_n, algorithmFPType &w_m,
        size_t nFeatures) = 0;
    virtual void finalize(size_t k, algorithmFPType denominator) = 0;
    virtual void setCovRegularizer(double _covRegularizer) {covRegularizer = _covRegularizer;}

protected:
    algorithmFPType **sigma;
    algorithmFPType *logSqrtInvDetSigma;
    size_t nComponents;
    size_t nFeatures;
    TSmartPtr<algorithmFPType *, cpu> sigmaPtr;
    TSmartPtr<algorithmFPType, cpu> logSqrtInvDetSigmaPtr;
    double covRegularizer;
    algorithmFPType EIGENVALUE_THRESHOLD;
};

template<typename algorithmFPType, CpuType cpu>
class GmmModelFull : public GmmModel<algorithmFPType, cpu>
{
public:
    using GmmModel<algorithmFPType, cpu>::nFeatures;
    using GmmModel<algorithmFPType, cpu>::nComponents;
    using GmmModel<algorithmFPType, cpu>::sigma;
    using GmmModel<algorithmFPType, cpu>::logSqrtInvDetSigma;
    using GmmModel<algorithmFPType, cpu>::covRegularizer;
    using GmmModel<algorithmFPType, cpu>::EIGENVALUE_THRESHOLD;
    typedef Blas<algorithmFPType, cpu> blas;

    GmmModelFull(size_t _nFeatures, size_t _nComponents) : GmmModel<algorithmFPType, cpu>(_nFeatures, _nComponents) {}
    size_t getOneCovSize() {return nFeatures * nFeatures;}
    size_t getNumberOfRowsInCov() {return nFeatures;}
    void computeSigmaInverse(size_t iteration, KernelErrorCollection *errorCollection);
    void multiplyByInverseMatrix(size_t nVectorsInCurrentBlock, size_t k, algorithmFPType *X, algorithmFPType *CovX)
    {
        char side = 'L';
        char uplo = 'U';
        DAAL_INT m = nFeatures;
        DAAL_INT n = nVectorsInCurrentBlock;
        algorithmFPType alphaCoeff = 1.0;
        algorithmFPType betaCoeff = 0.0;
        DAAL_INT lda = m;

        algorithmFPType *invSigma = sigma[k];
        blas::xxsymm(&side, &uplo, &m, &n, &alphaCoeff, invSigma, &lda, X, &m, &betaCoeff, CovX, &lda);
    }

    int computeThreadPartialResults(algorithmFPType *data, algorithmFPType *weights, size_t nFeatures, size_t nElements,
                                    algorithmFPType *sumOfWeights,
                                    algorithmFPType *partialMean,
                                    algorithmFPType *partialCovs)
    {
        __int64 matCompMethod = __DAAL_VSL_SS_METHOD_FAST;
        int errcode = Statistics<algorithmFPType, cpu>::xxcp_weight(data, nFeatures, nElements,
                      weights,
                      sumOfWeights,
                      partialMean,
                      partialCovs,
                      matCompMethod);
        return errcode;
    }

    void finalize(size_t k, algorithmFPType denominator)
    {
        algorithmFPType multplier = 1.0 / denominator;
        for(size_t i = 0; i < nFeatures; i++)
        {
            for(size_t j = 0; j < i; j++)
            {
                sigma[k][i * nFeatures + j] *= multplier;
                sigma[k][j * nFeatures + i] = sigma[k][i * nFeatures + j];
            }
            sigma[k][i * nFeatures + i] *= multplier;
        }
    }

    void stepM_mergeCovs(
        algorithmFPType *cp_n, algorithmFPType *cp_m,
        algorithmFPType *mean_n, algorithmFPType *mean_m,
        algorithmFPType &w_n, algorithmFPType &w_m,
        size_t nFeatures);

    void regularizeCovarianceMatrix(algorithmFPType *cov, Error *error);
};

template<typename algorithmFPType, CpuType cpu>
class GmmModelDiag : public GmmModel<algorithmFPType, cpu>
{
public:
    using GmmModel<algorithmFPType, cpu>::nFeatures;
    using GmmModel<algorithmFPType, cpu>::nComponents;
    using GmmModel<algorithmFPType, cpu>::sigma;
    using GmmModel<algorithmFPType, cpu>::logSqrtInvDetSigma;
    using GmmModel<algorithmFPType, cpu>::covRegularizer;
    using GmmModel<algorithmFPType, cpu>::EIGENVALUE_THRESHOLD;
    GmmModelDiag(size_t _nFeatures, size_t _nComponents) : GmmModel<algorithmFPType, cpu>(_nFeatures, _nComponents) {}
    size_t getOneCovSize() {return nFeatures;}
    size_t getNumberOfRowsInCov() {return 1;}
    void multiplyByInverseMatrix(size_t nVectorsInCurrentBlock, size_t k, algorithmFPType *X, algorithmFPType *CovX)
    {
        algorithmFPType *invSigma = sigma[k];
        for(size_t i = 0; i < nVectorsInCurrentBlock; i++)
        {
            for(size_t j = 0; j < nFeatures; j++)
            {
                CovX[i * nFeatures + j] = X[i * nFeatures + j] * invSigma[j];
            }
        }
    }
    void regularizeCovarianceMatrix(algorithmFPType *cov, Error *error)
    {
        TSmartPtr<algorithmFPType, cpu> sortedCovsPtr(nFeatures);
        algorithmFPType *sortedCovs = sortedCovsPtr.get();
        for(size_t j = 0; j < nFeatures; j++)
        {
            sortedCovs[j] = cov[j];
        }
        daal::algorithms::internal::qSort<algorithmFPType, cpu>(nFeatures, sortedCovs);

        algorithmFPType *eigenvalues = sortedCovs;
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
            cov[j] += cur_eigenvalue;
        }
    }

    void computeSigmaInverse(size_t iteration, KernelErrorCollection *errorCollection)
    {
        Error e;
        algorithmFPType **invSigma = sigma;
        algorithmFPType *sqrtInvDetSigma = logSqrtInvDetSigma;

        for(size_t k = 0; k < nComponents; k++)
        {
            for(size_t j = 0; j < nFeatures; j++)
            {
                if(sigma[k][j] < EIGENVALUE_THRESHOLD)
                {
                    regularizeCovarianceMatrix(sigma[k], &e);
                    break;
                }
            }
            if(e.id() != NoErrorMessageFound)
            {
                SharedPtr<Error> eCopy = SharedPtr<Error>(new Error(e));
                eCopy->addIntDetail(Iteration, iteration + 1);
                errorCollection->add(eCopy);
                return;
            }
            algorithmFPType determ = 1;
            for(size_t j = 0; j < nFeatures; j++)
            {
                determ *= sigma[k][j];
                invSigma[k][j] = 1.0 / sigma[k][j];
            }
            sqrtInvDetSigma[k] = 1.0 / Math<algorithmFPType, cpu>::sSqrt(determ);
        }
    }

    int computeThreadPartialResults(algorithmFPType *data, algorithmFPType *weights, size_t nFeatures, size_t nElements,
                                    algorithmFPType *sumOfWeights,
                                    algorithmFPType *partialMean,
                                    algorithmFPType *partialCovs)
    {
        __int64 matCompMethod = __DAAL_VSL_SS_METHOD_FAST;
        int errcode = Statistics<algorithmFPType, cpu>::xxvar_weight(data, nFeatures, nElements,
                      weights,
                      sumOfWeights,
                      partialMean,
                      partialCovs,
                      matCompMethod);
        return errcode;
    }

    void finalize(size_t k, algorithmFPType denominator)
    {
        algorithmFPType multplier = 1.0 / denominator;
        for(size_t i = 0; i < nFeatures; i++)
        {
            sigma[k][i] *= multplier;
        }
    }

    void stepM_mergeCovs(
        algorithmFPType *cp_n, algorithmFPType *cp_m,
        algorithmFPType *mean_n, algorithmFPType *mean_m,
        algorithmFPType &w_n, algorithmFPType &w_m,
        size_t nFeatures);
};

template<typename algorithmFPType, CpuType cpu>
struct Task
{
    Task(NumericTable *_dataTable, size_t blockSizeDefault, size_t _nFeatures, size_t _nComponents,
         algorithmFPType *_logAlpha, //placed in alpha memory
         algorithmFPType *_means,
         GmmModel<algorithmFPType, cpu> *_covs) :
        dataTable(_dataTable),
        dataBlock(nullptr),
        logAlpha(_logAlpha),
        means(_means),
        covs(_covs),
        invSigma(_covs->getSigma()),
        logSqrtInvDetSigma(_covs->getLogSqrtInvDetSigma()),
        nFeatures(_nFeatures),
        nComponents(_nComponents)
    {
        size_t sizeOfOneCov = covs->getOneCovSize();
        size_t memorySizeForOneThread = blockSizeDefault * nFeatures        + /* x_mu   */
                                        blockSizeDefault * nFeatures        + /* Ax_mu  */
                                        blockSizeDefault * nComponents      + /* p      */
                                        blockSizeDefault                    + /* rowSum */
                                        nComponents                         + /* wSums */
                                        nComponents * nFeatures             + /* partialMeans */
                                        nComponents * sizeOfOneCov          + /* partialCP */
                                        nComponents                         + /* mergedWSums */
                                        nComponents * nFeatures             + /* mergedPartialMeans */
                                        nComponents * sizeOfOneCov          ; /* mergedPartialCP */

        threadBufferPtr.reset(memorySizeForOneThread);
        localBuffer = threadBufferPtr.get();
        if(!localBuffer) {return;}

        x_mu                = localBuffer;
        Ax_mu               = &x_mu              [blockSizeDefault * nFeatures       ];
        p                   = &Ax_mu             [blockSizeDefault * nFeatures       ];
        rowSum              = &p                 [blockSizeDefault * nComponents     ];
        wSums               = &rowSum            [blockSizeDefault                   ];
        partialMeans        = &wSums             [nComponents                        ];
        partialCP           = &partialMeans      [nComponents * nFeatures            ];

        mergedWSums         = &partialCP         [nComponents * sizeOfOneCov];
        mergedPartialMeans  = &mergedWSums       [nComponents                        ];
        mergedPartialCP     = &mergedPartialMeans[nComponents * nFeatures            ];
        setMergedToZero();
    }

    void setMergedToZero()
    {
        size_t sizeOfOneCov = covs->getOneCovSize();
        for(size_t i = 0; i < nComponents; i++) { mergedWSums[i] = 0; }
        for(size_t i = 0; i < nComponents * nFeatures; i++) { mergedPartialMeans[i] = 0; }
        for(size_t i = 0; i < nComponents * sizeOfOneCov; i++) { mergedPartialCP[i] = 0; }
    }

    void next(size_t j0, size_t nVectorsInCurrentBlock, Error *localError)
    {
        dataTableBD.set(dataTable, j0, nVectorsInCurrentBlock);
        dataBlock = dataTableBD.get();
        if(!dataBlock) {localError->setId(ErrorMemoryAllocationFailed); return;}
    }

    NumericTable *dataTable;
    algorithmFPType *localBuffer;
    const algorithmFPType *dataBlock;
    ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD;
    TSmartPtr<algorithmFPType, cpu> threadBufferPtr;

    algorithmFPType *x_mu;
    algorithmFPType *Ax_mu;
    algorithmFPType *w;
    algorithmFPType *p;
    algorithmFPType *rowSum;
    algorithmFPType *rowSumInv;

    algorithmFPType *logAlpha;
    algorithmFPType *means;
    algorithmFPType **invSigma;
    algorithmFPType *logSqrtInvDetSigma;
    algorithmFPType partLogLikelyhood;

    algorithmFPType *wSums;
    algorithmFPType *partialMeans;
    algorithmFPType *partialCP;

    algorithmFPType *mergedWSums;
    algorithmFPType *mergedPartialMeans;
    algorithmFPType *mergedPartialCP;
    GmmModel<algorithmFPType, cpu> *covs;

    size_t nFeatures;
    size_t nComponents;
};

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif

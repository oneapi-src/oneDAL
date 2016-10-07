/* file: em_gmm_init_dense_default_batch_kernel.h */
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
//  Implementation of em algorithm
//--
*/

#ifndef __EM_GMM_INIT_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __EM_GMM_INIT_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "kernel.h"
#include "service_numeric_table.h"
#include "numeric_table.h"
#include "homogen_numeric_table.h"
#include "service_memory.h"
#include "em_gmm_init_types.h"
#include "em_gmm_init_batch.h"
#include "em_gmm.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace init
{
namespace internal
{

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template<typename algorithmFPType, CpuType cpu>
class GmmSigma
{
public:
    GmmSigma(em_gmm::CovarianceStorageId _covType, size_t _nComponents, size_t _nFeatures) : covType(_covType), nComponents(_nComponents), nFeatures(_nFeatures),
        sigma(new DataCollection())
    {
        nRows = nFeatures;
        if (covType == em_gmm::diagonal) {nRows = 1;}
        for(size_t i = 0; i < nComponents; i++)
        {
            sigma->push_back(SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > (new HomogenNumericTableCPU<algorithmFPType, cpu>(nFeatures, nRows)));
        }
    }
    void setVariance(algorithmFPType *varianceArray)
    {
        for(int k = 0; k < nComponents; k++)
        {
            SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > workSigma = staticPointerCast<HomogenNumericTableCPU<algorithmFPType, cpu>, SerializationIface>((*sigma)[k]);
            algorithmFPType *sigmaArray = workSigma->getArray();
            if(covType == em_gmm::diagonal)
            {
                for(int i = 0; i < nFeatures; i++)
                {
                    sigmaArray[i] = varianceArray[i];
                }
            }
            else
            {
                for(int i = 0; i < nFeatures * nFeatures; i++)
                {
                    sigmaArray[i] = 0.0;
                }
                for(int i = 0; i < nFeatures; i++)
                {
                    sigmaArray[i * nFeatures + i] = varianceArray[i];
                }
            }
        }
    }
    DataCollectionPtr &getSigma()
    {
        return sigma;
    }
    void writeToTables(DataCollectionPtr covariancesToInit)
    {
        for (size_t k = 0; k < nComponents; k++)
        {
            NumericTablePtr covariance = staticPointerCast<NumericTable, SerializationIface>((*covariancesToInit)[k]);
            BlockDescriptor<algorithmFPType> covarianceBlock;
            covariance->getBlockOfRows(0, nRows, writeOnly, covarianceBlock);
            algorithmFPType* covarianceArray = covarianceBlock.getBlockPtr();

            SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > workSigma =
                staticPointerCast<HomogenNumericTableCPU<algorithmFPType, cpu>, SerializationIface>((*sigma)[k]);
            for (size_t i = 0; i < nRows * nFeatures; i++)
            {
                covarianceArray[i] = (workSigma->getArray())[i];
            }

            covariance->releaseBlockOfRows(covarianceBlock);
        }
    }

private:
    DataCollectionPtr sigma;
    em_gmm::CovarianceStorageId covType;
    size_t nFeatures;
    size_t nComponents;
    size_t nRows;
};

template<typename algorithmFPType, Method method, CpuType cpu>
class EMInitKernel : public Kernel
{
public:
    EMInitKernel() :
        nComponents(0), nFeatures(0), nVectors(0), nTrials(0), nIterations(0), accuracyThreshold(0),
        selectedSet(NULL), seedArray(NULL),
        loglikelyhood(0), maxLoglikelyhood(0) {};
    void compute(const NumericTablePtr &data, const NumericTablePtr &weightsToInit, const NumericTablePtr &meansToInit,
                 const DataCollectionPtr &covariancesToInit, Parameter *par);

private:
    void writeValuesToTables(const NumericTablePtr &weightsToInit, const NumericTablePtr &meansToInit,
                             const DataCollectionPtr &covariancesToInit,  GmmSigma<algorithmFPType, cpu> &covs);
    void setSelectedSetAsInitialValues(int *selectedSet,  GmmSigma<algorithmFPType, cpu> &covs);
    ErrorID runEM( GmmSigma<algorithmFPType, cpu> &covs, const em_gmm::CovarianceStorageId covType);
    void generateSelectedSet(int *selectedSet, size_t length, int seed);

    NumericTablePtr data;
    size_t nComponents;
    size_t nFeatures;
    size_t nVectors;
    size_t nTrials;
    size_t nIterations;
    double accuracyThreshold;
    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > alpha;
    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > means;
    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > variance;
    algorithmFPType *buffer;
    int *selectedSet;
    int *seedArray;
    algorithmFPType loglikelyhood;
    algorithmFPType maxLoglikelyhood;
};

template<typename algorithmFPType>
class EMforKernel : public daal::algorithms::em_gmm::Batch<algorithmFPType, em_gmm::defaultDense>
{
public:
    EMforKernel(const size_t nComponents) : daal::algorithms::em_gmm::Batch<algorithmFPType, em_gmm::defaultDense>(nComponents) {}
    virtual ~EMforKernel()
    {}

    ErrorID run(const data_management::NumericTablePtr &inputData,
                const data_management::NumericTablePtr &inputWeights,
                const data_management::NumericTablePtr &inputMeans,
                const data_management::DataCollectionPtr &inputCov,
                const em_gmm::CovarianceStorageId covType,
                algorithmFPType &loglikelyhood);

};

} // namespace internal

} // namespace init

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif

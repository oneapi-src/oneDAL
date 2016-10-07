/* file: em_gmm_init_dense_default_batch_impl.i */
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

#include "em_gmm_init_dense_default_batch_kernel.h"
#include "em_gmm_dense_default_batch_kernel.h"
#include "service_data_utils.h"
#include "service_rng.h"
#include "service_stat.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::data_feature_utils::internal;
using namespace daal::services;

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

template<typename algorithmFPType, Method method, CpuType cpu>
void EMInitKernel<algorithmFPType, method, cpu>::compute(const NumericTablePtr &data, const NumericTablePtr &weightsToInit,
        const NumericTablePtr &meansToInit, const DataCollectionPtr &covariancesToInit, Parameter *parameter)
{
    this->selectedSet = NULL;
    this->seedArray = NULL;
    this->data = data;
    this->nComponents = parameter->nComponents;
    this->nTrials = parameter->nTrials;
    this->nIterations = parameter->nIterations;
    this->accuracyThreshold = parameter->accuracyThreshold;
    maxLoglikelyhood = -MaxVal<algorithmFPType, cpu>::get();

    BaseRNGs<cpu> baseRng(parameter->seed);
    RNGs<int, cpu> rng;

    nFeatures = data->getNumberOfColumns();
    nVectors  = data->getNumberOfRows();

    alpha = SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> >(new HomogenNumericTableCPU<algorithmFPType, cpu>(nComponents, 1));
    means = SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> >(new HomogenNumericTableCPU<algorithmFPType, cpu>(nFeatures, nComponents));
    GmmSigma<algorithmFPType, cpu> covs(parameter->covarianceStorage, nComponents, nFeatures);

    variance = SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> >(new HomogenNumericTableCPU<algorithmFPType, cpu>(nFeatures, 1));

    seedArray   = (int *) daal::services::daal_malloc(nTrials * sizeof(int));
    selectedSet = (int *) daal::services::daal_malloc(nComponents * sizeof(int));

    int errCode = rng.uniform(nTrials, seedArray, baseRng, 0, 1000000);
    if(errCode) { this->_errors->add(ErrorIncorrectErrorcodeFromGenerator); }

    BlockDescriptor<algorithmFPType> block;
    data->getBlockOfRows(0, nVectors, readOnly, block);
    algorithmFPType *dataArray = block.getBlockPtr();

    Statistics<algorithmFPType, cpu>::x2c_mom(dataArray, nFeatures, nVectors, variance->getArray(), __DAAL_VSL_SS_METHOD_FAST);

    data->releaseBlockOfRows(block);

    bool isInitialized = false;
    for(int idxTry = 0; idxTry < nTrials; idxTry++)
    {
        generateSelectedSet(selectedSet, nComponents, seedArray[idxTry]);

        setSelectedSetAsInitialValues(selectedSet, covs);

        ErrorID errorId = runEM(covs, parameter->covarianceStorage);

        if(!errorId && (loglikelyhood > maxLoglikelyhood))
        {
            isInitialized = true;
            maxLoglikelyhood = loglikelyhood;
            writeValuesToTables(weightsToInit, meansToInit, covariancesToInit, covs);
        }
    }

    if(seedArray   != NULL) { daal::services::daal_free(seedArray); }
    if(selectedSet != NULL) { daal::services::daal_free(selectedSet); }

    if(!isInitialized) {this->_errors->add(ErrorEMInitNoTrialConverges);}
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMInitKernel<algorithmFPType, method, cpu>::writeValuesToTables(const NumericTablePtr &weightsToInit,
        const NumericTablePtr &meansToInit, const DataCollectionPtr &covariancesToInit,  GmmSigma<algorithmFPType, cpu> &covs)
{
    algorithmFPType *weightsArray, *meansArray;

    BlockDescriptor<algorithmFPType> weightsBlock;
    weightsToInit->getBlockOfRows(0, 1, writeOnly, weightsBlock);
    weightsArray = weightsBlock.getBlockPtr();
    for (size_t i = 0; i < nComponents; i++)
    {
        weightsArray[i] = alpha->getArray()[i];
    }
    weightsToInit->releaseBlockOfRows(weightsBlock);

    BlockDescriptor<algorithmFPType> meansBlock;
    meansToInit->getBlockOfRows(0, nComponents, writeOnly, meansBlock);
    meansArray = meansBlock.getBlockPtr();
    for (size_t i = 0; i < nFeatures * nComponents; i++)
    {
        meansArray[i] = (means->getArray())[i];
    }
    meansToInit->releaseBlockOfRows(meansBlock);

    covs.writeToTables(covariancesToInit);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMInitKernel<algorithmFPType, method, cpu>::setSelectedSetAsInitialValues(int *selectedSet,  GmmSigma<algorithmFPType, cpu> &covs)
{
    algorithmFPType *alphaArray = alpha->getArray();
    for(int k = 0; k < nComponents; k++)
    {
        alphaArray[k] = 1.0 / nComponents;
    }

    BlockDescriptor<algorithmFPType> block;
    algorithmFPType *meansArray = means->getArray();
    for(int k = 0; k < nComponents; k++)
    {
        data->getBlockOfRows(selectedSet[k], 1, readOnly, block);
        algorithmFPType *selectedRow = block.getBlockPtr();

        for(int j = 0; j < nFeatures; j++)
        {
            meansArray[k * nFeatures + j] = selectedRow[j];
        }

        data->releaseBlockOfRows(block);
    }

    algorithmFPType *varianceArray = variance->getArray();
    covs.setVariance(varianceArray);
}

template<typename algorithmFPType, Method method, CpuType cpu>
ErrorID EMInitKernel<algorithmFPType, method, cpu>::runEM( GmmSigma<algorithmFPType, cpu> &covs, em_gmm::CovarianceStorageId covType)
{
    EMforKernel<algorithmFPType> em(nComponents);
    em.parameter.maxIterations = nIterations;
    em.parameter.accuracyThreshold = accuracyThreshold;
    ErrorID returnErrorId = em.run(data, alpha, means, covs.getSigma(), covType, loglikelyhood);
    if(returnErrorId != 0)
    {
        loglikelyhood = -MaxVal<algorithmFPType, cpu>::get();
    }
    return returnErrorId;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void EMInitKernel<algorithmFPType, method, cpu>::generateSelectedSet(int *selectedSet, size_t length, int seed)
{
    BaseRNGs<cpu> baseRng(seed);
    RNGs<int, cpu> rng;

    int number;
    for(int i = 0; i < length; i++)
    {
        bool isNumberUnique = false;
        while(isNumberUnique != true)
        {
            int errCode = rng.uniform(1, &number, baseRng, 0, (int)nVectors);
            if(errCode) { this->_errors->add(ErrorIncorrectErrorcodeFromGenerator); }

            isNumberUnique = true;
            for(int j = 0; j < i; j++)
            {
                if(number == selectedSet[j])
                {
                    isNumberUnique = false;
                }
            }
        }
        selectedSet[i] = number;
    }
}

} // namespace internal

} // namespace init

} // namespace em_gmm

} // namespace algorithms

}; // namespace daal

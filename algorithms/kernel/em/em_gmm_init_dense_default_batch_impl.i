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

template<typename AlgorithmFPType, Method method, CpuType cpu>
void EMInitKernel<AlgorithmFPType, method, cpu>::compute(const NumericTablePtr& data, const NumericTablePtr& weightsToInit,
        const NumericTablePtr& meansToInit, const DataCollectionPtr& covariancesToInit, Parameter *parameter)
{
    this->selectedSet = NULL;
    this->seedArray = NULL;
    this->data = data;
    this->nComponents = parameter->nComponents;
    this->nTrials = parameter->nTrials;
    this->nIterations = parameter->nIterations;
    this->accuracyThreshold = parameter->accuracyThreshold;
    maxLoglikelyhood = -MaxVal<AlgorithmFPType, cpu>::get();

    IntRng<int, cpu> rng(parameter->seed);

    nFeatures   = data->getNumberOfColumns();
    nVectors    = data->getNumberOfRows();

    alpha = SharedPtr<HomogenNumericTableCPU<AlgorithmFPType, cpu> >(new HomogenNumericTableCPU<AlgorithmFPType, cpu>(nComponents, 1));
    means = SharedPtr<HomogenNumericTableCPU<AlgorithmFPType, cpu> >(new HomogenNumericTableCPU<AlgorithmFPType, cpu>(nFeatures, nComponents));
    sigma = DataCollectionPtr(new DataCollection());
    for(int i = 0; i < nComponents; i++)
    {
        sigma->push_back(SharedPtr<HomogenNumericTableCPU<AlgorithmFPType, cpu> >(
                             new HomogenNumericTableCPU<AlgorithmFPType, cpu>(nFeatures, nFeatures)));
    }

    variance = SharedPtr<HomogenNumericTableCPU<AlgorithmFPType, cpu> >(new HomogenNumericTableCPU<AlgorithmFPType, cpu>(nFeatures, 1));

    seedArray   = (int *) daal::services::daal_malloc(nTrials * sizeof(int));
    selectedSet = (int *) daal::services::daal_malloc(nComponents * sizeof(int));

    rng.uniform(nTrials, 0, 1000000, seedArray);

    BlockDescriptor<AlgorithmFPType> block;
    data->getBlockOfRows(0, nVectors, readOnly, block);
    AlgorithmFPType *dataArray = block.getBlockPtr();

    Statistics<AlgorithmFPType, cpu>::x2c_mom(dataArray, nFeatures, nVectors, variance->getArray(), __DAAL_VSL_SS_METHOD_FAST);

    data->releaseBlockOfRows(block);

    bool isInitialized = false;
    for(int idxTry = 0; idxTry < nTrials; idxTry++)
    {
        generateSelectedSet(selectedSet, nComponents, seedArray[idxTry]);

        setSelectedSetAsInitialValues(selectedSet);

        ErrorID errorId = runEM();

        if(!errorId && (loglikelyhood > maxLoglikelyhood))
        {
            isInitialized = true;
            maxLoglikelyhood = loglikelyhood;
            writeValuesToTables(weightsToInit, meansToInit, covariancesToInit);
        }
    }

    if(seedArray   != NULL) { daal::services::daal_free(seedArray); }
    if(selectedSet != NULL) { daal::services::daal_free(selectedSet); }

    if(!isInitialized) {this->_errors->add(ErrorEMInitNoTrialConverges);}
}

template<typename AlgorithmFPType, Method method, CpuType cpu>
void EMInitKernel<AlgorithmFPType, method, cpu>::writeValuesToTables(const NumericTablePtr& weightsToInit,
        const NumericTablePtr& meansToInit, const DataCollectionPtr& covariancesToInit)
{
    AlgorithmFPType *weightsArray, *meansArray, *covarianceArray;

    BlockDescriptor<AlgorithmFPType> weightsBlock;
    weightsToInit->getBlockOfRows(0, 1, writeOnly, weightsBlock);
    weightsArray = weightsBlock.getBlockPtr();
    for (size_t i = 0; i < nComponents; i++)
    {
        weightsArray[i] = alpha->getArray()[i];
    }
    weightsToInit->releaseBlockOfRows(weightsBlock);

    BlockDescriptor<AlgorithmFPType> meansBlock;
    meansToInit->getBlockOfRows(0, nComponents, writeOnly, meansBlock);
    meansArray = meansBlock.getBlockPtr();
    for (size_t i = 0; i < nFeatures * nComponents; i++)
    {
        meansArray[i] = (means->getArray())[i];
    }
    meansToInit->releaseBlockOfRows(meansBlock);

    for (size_t k = 0; k < nComponents; k++)
    {
        NumericTablePtr covariance = staticPointerCast<NumericTable, SerializationIface>((*covariancesToInit)[k]);
        BlockDescriptor<AlgorithmFPType> covarianceBlock;
        covariance->getBlockOfRows(0, nFeatures, writeOnly, covarianceBlock);
        covarianceArray = covarianceBlock.getBlockPtr();

        SharedPtr<HomogenNumericTableCPU<AlgorithmFPType, cpu> > workSigma =
            staticPointerCast<HomogenNumericTableCPU<AlgorithmFPType, cpu>, SerializationIface>((*sigma)[k]);
        for (size_t i = 0; i < nFeatures * nFeatures; i++)
        {
            covarianceArray[i] = (workSigma->getArray())[i];
        }

        covariance->releaseBlockOfRows(covarianceBlock);
    }
}

template<typename AlgorithmFPType, Method method, CpuType cpu>
void EMInitKernel<AlgorithmFPType, method, cpu>::setSelectedSetAsInitialValues(int *selectedSet)
{
    AlgorithmFPType *alphaArray = alpha->getArray();
    for(int k = 0; k < nComponents; k++)
    {
        alphaArray[k] = 1.0 / nComponents;
    }

    BlockDescriptor<AlgorithmFPType> block;
    AlgorithmFPType *meansArray = means->getArray();
    for(int k = 0; k < nComponents; k++)
    {
        data->getBlockOfRows(selectedSet[k], 1, readOnly, block);
        AlgorithmFPType *selectedRow = block.getBlockPtr();

        for(int j = 0; j < nFeatures; j++)
        {
            meansArray[k * nFeatures + j] = selectedRow[j];
        }

        data->releaseBlockOfRows(block);
    }

    AlgorithmFPType *varianceArray = variance->getArray();
    for(int k = 0; k < nComponents; k++)
    {
        SharedPtr<HomogenNumericTableCPU<AlgorithmFPType, cpu> > workSigma =
            staticPointerCast<HomogenNumericTableCPU<AlgorithmFPType, cpu>, SerializationIface>((*sigma)[k]);
        AlgorithmFPType *sigmaArray = workSigma->getArray();
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

template<typename AlgorithmFPType, Method method, CpuType cpu>
ErrorID EMInitKernel<AlgorithmFPType, method, cpu>::runEM()
{
    EMforKernel<AlgorithmFPType> em(nComponents);
    em.parameter.maxIterations = nIterations;
    em.parameter.accuracyThreshold = accuracyThreshold;
    ErrorID returnErrorId = em.run(data, alpha, means, sigma, loglikelyhood);
    if(returnErrorId != 0)
        loglikelyhood = -MaxVal<AlgorithmFPType, cpu>::get();
    return returnErrorId;
}

template<typename AlgorithmFPType, Method method, CpuType cpu>
void EMInitKernel<AlgorithmFPType, method, cpu>::generateSelectedSet(int *selectedSet, size_t length, int seed)
{
    daal::internal::IntRng<int,cpu> rng(seed);

    int number;
    for(int i = 0; i < length; i++)
    {
        bool isNumberUnique = false;
        while(isNumberUnique != true)
        {
            rng.uniform(1, 0, (int)nVectors, &number);
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

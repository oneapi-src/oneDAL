/* file: em_gmm_init_dense_default_batch_impl.i */
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
services::Status EMInitKernel<algorithmFPType, method, cpu>::compute(NumericTable &data, NumericTable &weightsToInit,
        NumericTable &meansToInit, DataCollectionPtr &covariancesToInit, const Parameter &parameter)
{
    EMInitKernelTask<algorithmFPType, method, cpu> kernelTask(data, weightsToInit, meansToInit, covariancesToInit, parameter);
    Status s;
    DAAL_CHECK_STATUS(s, kernelTask.compute())
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status EMInitKernelTask<algorithmFPType, method, cpu>::compute()
{
    Status s;
    DAAL_CHECK_STATUS(s, initialize())

    bool isInitialized = false;
    for(int idxTry = 0; idxTry < nTrials; idxTry++)
    {
        DAAL_CHECK_STATUS(s, generateSelectedSet(seedArray[idxTry]))

        DAAL_CHECK_STATUS(s, setSelectedSetAsInitialValues())

        ErrorID errorId = runEM();

        if(!errorId && (loglikelyhood > maxLoglikelyhood))
        {
            isInitialized = true;
            maxLoglikelyhood = loglikelyhood;
            DAAL_CHECK_STATUS(s, writeValuesToTables())
        }
    }

    DAAL_CHECK(isInitialized, ErrorEMInitNoTrialConverges)
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
EMInitKernelTask<algorithmFPType, method, cpu>::EMInitKernelTask(NumericTable &data, NumericTable &weightsToInit,
        NumericTable &meansToInit, DataCollectionPtr &covariancesToInit, const Parameter &parameter) :
    data(data),
    weightsToInit(weightsToInit),
    meansToInit(meansToInit),
    covariancesToInit(covariancesToInit),
    parameter(parameter),
    nComponents(parameter.nComponents),
    nTrials(parameter.nTrials),
    nIterations(parameter.nIterations),
    accuracyThreshold(parameter.accuracyThreshold),
    maxLoglikelyhood(-MaxVal<algorithmFPType, cpu>::get()),
    nFeatures(data.getNumberOfColumns()),
    nVectors(data.getNumberOfRows()),
    covs(parameter.covarianceStorage, parameter.nComponents, data.getNumberOfColumns()),
    varianceArrayPtr(data.getNumberOfColumns()),
    seedArrayPtr(parameter.nTrials),
    selectedSetPtr(parameter.nComponents)
{}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EMInitKernelTask<algorithmFPType, method, cpu>::initialize()
{
    alpha = HomogenNTPtr(new HomogenNT(nComponents, 1));
    means = HomogenNTPtr(new HomogenNT(nFeatures, nComponents));
    varianceArray = varianceArrayPtr.get();
    seedArray = seedArrayPtr.get();
    selectedSet = selectedSetPtr.get();

    DAAL_CHECK(alpha && means && varianceArray && seedArray && selectedSet, ErrorMemoryAllocationFailed)

    BaseRNGs<cpu> baseRng(parameter.seed);
    RNGs<int, cpu> rng;
    DAAL_CHECK(!rng.uniform(nTrials, seedArray, baseRng, 0, 1000000), ErrorIncorrectErrorcodeFromGenerator)

    return computeVariance();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EMInitKernelTask<algorithmFPType, method, cpu>::computeVariance()
{
    ReadRows<algorithmFPType, cpu, NumericTable> block(data, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(block)
    const algorithmFPType *dataArray = block.get();

    DAAL_CHECK((Statistics<algorithmFPType, cpu>::x2c_mom(dataArray, nFeatures, nVectors, varianceArray, __DAAL_VSL_SS_METHOD_FAST)) == 0, ErrorVarianceComputation)
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EMInitKernelTask<algorithmFPType, method, cpu>::writeValuesToTables()
{
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> weightsBlock(weightsToInit, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(weightsBlock)
        algorithmFPType *weightsArray = weightsBlock.get();
        for (size_t i = 0; i < nComponents; i++)
        {
            weightsArray[i] = alpha->getArray()[i];
        }
    }

    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> meansBlock(meansToInit, 0, nComponents);
        DAAL_CHECK_BLOCK_STATUS(meansBlock)
        algorithmFPType *meansArray = meansBlock.get();
        for (size_t i = 0; i < nFeatures * nComponents; i++)
        {
            meansArray[i] = means->getArray()[i];
        }
    }

    covs.writeToTables(covariancesToInit);
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EMInitKernelTask<algorithmFPType, method, cpu>::setSelectedSetAsInitialValues()
{
    algorithmFPType *alphaArray = alpha->getArray();
    for(int k = 0; k < nComponents; k++)
    {
        alphaArray[k] = 1.0 / nComponents;
    }

    algorithmFPType *meansArray = means->getArray();
    ReadRows<algorithmFPType, cpu, NumericTable> block;
    for(int k = 0; k < nComponents; k++)
    {
        const algorithmFPType *selectedRow = block.set(data, selectedSet[k], 1);
        DAAL_CHECK(selectedRow, ErrorMemoryAllocationFailed)
        for(int j = 0; j < nFeatures; j++)
        {
            meansArray[k * nFeatures + j] = selectedRow[j];
        }
    }

    covs.setVariance(varianceArray);
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
ErrorID EMInitKernelTask<algorithmFPType, method, cpu>::runEM()
{
    EMforKernel<algorithmFPType> em(nComponents);
    em.parameter.maxIterations = nIterations;
    em.parameter.accuracyThreshold = accuracyThreshold;
    ErrorID returnErrorId = em.run(data, *alpha, *means, covs.getSigma(), parameter.covarianceStorage, loglikelyhood);
    if(returnErrorId != 0)
    {
        loglikelyhood = -MaxVal<algorithmFPType, cpu>::get();
    }
    return returnErrorId;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EMInitKernelTask<algorithmFPType, method, cpu>::generateSelectedSet(int seed)
{
    BaseRNGs<cpu> baseRng(seed);
    RNGs<int, cpu> rng;

    int number;
    for(int i = 0; i < nComponents; i++)
    {
        bool isNumberUnique = false;
        while(isNumberUnique != true)
        {
            DAAL_CHECK(!rng.uniform(1, &number, baseRng, 0, (int)nVectors), ErrorIncorrectErrorcodeFromGenerator);

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
    return Status();
}

} // namespace internal

} // namespace init

} // namespace em_gmm

} // namespace algorithms

}; // namespace daal

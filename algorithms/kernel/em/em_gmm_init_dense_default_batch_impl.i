/* file: em_gmm_init_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of em algorithm
//--
*/

#include "em_gmm_init_dense_default_batch_kernel.h"
#include "em_gmm_dense_default_batch_kernel.h"
#include "service_data_utils.h"
#include "service_stat.h"
#include "uniform_impl.i"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::distributions::uniform::internal;

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
        NumericTable &meansToInit, DataCollectionPtr &covariancesToInit, const Parameter &parameter, engines::BatchBase &engine)
{
    Status s;
    EMInitKernelTask<algorithmFPType, method, cpu> kernelTask(data, weightsToInit, meansToInit, covariancesToInit, parameter, engine, s);//?
    if (!s) return s;
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
        DAAL_CHECK_STATUS(s, generateSelectedSet())

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
        NumericTable &meansToInit, DataCollectionPtr &covariancesToInit, const Parameter &parameter, engines::BatchBase &engine, Status &status) :
    data(data),
    weightsToInit(weightsToInit),
    meansToInit(meansToInit),
    covariancesToInit(covariancesToInit),
    parameter(parameter),
    nComponents(parameter.nComponents),
    nTrials(parameter.nTrials),
    nIterations(parameter.nIterations),
    accuracyThreshold(parameter.accuracyThreshold),
    maxLoglikelyhood(-MaxVal<algorithmFPType>::get()),
    nFeatures(data.getNumberOfColumns()),
    nVectors(data.getNumberOfRows()),
    covs(parameter.covarianceStorage, parameter.nComponents, data.getNumberOfColumns(), status),
    varianceArrayPtr(data.getNumberOfColumns()),
    selectedSetPtr(parameter.nComponents),
    engine(engine)
{}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EMInitKernelTask<algorithmFPType, method, cpu>::initialize()
{
    Status st;
    alpha = HomogenNT::create(nComponents, 1, &st);
    DAAL_CHECK_STATUS_VAR(st);
    means = HomogenNT::create(nFeatures, nComponents, &st);
    DAAL_CHECK_STATUS_VAR(st);
    varianceArray = varianceArrayPtr.get();
    selectedSet = selectedSetPtr.get();

    DAAL_CHECK(alpha && means && varianceArray && selectedSet, ErrorMemoryAllocationFailed);

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
        loglikelyhood = -MaxVal<algorithmFPType>::get();
    }
    return returnErrorId;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status EMInitKernelTask<algorithmFPType, method, cpu>::generateSelectedSet()
{
    int number;
    Status s;
    for(int i = 0; i < nComponents; i++)
    {
        bool isNumberUnique = false;
        while(isNumberUnique != true)
        {
            DAAL_CHECK_STATUS(s, (distributions::uniform::internal::UniformKernelDefault<int, cpu>::compute(0, (int)nVectors, engine, 1, &number)));
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
    return s;
}

} // namespace internal

} // namespace init

} // namespace em_gmm

} // namespace algorithms

}; // namespace daal

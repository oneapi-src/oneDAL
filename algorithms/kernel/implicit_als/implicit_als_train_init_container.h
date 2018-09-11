/* file: implicit_als_train_init_container.h */
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
//  Implementation of implicit ALS initialization algorithm container.
//--
*/

#ifndef __IMPICIT_ALS_TRAIN_INIT_CONTAINER_H__
#define __IMPICIT_ALS_TRAIN_INIT_CONTAINER_H__

#include "implicit_als_training_init_batch.h"
#include "implicit_als_training_init_distributed.h"
#include "implicit_als_train_init_kernel.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
/**
 *  \brief Initialize list of implicit ALS initialization algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::init::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : TrainingContainerIface<batch>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSInitKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable *a = input->get(data).get();
    Model *m = result->get(training::init::model).get();
    NumericTable *r = m->getItemsFactors().get();

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSInitKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, a, r, par, *par->engine);
}

/**
 *  \brief Initialize list of implicit ALS initialization algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSInitDistrKernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedParameter *par = static_cast<DistributedParameter *>(_par);
    DistributedInput<step1Local> *input = static_cast<DistributedInput<step1Local> *>(_in);
    PartialResult *partialResult = static_cast<PartialResult *>(_pres);

    NumericTable *dataTable = input->get(data).get();
    NumericTable *partitionTable = par->partition.get();
    size_t nRows = partitionTable->getNumberOfRows();
    size_t nParts = nRows - 1;
    if (nParts == 0)
    {
        BlockDescriptor<int> block;
        partitionTable->getBlockOfRows(0, nRows, readOnly, block);
        int *nPartsData = block.getBlockPtr();
        nParts = nPartsData[0];
        partitionTable->releaseBlockOfRows(block);
    }

    implicit_als::PartialModel *pModel = partialResult->get(partialModel).get();
    NumericTable *result = pModel->getFactors().get();

    daal::internal::TArray<NumericTable *, cpu> dataPartsPtr    (nParts);
    daal::internal::TArray<NumericTable *, cpu> blocksToLocalPtr(nParts);
    daal::internal::TArray<NumericTable *, cpu> userOffsetsPtr  (nParts);
    KeyValueDataCollection &dataPartsCollection     = *(partialResult->get(outputOfStep1ForStep2));
    KeyValueDataCollection &blocksToLocalCollection = *(partialResult->get(outputOfInitForComputeStep3));
    KeyValueDataCollection &userOffsetsCollection   = *(partialResult->get(offsets));
    for (size_t i = 0; i < nParts; i++)
    {
        dataPartsPtr    [i] = static_cast<NumericTable *>(dataPartsCollection    [i].get());
        blocksToLocalPtr[i] = static_cast<NumericTable *>(blocksToLocalCollection[i].get());
        userOffsetsPtr  [i] = static_cast<NumericTable *>(userOffsetsCollection  [i].get());
    }

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSInitDistrKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, dataTable, partitionTable, dataPartsPtr.get(), blocksToLocalPtr.get(), userOffsetsPtr.get(),
                       result, par, *par->engine);
}

/**
 *  \brief Initialize list of implicit ALS initialization algorithm
 *  kernels with implementations for supported architectures
 */
template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::ImplicitALSInitDistrStep2Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, training::init::Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}


template <typename algorithmFPType, training::init::Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Local> *input = static_cast<DistributedInput<step2Local> *>(_in);
    DistributedPartialResultStep2 *partialResult = static_cast<DistributedPartialResultStep2 *>(_pres);

    KeyValueDataCollection &dataPartsCollection     = *(input->get(inputOfStep2FromStep1));
    KeyValueDataCollection &blocksToLocalCollection = *(partialResult->get(outputOfInitForComputeStep3));
    KeyValueDataCollection &itemOffsetsCollection   = *(partialResult->get(offsets));
    size_t nParts = dataPartsCollection.size();
    daal::internal::TArray<NumericTable *, cpu> dataPartsPtr(nParts);
    daal::internal::TArray<NumericTable *, cpu> blocksToLocalPtr(nParts);
    daal::internal::TArray<NumericTable *, cpu> itemOffsetsPtr(nParts);

    for (size_t i = 0; i < nParts; i++)
    {
        dataPartsPtr[i]     = static_cast<NumericTable *>(dataPartsCollection    [i].get());
        blocksToLocalPtr[i] = static_cast<NumericTable *>(blocksToLocalCollection[i].get());
        itemOffsetsPtr[i]   = static_cast<NumericTable *>(itemOffsetsCollection  [i].get());
    }

    NumericTable *dataTable = partialResult->get(transposedData).get();
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::ImplicitALSInitDistrStep2Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, nParts, dataPartsPtr.get(), dataTable, blocksToLocalPtr.get(), itemOffsetsPtr.get());
}

}
}
}
}
}

#endif

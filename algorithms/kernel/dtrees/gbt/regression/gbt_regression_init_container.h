/* file: gbt_regression_init_container.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of the container for initializing gradient boosted trees
//  regression training algorithm in the distributed processing mode
//--
*/

#ifndef __GBT_REGRESSION_INIT_CONTAINER_H__
#define __GBT_REGRESSION_INIT_CONTAINER_H__

#include "kernel.h"
#include "numeric_table.h"
#include "gbt_regression_init_distributed.h"
#include "gbt_regression_init_types.h"
#include "gbt_regression_init_kernel.h"
#include "service_numeric_table.h"
#include "service_algo_utils.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace init
{

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionInitStep1LocalKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step1Local> *input = static_cast<DistributedInput<step1Local> *>(_in);
    DistributedPartialResultStep1 *partialResult = static_cast<DistributedPartialResultStep1 *>(_pres);

    const NumericTablePtr ntLocalData = input->get(step1LocalData);
    const NumericTablePtr ntLocalDependentVariables = input->get(step1LocalDependentVariables);

    const HomogenNumericTable<algorithmFPType> *ntMeanDependentVariable = dynamic_cast<HomogenNumericTable<algorithmFPType>*>((partialResult->get(step1MeanDependentVariable)).get());
    const HomogenNumericTable<size_t> *ntNumberOfRows = dynamic_cast<HomogenNumericTable<size_t>*>((partialResult->get(step1NumberOfRows)).get());
    const HomogenNumericTable<algorithmFPType> *ntBinBorders = dynamic_cast<HomogenNumericTable<algorithmFPType>*>((partialResult->get(step1BinBorders)).get());
    const HomogenNumericTable<size_t> *ntBinSizes = dynamic_cast<HomogenNumericTable<size_t>*>((partialResult->get(step1BinSizes)).get());

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionInitStep1LocalKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, ntLocalData.get(), ntLocalDependentVariables.get(), ntMeanDependentVariable, ntNumberOfRows, ntBinBorders, ntBinSizes, *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
    __DAAL_INITIALIZE_KERNELS(internal::RegressionInitStep2MasterKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step2Master> *input = static_cast<DistributedInput<step2Master> *>(_in);
    DistributedPartialResultStep2 *partialResult = static_cast<DistributedPartialResultStep2 *>(_pres);

    const DataCollectionPtr dcBinBorders = input->get(step2BinBorders);
    const DataCollectionPtr dcBinSizes = input->get(step2BinSizes);
    const DataCollectionPtr dcMeanDependentVariable = input->get(step2MeanDependentVariable);
    const DataCollectionPtr dcNumberOfRows = input->get(step2NumberOfRows);

    size_t nNodes = dcBinBorders->size();

    HomogenNumericTable<algorithmFPType> *ntInitialResponse = dynamic_cast<HomogenNumericTable<algorithmFPType>*>((partialResult->get(step2InitialResponse)).get());
    const HomogenNumericTable<algorithmFPType> *ntMergedBinBorders = dynamic_cast<HomogenNumericTable<algorithmFPType>*>((partialResult->get(step2MergedBinBorders)).get());
    const HomogenNumericTable<size_t> *ntBinQuantities = dynamic_cast<HomogenNumericTable<size_t>*>((partialResult->get(step2BinQuantities)).get());
    const DataCollectionPtr dcBinValues = partialResult->get(step2BinValues);

    Parameter *par = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionInitStep2MasterKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, nNodes, dcNumberOfRows, dcMeanDependentVariable, dcBinBorders, dcBinSizes, ntInitialResponse, ntMergedBinBorders, ntBinQuantities,
                       dcBinValues.get(), *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv) :
    TrainingContainerIface<distributed>()
{
     __DAAL_INITIALIZE_KERNELS(internal::RegressionInitStep3LocalKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::compute()
{
    DistributedInput<step3Local> *input = static_cast<DistributedInput<step3Local> *>(_in);
    const DistributedPartialResultStep3 *partialResult = static_cast<DistributedPartialResultStep3 *>(_pres);
    Parameter *par = static_cast<Parameter *>(_par);

    const HomogenNumericTable<algorithmFPType> *ntMergedBinBorders = dynamic_cast<HomogenNumericTable<algorithmFPType>*>(input->get(step3MergedBinBorders).get());
    const HomogenNumericTable<size_t> *ntBinQuantities = static_cast<HomogenNumericTable<size_t>*>((input->get(step3BinQuantities)).get());
    const NumericTablePtr ntLocalData = input->get(step3LocalData);
    const HomogenNumericTable<algorithmFPType> *ntInitialResponse = dynamic_cast<HomogenNumericTable<algorithmFPType>*>(input->get(step3InitialResponse).get());

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::RegressionInitStep3LocalKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                       compute, ntMergedBinBorders, ntBinQuantities, ntLocalData.get(), ntInitialResponse, partialResult, *par);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

} // namespace init
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif

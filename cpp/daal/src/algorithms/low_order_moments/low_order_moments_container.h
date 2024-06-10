/* file: low_order_moments_container.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of low order moments calculation algorithm container.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_CONTAINER_H__
#define __LOW_ORDER_MOMENTS_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/moments/low_order_moments_batch.h"
#include "algorithms/moments/low_order_moments_online.h"
#include "algorithms/moments/low_order_moments_distributed.h"
#include "src/algorithms/low_order_moments/low_order_moments_kernel.h"
#include "services/internal/execution_context.h"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = daal::services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (method != defaultDense || deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS(internal::LowOrderMomentsBatchKernel, algorithmFPType, method);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    NumericTable * dataTable = input->get(data).get();

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    auto & context    = daal::services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (method != defaultDense || deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL(env, internal::LowOrderMomentsBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dataTable, result,
                           par);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::OnlineContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = daal::services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (method != defaultDense || deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS(internal::LowOrderMomentsOnlineKernel, algorithmFPType, method);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
OnlineContainer<algorithmFPType, method, cpu>::~OnlineContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::compute()
{
    bool isOnline                 = true;
    Input * input                 = static_cast<Input *>(_in);
    PartialResult * partialResult = static_cast<PartialResult *>(_pres);

    NumericTable * dataTable = input->get(data).get();

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    auto & context    = daal::services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (method != defaultDense || deviceInfo.isCpu)
    {
        __DAAL_CALL_KERNEL(env, internal::LowOrderMomentsOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, dataTable,
                           partialResult, par, isOnline);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status OnlineContainer<algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * partialResult = static_cast<PartialResult *>(_pres);
    Result * result               = static_cast<Result *>(_res);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    // for other methods oneapi isn't implemented yet
    auto & context    = daal::services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    result->set(minimum, partialResult->get(partialMinimum));
    result->set(maximum, partialResult->get(partialMaximum));
    result->set(sum, partialResult->get(partialSum));
    result->set(sumSquares, partialResult->get(partialSumSquares));
    result->set(sumSquaresCentered, partialResult->get(partialSumSquaresCentered));

    if (method != defaultDense || deviceInfo.isCpu)
    {
        NumericTable * nObservationsTable = partialResult->get(nObservations).get();
        NumericTable * sumTable           = partialResult->get(partialSum).get();
        NumericTable * sumSqTable         = partialResult->get(partialSumSquares).get();
        NumericTable * sumSqCenTable      = partialResult->get(partialSumSquaresCentered).get();

        NumericTable * meanTable      = result->get(mean).get();
        NumericTable * raw2MomTable   = result->get(secondOrderRawMoment).get();
        NumericTable * varianceTable  = result->get(variance).get();
        NumericTable * stDevTable     = result->get(standardDeviation).get();
        NumericTable * variationTable = result->get(variation).get();

        __DAAL_CALL_KERNEL(env, internal::LowOrderMomentsOnlineKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), finalizeCompute,
                           nObservationsTable, sumTable, sumSqTable, sumSqCenTable, meanTable, raw2MomTable, varianceTable, stDevTable,
                           variationTable, par);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    auto & context    = daal::services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    if (method != defaultDense || deviceInfo.isCpu)
    {
        __DAAL_INITIALIZE_KERNELS(internal::LowOrderMomentsDistributedKernel, algorithmFPType, method);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    PartialResult * partialResult                = static_cast<PartialResult *>(_pres);
    DistributedInput<step2Master> * input        = static_cast<DistributedInput<step2Master> *>(_in);
    data_management::DataCollection * collection = input->get(low_order_moments::partialResults).get();

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    // for other methods oneapi isn't implemented yet
    auto & context    = daal::services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    services::Status s;

    if (method != defaultDense || deviceInfo.isCpu)
    {
        s = __DAAL_CALL_KERNEL_STATUS(env, internal::LowOrderMomentsDistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                                      collection, partialResult, par);
    }

    collection->clear();
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    PartialResult * partialResult = static_cast<PartialResult *>(_pres);
    Result * result               = static_cast<Result *>(_res);

    Parameter * par                        = static_cast<Parameter *>(_par);
    daal::services::Environment::env & env = *_env;

    // for other methods oneapi isn't implemented yet
    auto & context    = daal::services::internal::getDefaultContext();
    auto & deviceInfo = context.getInfoDevice();

    result->set(minimum, partialResult->get(partialMinimum));
    result->set(maximum, partialResult->get(partialMaximum));
    result->set(sum, partialResult->get(partialSum));
    result->set(sumSquares, partialResult->get(partialSumSquares));
    result->set(sumSquaresCentered, partialResult->get(partialSumSquaresCentered));

    services::Status s;

    if (method != defaultDense || deviceInfo.isCpu)
    {
        NumericTable * nObservationsTable = partialResult->get(nObservations).get();
        NumericTable * sumTable           = partialResult->get(partialSum).get();
        NumericTable * sumSqTable         = partialResult->get(partialSumSquares).get();
        NumericTable * sumSqCenTable      = partialResult->get(partialSumSquaresCentered).get();

        NumericTable * meanTable      = result->get(mean).get();
        NumericTable * raw2MomTable   = result->get(secondOrderRawMoment).get();
        NumericTable * varianceTable  = result->get(variance).get();
        NumericTable * stDevTable     = result->get(standardDeviation).get();
        NumericTable * variationTable = result->get(variation).get();

        s = __DAAL_CALL_KERNEL_STATUS(env, internal::LowOrderMomentsDistributedKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method),
                                      finalizeCompute, nObservationsTable, sumTable, sumSqTable, sumSqCenTable, meanTable, raw2MomTable,
                                      varianceTable, stDevTable, variationTable, par);
    }

    return s;
}

} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif

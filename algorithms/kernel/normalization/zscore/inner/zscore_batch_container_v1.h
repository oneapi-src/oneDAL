/* file: zscore_batch_container_v1.h */
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
//  Implementation of zscore normalization calculation algorithm container.
//--
*/

#include "zscore_v1.h"
#include "zscore_base.h"
#include "zscore_dense_default_kernel.h"
#include "zscore_dense_sum_kernel.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{

namespace interface1
{

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ZScoreKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);
    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;

    NumericTablePtr inputTable  = input->get(data);
    NumericTablePtr resultTable = result->get(normalizedData);

    if (method == defaultDense)
    {
        interface1::Parameter<algorithmFPType, defaultDense> *parameter = static_cast<interface1::Parameter<algorithmFPType, defaultDense> *>(par);

        interface3::Parameter<algorithmFPType, defaultDense> internalPar;
        internalPar.resultsToCompute = none;
        internalPar.doScale = true;
        internalPar.moments->input.set(low_order_moments::data, inputTable);
        __DAAL_CALL_KERNEL(env, internal::ZScoreKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, defaultDense), compute, *inputTable, *resultTable, internalPar);
    }
    else
    {
        interface1::Parameter<algorithmFPType, sumDense> *parameter = static_cast<interface1::Parameter<algorithmFPType, sumDense> *>(par);

        interface3::Parameter<algorithmFPType, sumDense> internalPar;
        internalPar.resultsToCompute = none;
        internalPar.doScale = true;

        __DAAL_CALL_KERNEL(env, internal::ZScoreKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, sumDense), compute, *inputTable, *resultTable, internalPar);
    }


}
} // interface 1

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

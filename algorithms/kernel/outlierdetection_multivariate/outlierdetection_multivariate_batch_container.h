/* file: outlierdetection_multivariate_batch_container.h */
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
//  Implementation of Outlier Detection algorithm container.
//--
*/

#include "outlier_detection_multivariate.h"
#include "outlierdetection_multivariate_kernel.h"

namespace daal
{
namespace algorithms
{
namespace multivariate_outlier_detection
{
namespace interface1
{

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::OutlierDetectionKernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    NumericTable &dataTable    = *(static_cast<NumericTable *>(input->get(data).get()));
    NumericTable &weightsTable = *(static_cast<NumericTable *>(result->get(weights).get()));

    NumericTable *locationTable  = static_cast<NumericTable *>(input->get(location).get());
    NumericTable *scatterTable   = static_cast<NumericTable *>(input->get(scatter).get());
    NumericTable *thresholdTable = static_cast<NumericTable *>(input->get(threshold).get());

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::OutlierDetectionKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, defaultDense), compute, dataTable, locationTable, scatterTable, thresholdTable, weightsTable);
}
}
} // namespace multivariate_outlier_detection

} // namespace algorithms

} // namespace daal

/* file: pca_transform_container.h */
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
//  Implementation of pca transformation algorithm container -- a class
//  that contains fast pca transformation kernels
//  for supported architectures.
//--
*/

#ifndef __PCA_TRANSFORM_CONTAINER_H__
#define __PCA_TRANSFORM_CONTAINER_H__

#include "pca_transform_batch.h"
#include "pca_transform_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{

template <typename algorithmFPType, transform::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TransformKernel, algorithmFPType, method);
}

template <typename algorithmFPType, transform::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, transform::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    bool hasTransform = input->get(dataForTransform).get() != NULL;
    NumericTable *pMeans = hasTransform ? input->get(dataForTransform, mean).get() : NULL;
    NumericTable *pVariances = hasTransform ? input->get(dataForTransform, variance).get() : NULL;
    NumericTable *pEigenvalues = hasTransform ? input->get(dataForTransform, eigenvalue).get() : NULL;

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::TransformKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,
                       *(input->get(data)),
                       *(input->get(eigenvectors)),
                       pMeans, pVariances, pEigenvalues, *(result->get(transformedData)));
}

} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif

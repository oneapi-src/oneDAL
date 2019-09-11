/* file: multiclassclassifier_predict_batch_container.h */
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
//  Implementation of Multi-class classifier prediction algorithm container --
//  a class that contains Multi-class classifier kernels for supported
//  architectures.
//--
*/

#include "multi_class_classifier_predict.h"
#include "multiclassclassifier_predict_kernel.h"
#include "multiclassclassifier_predict_mccwu_kernel.h"
#include "kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{

template<typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MultiClassClassifierPredictKernel, pmethod, tmethod, algorithmFPType);
}

template<typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
services::Status BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::compute()
{
    classifier::prediction::Input *input = static_cast<classifier::prediction::Input *>(_in);
    classifier::prediction::Result *result = static_cast<classifier::prediction::Result *>(_res);

    NumericTable *a = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    multi_class_classifier::Model *m = static_cast<multi_class_classifier::Model *>(input->get(classifier::prediction::model).get());
    NumericTable *r[1];
    r[0] = static_cast<NumericTable *>(result->get(classifier::prediction::prediction).get());

    daal::algorithms::Parameter *par = _par;
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::MultiClassClassifierPredictKernel,
                       __DAAL_KERNEL_ARGUMENTS(pmethod, tmethod,  algorithmFPType), compute, a, m, r[0], par);
}

} // namespace prediction

} // namespace multi_class_classifier

} // namespace algorithms

} // namespace daal

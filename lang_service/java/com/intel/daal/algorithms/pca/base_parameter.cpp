/* file: base_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include <jni.h>

#include "daal.h"
#include "com_intel_daal_algorithms_pca_Online.h"
#include "com_intel_daal_algorithms_pca_Method.h"
#include "com_intel_daal_algorithms_pca_BaseParameter.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;

#define CorrelationDenseValue com_intel_daal_algorithms_pca_Method_correlationDenseValue
#define SVDDenseValue         com_intel_daal_algorithms_pca_Method_svdDenseValue

#include "common_defines.i"

/*
 * Class:     com_intel_daal_algorithms_pca_BaseParameter
 * Method:    cSetCovariance
 * Signature: (JJJIIII)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_BaseParameter_cSetCovariance(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                       jlong covarianceAddr, jint method, jint cmode,
                                                                                       jint computeStep, jint prec)
{
    using namespace daal::algorithms::pca;

    if (method == CorrelationDenseValue)
    {
        if (cmode == jBatch)
        {
            if (prec == 0) //double
            {
                BatchParameter<double, correlationDense> * parameterAddr = (BatchParameter<double, correlationDense> *)parAddr;
                parameterAddr->covariance                                = *((SharedPtr<daal::algorithms::covariance::BatchImpl> *)covarianceAddr);
            }
            else
            {
                BatchParameter<float, correlationDense> * parameterAddr = (BatchParameter<float, correlationDense> *)parAddr;
                parameterAddr->covariance                               = *((SharedPtr<daal::algorithms::covariance::BatchImpl> *)covarianceAddr);
            }
        }
        else if (cmode == jOnline || (cmode == jDistributed && computeStep == jStep1Local))
        {
            if (prec == 0) //double
            {
                OnlineParameter<double, correlationDense> * parameterAddr = (OnlineParameter<double, correlationDense> *)parAddr;
                parameterAddr->covariance                                 = *((SharedPtr<daal::algorithms::covariance::OnlineImpl> *)covarianceAddr);
            }
            else
            {
                OnlineParameter<float, correlationDense> * parameterAddr = (OnlineParameter<float, correlationDense> *)parAddr;
                parameterAddr->covariance                                = *((SharedPtr<daal::algorithms::covariance::OnlineImpl> *)covarianceAddr);
            }
        }
        else if (cmode == jDistributed)
        {
            if (prec == 0) //double
            {
                DistributedParameter<step2Master, double, correlationDense> * parameterAddr =
                    (DistributedParameter<step2Master, double, correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::DistributedIface<step2Master> > *)covarianceAddr);
            }
            else
            {
                DistributedParameter<step2Master, float, correlationDense> * parameterAddr =
                    (DistributedParameter<step2Master, float, correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::DistributedIface<step2Master> > *)covarianceAddr);
            }
        }
    }
}

/* file: base_parameter.cpp */
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
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_BaseParameter_cSetCovariance
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong covarianceAddr, jint method, jint cmode, jint computeStep, jint prec)
{
    using namespace daal::algorithms::pca;

    if(method == CorrelationDenseValue)
    {
        if(cmode == jBatch)
        {
            if(prec == 0) //double
            {
                BatchParameter<double, correlationDense> *parameterAddr = (BatchParameter<double, correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::BatchImpl> *)covarianceAddr);
            }
            else
            {
                BatchParameter<float, correlationDense> *parameterAddr = (BatchParameter<float, correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::BatchImpl> *)covarianceAddr);
            }
        }
        else if(cmode == jOnline || (cmode == jDistributed && computeStep == jStep1Local))
        {
            if(prec == 0) //double
            {
                OnlineParameter<double, correlationDense> *parameterAddr = (OnlineParameter<double, correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::OnlineImpl> *)covarianceAddr);
            }
            else
            {
                OnlineParameter<float, correlationDense> *parameterAddr = (OnlineParameter<float, correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::OnlineImpl> *)covarianceAddr);
            }
        }
        else if(cmode == jDistributed)
        {
            if(prec == 0) //double
            {
                DistributedParameter<step2Master, double, correlationDense> *parameterAddr =
                    (DistributedParameter<step2Master, double, correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::DistributedIface<step2Master> > *)covarianceAddr);
            }
            else
            {
                DistributedParameter<step2Master, float, correlationDense> *parameterAddr =
                    (DistributedParameter<step2Master, float, correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::DistributedIface<step2Master> > *)covarianceAddr);
            }
        }
    }
}

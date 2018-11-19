/* file: base_parameter.cpp */
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

#include <jni.h>

#include "daal.h"
#include "pca/JOnline.h"
#include "pca/JMethod.h"
#include "pca/JBaseParameter.h"
#include "JComputeMode.h"
#include "JComputeStep.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;

#define CorrelationDenseValue com_intel_daal_algorithms_pca_Method_correlationDenseValue
#define SVDDenseValue         com_intel_daal_algorithms_pca_Method_svdDenseValue

#define batchValue com_intel_daal_algorithms_ComputeMode_batchValue
#define onlineValue com_intel_daal_algorithms_ComputeMode_onlineValue
#define distributedValue com_intel_daal_algorithms_ComputeMode_distributedValue

#define step1Value com_intel_daal_algorithms_ComputeStep_step1LocalValue
#define step2Value com_intel_daal_algorithms_ComputeStep_step2MasterValue
#define step3Value com_intel_daal_algorithms_ComputeStep_step3LocalValue

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
        if(cmode == batchValue)
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
        else if(cmode == onlineValue || (cmode == distributedValue && computeStep == step1Value))
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
        else if(cmode == distributedValue)
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

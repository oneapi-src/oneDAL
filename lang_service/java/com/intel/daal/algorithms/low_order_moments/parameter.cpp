/* file: parameter.cpp */
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
#include "com_intel_daal_algorithms_low_order_moments_Parameter.h"
#include "common_helpers.h"

#include "com_intel_daal_algorithms_low_order_moments_EstimatesToCompute.h"
#define EstimatesAll          com_intel_daal_algorithms_low_order_moments_EstimatesToCompute_EstimatesAll
#define EstimatesMinMax       com_intel_daal_algorithms_low_order_moments_EstimatesToCompute_EstimatesMinMax
#define EstimatesMeanVariance com_intel_daal_algorithms_low_order_moments_EstimatesToCompute_EstimatesMeanVariance

USING_COMMON_NAMESPACES()
using namespace daal;

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Parameter
 * Method:    cSetEstimatesToCompute
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_Parameter_cSetEstimatesToCompute(JNIEnv * env, jobject thisObj,
                                                                                                           jlong parAddr, jint estComp)
{
    using namespace daal::algorithms;
    low_order_moments::Parameter * parameterAddr = (low_order_moments::Parameter *)parAddr;

    if (estComp == EstimatesAll)
    {
        parameterAddr->estimatesToCompute = low_order_moments::estimatesAll;
    }
    else if (estComp == EstimatesMinMax)
    {
        parameterAddr->estimatesToCompute = low_order_moments::estimatesMinMax;
    }
    else if (estComp == EstimatesMeanVariance)
    {
        parameterAddr->estimatesToCompute = low_order_moments::estimatesMeanVariance;
    }
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Parameter
 * Method:    cGetEstimatesToCompute
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_Parameter_cGetEstimatesToCompute(JNIEnv * env, jobject thisObj,
                                                                                                           jlong parAddr)
{
    using namespace daal::algorithms;
    low_order_moments::Parameter * parameterAddr = (low_order_moments::Parameter *)parAddr;

    return (jint)(parameterAddr->estimatesToCompute);
}

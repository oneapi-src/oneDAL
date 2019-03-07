/* file: parameter.cpp */
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
#include "low_order_moments/JParameter.h"
#include "low_order_moments/JEstimatesToCompute.h"
#include "common_helpers.h"

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
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_Parameter_cSetEstimatesToCompute
  (JNIEnv *env, jobject thisObj, jlong parAddr, jint estComp)
{
    using namespace daal::algorithms;
    low_order_moments::Parameter *parameterAddr = (low_order_moments::Parameter *)parAddr;

    if(estComp == EstimatesAll)
    {
        parameterAddr->estimatesToCompute = low_order_moments::estimatesAll;
    }
    else if(estComp == EstimatesMinMax)
    {
        parameterAddr->estimatesToCompute = low_order_moments::estimatesMinMax;
    }
    else if(estComp == EstimatesMeanVariance)
    {
        parameterAddr->estimatesToCompute = low_order_moments::estimatesMeanVariance;
    }
}

/*
 * Class:     com_intel_daal_algorithms_low_order_moments_Parameter
 * Method:    cGetEstimatesToCompute
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_low_1order_1moments_Parameter_cGetEstimatesToCompute
  (JNIEnv *env, jobject thisObj, jlong parAddr)
{
    using namespace daal::algorithms;
    low_order_moments::Parameter *parameterAddr = (low_order_moments::Parameter *)parAddr;

    return (jint)(parameterAddr->estimatesToCompute);

}

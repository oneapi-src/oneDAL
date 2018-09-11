/* file: bacon_parameter.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_bacon_outlier_detection_Parameter */

#include "daal.h"
#include "bacon_outlier_detection/JParameter.h"
#include "bacon_outlier_detection/JInitializationMethod.h"

#define baconMedianValue        com_intel_daal_algorithms_bacon_outlier_detection_InitializationMethod_baconMedianValue
#define baconMahalanobisValue   com_intel_daal_algorithms_bacon_outlier_detection_InitializationMethod_baconMahalanobisValue

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cSetInitializationMethod
 * Signature:(JI)I
 */
JNIEXPORT void JNICALL
Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cSetInitializationMethod
(JNIEnv *env, jobject thisObj, jlong parAddr, jint initMethodValue)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    ((Parameter *)parAddr)->initMethod = (InitializationMethod)initMethodValue;
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cGetInitializationMethod
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL
Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cGetInitializationMethod
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    return(int)(((Parameter *)parAddr)->initMethod);
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cSetAlpha
 * Signature:(JD)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cSetAlpha
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble alpha)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    ((Parameter *)parAddr)->alpha = alpha;
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cGetAlpha
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cGetAlpha
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    return((Parameter *)parAddr)->alpha;
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cSetToleranceToConverge
 * Signature:(JD)I
 */
JNIEXPORT void JNICALL
Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cSetToleranceToConverge
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble tol)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    ((Parameter *)parAddr)->toleranceToConverge = tol;
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cGetToleranceToConverge
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL
Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cGetToleranceToConverge
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    return((Parameter *)parAddr)->toleranceToConverge;
}

/* file: bacon_parameter.cpp */
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_bacon_outlier_detection_Parameter */

#include "daal.h"
#include "com_intel_daal_algorithms_bacon_outlier_detection_Parameter.h"

#define baconMedianValue      com_intel_daal_algorithms_bacon_outlier_detection_InitializationMethod_baconMedianValue
#define baconMahalanobisValue com_intel_daal_algorithms_bacon_outlier_detection_InitializationMethod_baconMahalanobisValue

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cSetInitializationMethod
 * Signature:(JI)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cSetInitializationMethod(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong parAddr,
                                                                                                                   jint initMethodValue)
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
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cGetInitializationMethod(JNIEnv * env, jobject thisObj,
                                                                                                                   jlong parAddr)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    return (int)(((Parameter *)parAddr)->initMethod);
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cSetAlpha
 * Signature:(JD)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cSetAlpha(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                    jdouble alpha)
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
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cGetAlpha(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    return ((Parameter *)parAddr)->alpha;
}

/*
 * Class:     com_intel_daal_algorithms_bacon_outlier_detection_Parameter
 * Method:    cSetToleranceToConverge
 * Signature:(JD)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cSetToleranceToConverge(JNIEnv * env, jobject thisObj,
                                                                                                                  jlong parAddr, jdouble tol)
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
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_bacon_1outlier_1detection_Parameter_cGetToleranceToConverge(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong parAddr)
{
    using namespace daal;
    using namespace daal::algorithms::bacon_outlier_detection;
    return ((Parameter *)parAddr)->toleranceToConverge;
}

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

#include "com_intel_daal_algorithms_implicit_als_Parameter.h"

using namespace daal;
using namespace daal::algorithms::implicit_als;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetNFactors
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetNFactors(JNIEnv *, jobject, jlong parAddr, jlong nFactors)
{
    ((Parameter *)parAddr)->nFactors = nFactors;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetNFactors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetNFactors(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->nFactors;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetMaxIterations
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetMaxIterations(JNIEnv *, jobject, jlong parAddr, jlong maxIterations)
{
    ((Parameter *)parAddr)->maxIterations = maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetMaxIterations
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetMaxIterations(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetAlpha
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetAlpha(JNIEnv *, jobject, jlong parAddr, jdouble alpha)
{
    ((Parameter *)parAddr)->alpha = alpha;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetAlpha
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetAlpha(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->alpha;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetLambda
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetLambda(JNIEnv *, jobject, jlong parAddr, jdouble lambda)
{
    ((Parameter *)parAddr)->lambda = lambda;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetLambda
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetLambda(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->lambda;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetPreferenceThreshold
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetPreferenceThreshold(JNIEnv *, jobject, jlong parAddr,
                                                                                                      jdouble preferenceThreshold)
{
    ((Parameter *)parAddr)->preferenceThreshold = preferenceThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetPreferenceThreshold
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetPreferenceThreshold(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->preferenceThreshold;
}

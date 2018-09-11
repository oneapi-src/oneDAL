/* file: parameter.cpp */
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

#include "implicit_als/JParameter.h"

using namespace daal;
using namespace daal::algorithms::implicit_als;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetNFactors
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetNFactors
(JNIEnv *, jobject, jlong parAddr, jlong nFactors)
{
    ((Parameter *)parAddr)->nFactors = nFactors;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetNFactors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetNFactors
(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->nFactors;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetMaxIterations
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetMaxIterations
(JNIEnv *, jobject, jlong parAddr, jlong maxIterations)
{
    ((Parameter *)parAddr)->maxIterations = maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetMaxIterations
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetMaxIterations
(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetAlpha
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetAlpha
(JNIEnv *, jobject, jlong parAddr, jdouble alpha)
{
    ((Parameter *)parAddr)->alpha = alpha;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetAlpha
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetAlpha
(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->alpha;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetLambda
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetLambda
(JNIEnv *, jobject, jlong parAddr, jdouble lambda)
{
    ((Parameter *)parAddr)->lambda = lambda;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetLambda
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetLambda
(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->lambda;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cSetPreferenceThreshold
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cSetPreferenceThreshold
(JNIEnv *, jobject, jlong parAddr, jdouble preferenceThreshold)
{
    ((Parameter *)parAddr)->preferenceThreshold = preferenceThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_Parameter
 * Method:    cGetPreferenceThreshold
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_implicit_1als_Parameter_cGetPreferenceThreshold
(JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->preferenceThreshold;
}

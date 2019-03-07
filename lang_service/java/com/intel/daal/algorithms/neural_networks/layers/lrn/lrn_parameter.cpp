/* file: lrn_parameter.cpp */
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
#include "neural_networks/layers/lrn/JLrnParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new lrn::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cGetDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cGetDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    NumericTablePtr *ntShPtr = new NumericTablePtr();
    *ntShPtr = (((lrn::Parameter *)cParameter))->dimension;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cSetDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong dimension)
{
    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)dimension;
    (((lrn::Parameter *)cParameter))->dimension = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cGetkappa
 * Signature: (J)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cGetkappa
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((lrn::Parameter *)cParameter))->kappa);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cSetKappa
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cSetKappa
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong kappa)
{
    (((lrn::Parameter *)cParameter))->kappa = (size_t)kappa;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cGetAlpha
 * Signature: (J)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cGetAlpha
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((lrn::Parameter *)cParameter))->alpha);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cSetAlpha
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cSetAlpha
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong alpha)
{
    (((lrn::Parameter *)cParameter))->alpha = alpha;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cGetBeta
 * Signature: (J)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cGetBeta
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((lrn::Parameter *)cParameter))->beta);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cSetBeta
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cSetBeta
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong beta)
{
    (((lrn::Parameter *)cParameter))->beta = beta;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cGetNAdjust
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cGetNAdjust
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((lrn::Parameter *)cParameter))->nAdjust);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_LrnParameter
 * Method:    cSetNAdjust
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_LrnParameter_cSetNAdjust
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong nAdjust)
{
    (((lrn::Parameter *)cParameter))->nAdjust = (size_t)nAdjust;
}

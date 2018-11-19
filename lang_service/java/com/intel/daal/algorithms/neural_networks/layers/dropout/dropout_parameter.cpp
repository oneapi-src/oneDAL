/* file: dropout_parameter.cpp */
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
#include "neural_networks/layers/dropout/JDropoutParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutParameter_cInit
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new dropout::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutParameter
 * Method:    cGetRetainRatio
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutParameter_cGetRetainRatio
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((dropout::Parameter *)cParameter))->retainRatio;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutParameter
 * Method:    cSetRetainRatio
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutParameter_cSetRetainRatio
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble retainRatio)
{
    (((dropout::Parameter *)cParameter))->retainRatio = retainRatio;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutParameter
 * Method:    cGetSeed
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutParameter_cGetSeed
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((dropout::Parameter *)cParameter))->seed);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutParameter
 * Method:    cSetSeed
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutParameter_cSetSeed
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong seed)
{
    (((dropout::Parameter *)cParameter))->seed = (size_t)seed;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutParameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutParameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((dropout::Parameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}

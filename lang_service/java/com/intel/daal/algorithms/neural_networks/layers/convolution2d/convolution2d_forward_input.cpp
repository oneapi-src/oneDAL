/* file: convolution2d_forward_input.cpp */
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
#include "neural_networks/layers/convolution2d/JConvolution2dForwardInput.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Convolution2dForwardInput
 * Method:    cGetWeightsSizes
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Convolution2dForwardInput_cGetWeightsSizes
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jlong paramAddr)
{
    Collection<size_t> dims = ((convolution2d::forward::Input *)inputAddr)->getWeightsSizes((convolution2d::Parameter *)paramAddr);
    return getJavaLongArrayFromSizeTCollection(env, dims);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Convolution2dForwardInput
 * Method:    cGetBiasesSizes
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Convolution2dForwardInput_cGetBiasesSizes
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jlong paramAddr)
{
    Collection<size_t> dims = ((convolution2d::forward::Input *)inputAddr)->getBiasesSizes((convolution2d::Parameter *)paramAddr);
    return getJavaLongArrayFromSizeTCollection(env, dims);
}

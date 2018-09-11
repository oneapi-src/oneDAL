/* file: spatial_pooling2d_parameter.cpp */
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
#include "neural_networks/layers/spatial_pooling2d/JSpatialPooling2dParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_SpatialPooling2dParameter
 * Method:    cSetIndices
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1pooling2d_SpatialPooling2dParameter_cSetIndices
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((spatial_pooling2d::Parameter *)cParameter))->indices.size[0] = first;
    (((spatial_pooling2d::Parameter *)cParameter))->indices.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_SpatialPooling2dParameter
 * Method:    cSetPyramidHeight
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1pooling2d_SpatialPooling2dParameter_cSetPyramidHeight
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong pyramidHeight)
{
    (((spatial_pooling2d::Parameter *)cParameter))->pyramidHeight = pyramidHeight;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_SpatialPooling2dParameter
 * Method:    cGetPyramidHeight
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1pooling2d_SpatialPooling2dParameter_cGetPyramidHeight
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((spatial_pooling2d::Parameter *)cParameter))->pyramidHeight);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_pooling2d_SpatialPooling2dParameter
 * Method:    cGetIndices
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_spatial_1pooling2d_SpatialPooling2dParameter_cGetIndices
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((spatial_pooling2d::Parameter *)cParameter))->indices.size[0]);
    tmp[1] = (jlong)((((spatial_pooling2d::Parameter *)cParameter))->indices.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

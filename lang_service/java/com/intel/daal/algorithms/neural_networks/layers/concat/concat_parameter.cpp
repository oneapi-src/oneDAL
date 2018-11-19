/* file: concat_parameter.cpp */
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
#include "neural_networks/layers/concat/JConcatParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_concat_ConcatParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_concat_ConcatParameter_cInit
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new concat::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_concat_ConcatParameter
 * Method:    cGetConcatDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_concat_ConcatParameter_cGetConcatDimension
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((concat::Parameter *)cParameter))->concatDimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_concat_ConcatParameter
 * Method:    cSetConcatDimension
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_concat_ConcatParameter_cSetConcatDimension
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong concatDimension)
{
    (((concat::Parameter *)cParameter))->concatDimension = (size_t)concatDimension;
}

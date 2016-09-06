/* file: prelu_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
#include "neural_networks/layers/prelu/JPreluParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new prelu::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cGetDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_cGetDataDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((prelu::Parameter *)cParameter))->dataDimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_cSetDataDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong dataDimension)
{
    (((prelu::Parameter *)cParameter))->dataDimension = (size_t)dataDimension;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cGetDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_cgetWeightsDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((prelu::Parameter *)cParameter))->weightsDimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_prelu_PreluParameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_prelu_PreluParameter_csetWeightsDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong weightsDimension)
{
    (((prelu::Parameter *)cParameter))->weightsDimension = (size_t)weightsDimension;
}

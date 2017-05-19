/* file: reshape_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "neural_networks/layers/reshape/JReshapeParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();

using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_reshape_ReshapeParameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new reshape::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeParameter
 * Method:    cSetReshapeDimensions
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_reshape_ReshapeParameter_cSetReshapeDimensions
(JNIEnv *env, jobject thisObj, jlong cParameter, jlongArray jArray)
{
    jint size = env->GetArrayLength(jArray);
    jlong* jarr;
    jboolean jniNoCopy = JNI_FALSE;
    jarr = env->GetLongArrayElements(jArray, &jniNoCopy);

    (((reshape::Parameter *)cParameter))->reshapeDimensions.clear();
    for(int i=0; i<size; i++)
    {
        (((reshape::Parameter *)cParameter))->reshapeDimensions.push_back( (size_t)jarr[i] );
    }

    env->ReleaseLongArrayElements(jArray,jarr,0);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeParameter
 * Method:    cGetReshapeDimensions
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_reshape_ReshapeParameter_cGetReshapeDimensions
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    size_t size = (((reshape::Parameter *)cParameter))->reshapeDimensions.size();
    jlongArray sizeArray = env->NewLongArray( size );

    jlong* tmp = (jlong*)daal_malloc(size*sizeof(jlong));

    if(!tmp)
    {
        Error e(services::ErrorMemoryAllocationFailed);
        const char *description = e.description();
        env->ThrowNew(env->FindClass("java/lang/Exception"),description);
        return sizeArray;
    }

    for(int i=0; i<size; i++)
    {
        tmp[i] = (((reshape::Parameter *)cParameter))->reshapeDimensions[i];
    }

    env->SetLongArrayRegion(sizeArray, 0, size, tmp);

    daal_free(tmp);

    return sizeArray;
}

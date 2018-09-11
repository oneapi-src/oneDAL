/* file: reshape_parameter.cpp */
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

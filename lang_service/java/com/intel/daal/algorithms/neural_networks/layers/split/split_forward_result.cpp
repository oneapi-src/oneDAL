/* file: split_forward_result.cpp */
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
#include "neural_networks/layers/split/JSplitForwardResult.h"
#include "neural_networks/layers/split/JSplitForwardResultLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define valueCollectionId com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResultLayerDataId_valueCollectionId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::split;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitForwardResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<forward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitForwardResult_cGetValue__JI
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == valueCollectionId)
    {
        return jniArgument<forward::Result>::get<forward::ResultLayerDataId, KeyValueDataCollection>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResult
 * Method:    cGetValue
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitForwardResult_cGetValue__JIJ
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong index)
{
    if (id == valueCollectionId)
    {
        return jniArgument<forward::Result>::get<forward::ResultLayerDataId, Tensor>(resAddr, id, (size_t)index);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResult
 * Method:    cSetValue
 * Signature: (JIJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitForwardResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr, jlong index)
{
    if (id == valueCollectionId)
    {
        jniArgument<forward::Result>::set<forward::ResultLayerDataId, Tensor>(resAddr, id, ntAddr, (size_t)index);
    }
}

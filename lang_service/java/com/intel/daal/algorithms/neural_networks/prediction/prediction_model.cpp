/* file: prediction_model.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_neural_networks_prediction_PredictionModel.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cInit__
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new prediction::ModelPtr(new prediction::Model()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cInit
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cInit__J
(JNIEnv *env, jobject thisObj, jlong cModel)
{
    prediction::ModelPtr model = *((prediction::ModelPtr *)cModel);
    return (jlong)(new prediction::ModelPtr(new prediction::Model(*model)));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cInit
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cInit__JJ
  (JNIEnv *env, jobject thisObj, jlong forwardLayersAddr, jlong nextLayersCollectionAddr)
{
    ForwardLayersPtr forwardLayers = *((ForwardLayersPtr *)forwardLayersAddr);
    SharedPtr<Collection<layers::NextLayers> > nextLayersCollection = *((SharedPtr<Collection<layers::NextLayers> > *)nextLayersCollectionAddr);
    return (jlong)(new prediction::ModelPtr(new prediction::Model(forwardLayers, nextLayersCollection)));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cInitFromLayerDescriptors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cInitFromPredictionTopology
  (JNIEnv *env, jobject thisObj, jlong topAddr)
{
    prediction::TopologyPtr ptr = *(prediction::TopologyPtr *)topAddr;
    return (jlong)(new prediction::ModelPtr(new prediction::Model(*ptr)));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cAllocate
 * Signature: (JI[JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cAllocate
  (JNIEnv *env, jobject thisObj, jlong cModel, jint prec, jlongArray dataSizeArray, jlong parameterAddr)
{
    size_t len = (size_t)(env->GetArrayLength(dataSizeArray));
    jlong *dataSize = env->GetLongArrayElements(dataSizeArray, 0);

    Collection<size_t> dataSizeCollection;

    for (size_t i = 0; i < len; i++)
    {
        dataSizeCollection.push_back((size_t)dataSize[i]);
    }

    env->ReleaseLongArrayElements(dataSizeArray, dataSize, JNI_ABORT);

    prediction::ModelPtr model = *((prediction::ModelPtr *)cModel);

    if (prec == 0)
    {
        model->allocate<double>(dataSizeCollection, (prediction::Parameter *)parameterAddr);
    }
    else
    {
        model->allocate<float>(dataSizeCollection, (prediction::Parameter *)parameterAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cSetLayers
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cSetLayers
  (JNIEnv *env, jobject thisObj, jlong cModel, jlong forwardLayersAddr, jlong nextLayersCollectionAddr)
{
    prediction::ModelPtr model = *((prediction::ModelPtr *)cModel);
    ForwardLayersPtr forwardLayers = *((ForwardLayersPtr *)forwardLayersAddr);
    SharedPtr<Collection<layers::NextLayers> > nextLayersCollection = *((SharedPtr<Collection<layers::NextLayers> > *)nextLayersCollectionAddr);
    model->setLayers(forwardLayers, nextLayersCollection);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cGetForwardLayers
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cGetForwardLayers
(JNIEnv *env, jobject thisObj, jlong cModel)
{
    prediction::ModelPtr model = *((prediction::ModelPtr *)cModel);
    ForwardLayersPtr *forwardLayersAddr = new ForwardLayersPtr(model->getLayers());
    return (jlong) forwardLayersAddr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cGetForwardLayer
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cGetForwardLayer
  (JNIEnv *env, jobject thisObj, jlong cModel, jlong index)
{
    prediction::ModelPtr model = *((prediction::ModelPtr *)cModel);
    layers::forward::LayerIfacePtr *forwardLayerAddr = new layers::forward::LayerIfacePtr(model->getLayer(index));
    return (jlong) forwardLayerAddr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cGetForwardLayers
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cGetNextLayers
(JNIEnv *env, jobject thisObj, jlong cModel)
{
    prediction::ModelPtr model = *((prediction::ModelPtr *)cModel);
    SharedPtr<Collection<layers::NextLayers> > *nextLayersCollectionAddr = new SharedPtr<Collection<layers::NextLayers> >(model->getNextLayers());
    return (jlong) nextLayersCollectionAddr;
}

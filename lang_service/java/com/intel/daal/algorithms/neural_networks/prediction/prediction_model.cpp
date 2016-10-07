/* file: prediction_model.cpp */
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
#include "neural_networks/prediction/JPredictionModel.h"

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
    return (jlong)(new SharedPtr<prediction::Model>(new prediction::Model()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cInit
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cInit__J
(JNIEnv *env, jobject thisObj, jlong cModel)
{
    SharedPtr<prediction::Model> model = *((SharedPtr<prediction::Model> *)cModel);
    return (jlong)(new SharedPtr<prediction::Model>(new prediction::Model(*model)));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionModel
 * Method:    cInit
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionModel_cInit__JJ
  (JNIEnv *env, jobject thisObj, jlong forwardLayersAddr, jlong nextLayersCollectionAddr)
{
    SharedPtr<ForwardLayers> forwardLayers = *((SharedPtr<ForwardLayers> *)forwardLayersAddr);
    SharedPtr<Collection<layers::NextLayers> > nextLayersCollection = *((SharedPtr<Collection<layers::NextLayers> > *)nextLayersCollectionAddr);
    return (jlong)(new SharedPtr<prediction::Model>(new prediction::Model(forwardLayers, nextLayersCollection)));
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
    return (jlong)(new SharedPtr<prediction::Model>(new prediction::Model(*ptr)));
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

    SharedPtr<prediction::Model> model = *((SharedPtr<prediction::Model> *)cModel);

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
    SharedPtr<prediction::Model> model = *((SharedPtr<prediction::Model> *)cModel);
    SharedPtr<ForwardLayers> forwardLayers = *((SharedPtr<ForwardLayers> *)forwardLayersAddr);
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
    SharedPtr<prediction::Model> model = *((SharedPtr<prediction::Model> *)cModel);
    SharedPtr<ForwardLayers> *forwardLayersAddr = new SharedPtr<ForwardLayers>(model->getLayers());
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
    SharedPtr<prediction::Model> model = *((SharedPtr<prediction::Model> *)cModel);
    SharedPtr<layers::forward::LayerIface> *forwardLayerAddr = new SharedPtr<layers::forward::LayerIface>(model->getLayer(index));
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
    SharedPtr<prediction::Model> model = *((SharedPtr<prediction::Model> *)cModel);
    SharedPtr<Collection<layers::NextLayers> > *nextLayersCollectionAddr = new SharedPtr<Collection<layers::NextLayers> >(model->getNextLayers());
    return (jlong) nextLayersCollectionAddr;
}

/* file: training_model.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_training_TrainingModel.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cInit__
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new training::ModelPtr(new training::Model()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cInit
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cInit__J
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    training::ModelPtr model = *((training::ModelPtr *)modelAddr);
    return (jlong)(new training::ModelPtr(new training::Model(*model)));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cInitialize
 * Signature: (JI[JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cInitialize
  (JNIEnv *env, jobject thisObj, jlong cModel, jint prec, jlongArray dataSizeArray, jlong topologyAddr)
{
    training::TopologyPtr topology = *(training::TopologyPtr *)topologyAddr;

    size_t len = (size_t)(env->GetArrayLength(dataSizeArray));
    jlong *dataSize = env->GetLongArrayElements(dataSizeArray, 0);

    Collection<size_t> dataSizeCollection;

    for (size_t i = 0; i < len; i++) {
        dataSizeCollection.push_back((size_t)dataSize[i]);
    }

    env->ReleaseLongArrayElements(dataSizeArray, dataSize, JNI_ABORT);

    training::ModelPtr model = *((training::ModelPtr *)cModel);

    if (prec == 0)
    {
        model->initialize<double>(dataSizeCollection, *topology);
    }
    else
    {
        model->initialize<float>(dataSizeCollection, *topology);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cGetForwardLayers
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cGetForwardLayers
(JNIEnv *env, jobject thisObj, jlong cModel)
{
    training::ModelPtr model = *((training::ModelPtr *)cModel);
    ForwardLayersPtr *forwardLayersAddr = new ForwardLayersPtr(model->getForwardLayers());
    return (jlong) forwardLayersAddr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cGetForwardLayer
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cGetForwardLayer
(JNIEnv *env, jobject thisObj, jlong cModel, jlong index)
{
    training::ModelPtr model = *((training::ModelPtr *)cModel);
    layers::forward::LayerIfacePtr *forwardLayerAddr =
        new layers::forward::LayerIfacePtr(model->getForwardLayer((size_t)index));
    return (jlong) forwardLayerAddr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cGetBackwardLayers
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cGetBackwardLayers
(JNIEnv *env, jobject thisObj, jlong cModel)
{
    training::ModelPtr model = *((training::ModelPtr *)cModel);
    BackwardLayersPtr *backwardLayersAddr = new BackwardLayersPtr(model->getBackwardLayers());
    return (jlong) backwardLayersAddr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cGetBackwardLayer
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cGetBackwardLayer
(JNIEnv *env, jobject thisObj, jlong cModel, jlong index)
{
    training::ModelPtr model = *((training::ModelPtr *)cModel);
    layers::backward::LayerIfacePtr *backwardLayerAddr =
        new layers::backward::LayerIfacePtr(model->getBackwardLayer((size_t)index));
    return (jlong) backwardLayerAddr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cGetNextLayers
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cGetNextLayers
(JNIEnv *env, jobject thisObj, jlong cModel)
{
    training::ModelPtr model = *((training::ModelPtr *)cModel);
    SharedPtr<Collection<layers::NextLayers> > *nextLayersCollectionAddr = new SharedPtr<Collection<layers::NextLayers> >(model->getNextLayers());
    return (jlong) nextLayersCollectionAddr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cGetPredictionModel
 * Signature: (IJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cGetPredictionModel
(JNIEnv *env, jobject thisObj, jint prec, jlong cModel)
{
    training::ModelPtr model = *((training::ModelPtr *)cModel);
    if (prec == 0)
    {
        return (jlong) (new prediction::ModelPtr(model->getPredictionModel<double>()));
    }
    else
    {
        return (jlong) (new prediction::ModelPtr(model->getPredictionModel<float>()));
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cGetWeightsAndBiases
 * Signature: (IJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cGetWeightsAndBiases
(JNIEnv *env, jobject thisObj, jlong cModel)
{
    training::ModelPtr model = *((training::ModelPtr *)cModel);

    NumericTablePtr *nt = new NumericTablePtr();

    *nt = model->getWeightsAndBiases();

    return (jlong)nt;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingModel
 * Method:    cSetWeightsAndBiases
 * Signature: (IJ)J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingModel_cSetWeightsAndBiases
(JNIEnv *env, jobject thisObj, jlong cModel, jlong ntAddr)
{
    training::ModelPtr model = *((training::ModelPtr *)cModel);

    SerializationIfacePtr *ntShPtr = (SerializationIfacePtr *)ntAddr;
    model->setWeightsAndBiases(staticPointerCast<NumericTable, SerializationIface>(*ntShPtr));
}

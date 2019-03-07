/* file: training_batch.cpp */
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
#include "neural_networks/training/JTrainingBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingBatch_cInit
  (JNIEnv *env, jobject thisObj, jint prec, jint method, jlong optAddr)
{
    services::SharedPtr<optimization_solver::iterative_solver::Batch > opt =
        *((services::SharedPtr<optimization_solver::iterative_solver::Batch > *)optAddr);
    return jniBatch<training::Method, training::Batch, training::defaultDense>::newObj(prec, method, opt);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingBatch_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<training::Method, training::Batch, training::defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingBatch_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<training::Method, training::Batch, training::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingBatch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<training::Method, training::Batch, training::defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingBatch_cSetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniBatch<training::Method, training::Batch, training::defaultDense>::
        setResult<training::Result>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingBatch
 * Method:    cInitialize
 * Signature: (JII[JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingBatch_cInitialize
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlongArray dataSizeArray, jlong topologyAddr)
{
    training::TopologyPtr topology = *(training::TopologyPtr *)topologyAddr;

    size_t len = (size_t)(env->GetArrayLength(dataSizeArray));
    jlong *dataSize = env->GetLongArrayElements(dataSizeArray, 0);

    Collection<size_t> dataSizeCollection;

    for (size_t i = 0; i < len; i++)
    {
        dataSizeCollection.push_back((size_t)dataSize[i]);
    }

    if (prec == 0)
    {
        SharedPtr<training::Batch<double, training::defaultDense>> alg =
            staticPointerCast<training::Batch<double, training::defaultDense>, AlgorithmIface>
                (*(SharedPtr<AlgorithmIface> *)algAddr);
        alg->initialize(dataSizeCollection, *topology);
    }
    else
    {
        SharedPtr<training::Batch<float, training::defaultDense>> alg =
            staticPointerCast<training::Batch<float, training::defaultDense>, AlgorithmIface>
                (*(SharedPtr<AlgorithmIface> *)algAddr);
        alg->initialize(dataSizeCollection, *topology);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingBatch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<training::Method, training::Batch, training::defaultDense>::getClone(prec, method, algAddr);
}

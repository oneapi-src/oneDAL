/* file: distributed_step2_master.cpp */
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
#include "neural_networks/training/JDistributedStep2Master.h"

#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2Master
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2Master_cInit
  (JNIEnv *, jobject, jint prec, jint method, jlong optAddr)
{
    services::SharedPtr<optimization_solver::iterative_solver::Batch > opt =
        *((services::SharedPtr<optimization_solver::iterative_solver::Batch > *)optAddr);
    return jniDistributed<step2Master, training::Method, training::Distributed, training::defaultDense>::newObj(prec, method, opt);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2Master
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2Master_cInitParameter
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, training::Method, training::Distributed, training::defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2Master
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2Master_cGetInput
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, training::Method, training::Distributed, training::defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2Master
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2Master_cGetResult
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, training::Method, training::Distributed, training::defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2Master
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2Master_cGetPartialResult
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, training::Method, training::Distributed, training::defaultDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2Master
 * Method:    cSetPartialResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2Master_cSetPartialResult
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniDistributed<step2Master, training::Method, training::Distributed, training::defaultDense>::setPartialResult<training::DistributedPartialResult>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2Master
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2Master_cClone
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, training::Method, training::Distributed, training::defaultDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_DistributedStep2Master
 * Method:    cInitialize
 * Signature: (JII[JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_DistributedStep2Master_cInitialize
  (JNIEnv *env, jobject, jlong algAddr, jint prec, jint method, jlongArray dataSizeArray, jlong topologyAddr)
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
        SharedPtr<training::Distributed<step2Master, double, training::defaultDense>> alg =
            staticPointerCast<training::Distributed<step2Master, double, training::defaultDense>, AlgorithmIface>
                (*(SharedPtr<AlgorithmIface> *)algAddr);
        alg->initialize(dataSizeCollection, *topology);
    }
    else
    {
        SharedPtr<training::Distributed<step2Master, float, training::defaultDense>> alg =
            staticPointerCast<training::Distributed<step2Master, float, training::defaultDense>, AlgorithmIface>
                (*(SharedPtr<AlgorithmIface> *)algAddr);
        alg->initialize(dataSizeCollection, *topology);
    }
}

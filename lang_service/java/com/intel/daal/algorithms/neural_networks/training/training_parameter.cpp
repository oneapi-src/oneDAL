/* file: training_parameter.cpp */
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
#include "neural_networks/training/JTrainingParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new training::Parameter());
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cGetBatchSize
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cGetBatchSize
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (((training::Parameter *)cParameter))->batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cSetBatchSize
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cSetBatchSize
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong batchSize)
{
    (((training::Parameter *)cParameter))->batchSize = batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cSetOptimizationSolver
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cSetOptimizationSolver
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong optAddr)
{
    services::SharedPtr<optimization_solver::iterative_solver::Batch > opt =
        *((services::SharedPtr<optimization_solver::iterative_solver::Batch > *)optAddr);
    (((training::Parameter *)cParameter))->optimizationSolver = opt;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cGetOptimizationSolver
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cGetOptimizationSolver
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    SharedPtr<optimization_solver::iterative_solver::Batch > *opt =
        new SharedPtr<optimization_solver::iterative_solver::Batch >((((training::Parameter *)cParameter))->optimizationSolver);
    return (jlong)opt;
}

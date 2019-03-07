/* file: training_parameter.cpp */
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
(JNIEnv *env, jobject thisObj, jlong optAddr)
{
    services::SharedPtr<optimization_solver::iterative_solver::Batch > opt =
        *((services::SharedPtr<optimization_solver::iterative_solver::Batch > *)optAddr);
    return (jlong)(new training::Parameter(opt));
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

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((training::Parameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}

/* file: parameter.cpp */
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

#include "daal.h"

#include "optimization_solver/iterative_solver/JParameter.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Parameter
 * Method:    cSetFunction
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cSetFunction
(JNIEnv *, jobject, jlong parAddr, jlong cFunction)
{
    iterative_solver::Parameter *parameterAddr = (iterative_solver::Parameter *)parAddr;
    SharedPtr<optimization_solver::sum_of_functions::Batch> objectiveFunction =
        staticPointerCast<optimization_solver::sum_of_functions::Batch, AlgorithmIface>
        (*(SharedPtr<AlgorithmIface> *)cFunction);
    parameterAddr->function = objectiveFunction;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Parameter
 * Method:    cSetNIterations
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cSetNIterations
(JNIEnv *, jobject, jlong parAddr, jlong nIterations)
{
    ((iterative_solver::Parameter *)parAddr)->nIterations = nIterations;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Parameter
 * Method:    cGetNIterations
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cGetNIterations
(JNIEnv *, jobject, jlong parAddr)
{
    return ((iterative_solver::Parameter *)parAddr)->nIterations;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Parameter
 * Method:    cSetAccuracyThreshold
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cSetAccuracyThreshold
(JNIEnv *, jobject, jlong parAddr, jdouble accuracyThreshold)
{
    ((iterative_solver::Parameter *)parAddr)->accuracyThreshold = accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Parameter
 * Method:    cGetAccuracyThreshold
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cGetAccuracyThreshold
(JNIEnv *, jobject, jlong parAddr)
{
    return ((iterative_solver::Parameter *)parAddr)->accuracyThreshold;
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Parameter
* Method:    cSetOptionalResultRequired
* Signature: (JZ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cSetOptionalResultRequired
(JNIEnv *, jobject, jlong parAddr, jboolean flag)
{
    ((iterative_solver::Parameter *)parAddr)->optionalResultRequired = flag;
}

/*
* Class:     com_intel_daal_algorithms_optimization_solver_iterative_solver_Parameter
* Method:    cGetOptionalResultRequired
* Signature: (J)Z
*/
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cGetOptionalResultRequired
(JNIEnv *, jobject, jlong parAddr)
{
    return ((iterative_solver::Parameter *)parAddr)->optionalResultRequired;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cSetBatchSize
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cSetBatchSize
(JNIEnv *, jobject, jlong parAddr, jlong batchSize)
{
    ((iterative_solver::Parameter *)parAddr)->batchSize = batchSize;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_adagrad_Parameter
 * Method:    cGetBatchSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_iterative_1solver_Parameter_cGetBatchSize
(JNIEnv *, jobject, jlong parAddr)
{
    return ((iterative_solver::Parameter *)parAddr)->batchSize;
}

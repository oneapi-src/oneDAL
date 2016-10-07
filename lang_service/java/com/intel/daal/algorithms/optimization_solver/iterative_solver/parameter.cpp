/* file: parameter.cpp */
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

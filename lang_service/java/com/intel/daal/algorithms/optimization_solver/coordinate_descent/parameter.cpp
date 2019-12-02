/* file: parameter.cpp */
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

#include "daal.h"

#include "com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetSeed
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetSeed(JNIEnv *, jobject, jlong parAddr,
                                                                                                                  jlong seed)
{
    ((coordinate_descent::Parameter *)parAddr)->seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cGetSeed
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cGetSeed(JNIEnv *, jobject, jlong parAddr)
{
    return ((coordinate_descent::Parameter *)parAddr)->seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetSelection
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetSelection(JNIEnv *, jobject,
                                                                                                                       jlong parAddr, jint selection)
{
    ((coordinate_descent::Parameter *)parAddr)->selection = (coordinate_descent::SelectionStrategy)selection;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cGetSelection
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cGetSelection(JNIEnv *, jobject,
                                                                                                                       jlong parAddr)
{
    return (jint)(((coordinate_descent::Parameter *)parAddr)->selection);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetPositive
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetPositiveFlag(JNIEnv *, jobject,
                                                                                                                          jlong parAddr,
                                                                                                                          jboolean positive)
{
    ((coordinate_descent::Parameter *)parAddr)->positive = positive;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cGetSelectionFlag
 * Signature: (J)I
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cGetPositiveFlag(JNIEnv *, jobject,
                                                                                                                              jlong parAddr)
{
    return ((coordinate_descent::Parameter *)parAddr)->positive;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetSkipTheFirstComponentsFlag
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetSkipTheFirstComponentsFlag(
    JNIEnv *, jobject, jlong parAddr, jboolean skipTheFirstComponents)
{
    ((coordinate_descent::Parameter *)parAddr)->skipTheFirstComponents = skipTheFirstComponents;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cGetSkipTheFirstComponentsFlag
 * Signature: (J)I
 */
JNIEXPORT jboolean JNICALL
    Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cGetSkipTheFirstComponentsFlag(JNIEnv *, jobject, jlong parAddr)
{
    return ((coordinate_descent::Parameter *)parAddr)->skipTheFirstComponents;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetEngine(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong cParameter,
                                                                                                                    jlong engineAddr)
{
    (((coordinate_descent::Parameter *)cParameter))->engine =
        staticPointerCast<engines::BatchBase, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)engineAddr);
}

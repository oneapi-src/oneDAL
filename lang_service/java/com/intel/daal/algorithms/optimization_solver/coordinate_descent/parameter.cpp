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

#include "com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetSeed
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetSeed
(JNIEnv *, jobject, jlong parAddr, jlong seed)
{
    ((coordinate_descent::Parameter *)parAddr)->seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cGetSeed
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cGetSeed
(JNIEnv *, jobject, jlong parAddr)
{
    return ((coordinate_descent::Parameter *)parAddr)->seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetSelection
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetSelection
(JNIEnv *, jobject, jlong parAddr, jint selection)
{
    ((coordinate_descent::Parameter *)parAddr)->selection = (coordinate_descent::SelectionStrategy)selection;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cGetSelection
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cGetSelection
(JNIEnv *, jobject, jlong parAddr)
{
    return (jint)(((coordinate_descent::Parameter *)parAddr)->selection);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetPositive
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetPositiveFlag
(JNIEnv *, jobject, jlong parAddr, jboolean positive)
{
    ((coordinate_descent::Parameter *)parAddr)->positive = positive;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cGetSelectionFlag
 * Signature: (J)I
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cGetPositiveFlag
(JNIEnv *, jobject, jlong parAddr)
{
    return ((coordinate_descent::Parameter *)parAddr)->positive;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetSkipTheFirstComponentsFlag
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetSkipTheFirstComponentsFlag
(JNIEnv *, jobject, jlong parAddr, jboolean skipTheFirstComponents)
{
    ((coordinate_descent::Parameter *)parAddr)->skipTheFirstComponents = skipTheFirstComponents;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cGetSkipTheFirstComponentsFlag
 * Signature: (J)I
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cGetSkipTheFirstComponentsFlag
(JNIEnv *, jobject, jlong parAddr)
{
    return ((coordinate_descent::Parameter *)parAddr)->skipTheFirstComponents;
}


/*
 * Class:     com_intel_daal_algorithms_optimization_solver_coordinate_descent_Parameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_coordinate_1descent_Parameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((coordinate_descent::Parameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}

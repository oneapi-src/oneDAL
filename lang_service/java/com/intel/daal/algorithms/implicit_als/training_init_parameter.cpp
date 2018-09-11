/* file: training_init_parameter.cpp */
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

#include "daal.h"

#include "implicit_als/training/init/JInitParameter.h"

using namespace daal;
using namespace daal::algorithms::implicit_als::training::init;
using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cSetFullNUsers
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitParameter_cSetFullNUsers
  (JNIEnv *, jobject, jlong parAddr, jlong fullNUsers)
{
    ((Parameter *)parAddr)->fullNUsers = fullNUsers;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cGetFullNUsers
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitParameter_cGetFullNUsers
  (JNIEnv *, jobject, jlong parAddr)
{
    return (jlong)(((Parameter *)parAddr)->fullNUsers);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cSetNFactors
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitParameter_cSetNFactors
  (JNIEnv *, jobject, jlong parAddr, jlong nFactors)
{
    ((Parameter *)parAddr)->nFactors = nFactors;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cGetNFactors
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitParameter_cGetNFactors
  (JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->nFactors;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cSetSeed
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitParameter_cSetSeed
  (JNIEnv *, jobject, jlong parAddr, jlong seed)
{
    ((Parameter *)parAddr)->seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cGetSeed
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitParameter_cGetSeed
    (JNIEnv *, jobject, jlong parAddr)
{
    return ((Parameter *)parAddr)->seed;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitParameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitParameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((Parameter *)cParameter))->engine = staticPointerCast<algorithms::engines::BatchBase, algorithms::AlgorithmIface> (*(SharedPtr<algorithms::AlgorithmIface> *)engineAddr);
}

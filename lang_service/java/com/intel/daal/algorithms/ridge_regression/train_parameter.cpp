/* file: train_parameter.cpp */
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
#include "ridge_regression/JTrainParameter.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_TrainParameter
 * Method:    cSetRidgeParameters
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_ridge_1regression_TrainParameter_cSetRidgeParameters
(JNIEnv * env, jobject thisObj, jlong parAddr, jlong cRidgeParameters)
{
    SerializationIfacePtr * const ntShPtr = (SerializationIfacePtr *)cRidgeParameters;
    ((ridge_regression::TrainParameter *)parAddr)->ridgeParameters = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_TrainParameter
 * Method:    cGetRidgeParameters
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_TrainParameter_cGetRidgeParameters
(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    NumericTablePtr * const ntShPtr = new NumericTablePtr();
    *ntShPtr = ((ridge_regression::TrainParameter *)parAddr)->ridgeParameters;
    return (jlong)ntShPtr;
}

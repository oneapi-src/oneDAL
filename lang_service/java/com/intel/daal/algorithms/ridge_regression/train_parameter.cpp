/* file: train_parameter.cpp */
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

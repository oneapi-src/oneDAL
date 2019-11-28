/* file: batch_input.cpp */
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
#include "com_intel_daal_algorithms_pca_transform_TransformInput.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::pca::transform;

#include "com_intel_daal_algorithms_pca_transform_TransformInputId.h"
#define InputDataId         com_intel_daal_algorithms_pca_transform_TransformInputId_InputDataId
#define InputEigenvectorsId com_intel_daal_algorithms_pca_transform_TransformInputId_InputEigenvectorsId
#include "com_intel_daal_algorithms_pca_transform_TransformDataInputId.h"
#define DataForTransformId com_intel_daal_algorithms_pca_transform_TransformDataInputId_DataForTransformId
#include "com_intel_daal_algorithms_pca_transform_TransformComponentId.h"
#define TransformComponentMeansId       com_intel_daal_algorithms_pca_transform_TransformComponentId_TransformComponentMeansId
#define TransformComponentVariancesId   com_intel_daal_algorithms_pca_transform_TransformComponentId_TransformComponentVariancesId
#define TransformComponentEigenvaluesId com_intel_daal_algorithms_pca_transform_TransformComponentId_TransformComponentEigenvaluesId

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cSetInputTable(JNIEnv * jenv, jobject thisObj, jlong inputAddr,
                                                                                                  jint id, jlong ntAddr)
{
    if (id == InputDataId || id == InputEigenvectorsId)
    {
        jniInput<pca::transform::Input>::set<pca::transform::InputId, NumericTable>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cSetInputTransformData
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cSetInputTransformData(JNIEnv * jenv, jobject thisObj,
                                                                                                          jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == DataForTransformId)
    {
        jniInput<pca::transform::Input>::set<pca::transform::TransformDataInputId, KeyValueDataCollection>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cSetInputTransformComponent
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cSetInputTransformComponent(JNIEnv * jenv, jobject thisObj,
                                                                                                               jlong inputAddr, jint wid, jint id,
                                                                                                               jlong ntAddr)
{
    if (wid == DataForTransformId
        && (id == TransformComponentMeansId || id == TransformComponentVariancesId || id == TransformComponentEigenvaluesId))
    {
        jniInput<pca::transform::Input>::setex<pca::transform::TransformDataInputId, pca::transform::TransformComponentId, NumericTable>(
            inputAddr, wid, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cGetInputTable
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cGetInputTable(JNIEnv * jenv, jobject thisObj, jlong inputAddr,
                                                                                                   jint id)
{
    if (id == InputDataId || id == InputEigenvectorsId)
    {
        return jniInput<pca::transform::Input>::get<pca::transform::InputId, NumericTable>(inputAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cGetInputTransformData
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cGetInputTransformData(JNIEnv * jenv, jobject thisObj,
                                                                                                           jlong inputAddr, jint id)
{
    if (id == DataForTransformId)
    {
        return jniInput<pca::transform::Input>::get<pca::transform::TransformDataInputId, KeyValueDataCollection>(inputAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cGetInputTransformComponent
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cGetInputTransformComponent(JNIEnv * jenv, jobject thisObj,
                                                                                                                jlong inputAddr, jint wid, jint id)
{
    if (wid == DataForTransformId
        && (id == TransformComponentMeansId || id == TransformComponentVariancesId || id == TransformComponentEigenvaluesId))
    {
        return jniInput<pca::transform::Input>::getex<pca::transform::TransformDataInputId, pca::transform::TransformComponentId, NumericTable>(
            inputAddr, wid, id);
    }

    return (jlong)0;
}

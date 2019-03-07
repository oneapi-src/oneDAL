/* file: batch_input.cpp */
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
#include "pca/transform/JTransformInput.h"
#include "pca/transform/JTransformInputId.h"
#include "pca/transform/JTransformDataInputId.h"
#include "pca/transform/JTransformComponentId.h"
#include "pca/transform/JTransformMethod.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::pca::transform;

#define InputDataId  com_intel_daal_algorithms_pca_transform_TransformInputId_InputDataId
#define InputEigenvectorsId  com_intel_daal_algorithms_pca_transform_TransformInputId_InputEigenvectorsId
#define DataForTransformId  com_intel_daal_algorithms_pca_transform_TransformDataInputId_DataForTransformId
#define TransformComponentMeansId  com_intel_daal_algorithms_pca_transform_TransformComponentId_TransformComponentMeansId
#define TransformComponentVariancesId  com_intel_daal_algorithms_pca_transform_TransformComponentId_TransformComponentVariancesId
#define TransformComponentEigenvaluesId  com_intel_daal_algorithms_pca_transform_TransformComponentId_TransformComponentEigenvaluesId

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cSetInputTable
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id == InputDataId || id == InputEigenvectorsId)
    {
        jniInput<pca::transform::Input>::
            set<pca::transform::InputId, NumericTable>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cSetInputTransformData
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cSetInputTransformData
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id == DataForTransformId)
    {
        jniInput<pca::transform::Input>::
            set<pca::transform::TransformDataInputId, KeyValueDataCollection>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cSetInputTransformComponent
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cSetInputTransformComponent
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint wid, jint id, jlong ntAddr)
{
    if(wid == DataForTransformId && (id == TransformComponentMeansId || id == TransformComponentVariancesId || id == TransformComponentEigenvaluesId))
    {
        jniInput<pca::transform::Input>::
            setex<pca::transform::TransformDataInputId, pca::transform::TransformComponentId, NumericTable>(inputAddr, wid, id, ntAddr);
    }
}



/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cGetInputTable
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cGetInputTable
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    if(id == InputDataId || id == InputEigenvectorsId)
    {
        return jniInput<pca::transform::Input>::
            get<pca::transform::InputId, NumericTable>(inputAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cGetInputTransformData
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cGetInputTransformData
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint id)
{
    if(id == DataForTransformId)
    {
        return jniInput<pca::transform::Input>::
            get<pca::transform::TransformDataInputId, KeyValueDataCollection>(inputAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformInput
 * Method:    cGetInputTransformComponent
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformInput_cGetInputTransformComponent
(JNIEnv *jenv, jobject thisObj, jlong inputAddr, jint wid, jint id)
{
    if(wid == DataForTransformId && (id == TransformComponentMeansId || id == TransformComponentVariancesId || id == TransformComponentEigenvaluesId))
    {
        return jniInput<pca::transform::Input>::
            getex<pca::transform::TransformDataInputId, pca::transform::TransformComponentId, NumericTable>(inputAddr, wid, id);
    }

    return (jlong)0;
}

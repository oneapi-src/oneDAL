/* file: predict_input.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#include "kdtree_knn_classification/prediction/JPredictionInput.h"
#include "kdtree_knn_classification/prediction/JPredictionMethod.h"
#include "classifier/prediction/JModelInputId.h"
#include "classifier/prediction/JNumericTableInputId.h"

#include "common_helpers.h"

#define dataId com_intel_daal_algorithms_classifier_prediction_NumericTableInputId_Data
#define modelId com_intel_daal_algorithms_classifier_prediction_ModelInputId_Model

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::kdtree_knn_classification::prediction;

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_prediction_Input
 * Method:    cSetInput
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_prediction_Input_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id == dataId)
    {
        jniInput<kdtree_knn_classification::prediction::Input>::
            set<classifier::prediction::NumericTableInputId, NumericTable>(inputAddr, classifier::prediction::data, ntAddr);
    }
    else if(id == modelId)
    {
        jniInput<kdtree_knn_classification::prediction::Input>::
            set<classifier::prediction::ModelInputId, kdtree_knn_classification::Model>(inputAddr, classifier::prediction::model, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_prediction_PredictionBatch
 * Method:    cGetInput
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_prediction_Input_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if(id == dataId)
    {
        return jniInput<kdtree_knn_classification::prediction::Input>::
            get<classifier::prediction::NumericTableInputId, NumericTable>(inputAddr, classifier::prediction::data);
    }
    else if(id == modelId)
    {
        return jniInput<kdtree_knn_classification::prediction::Input>::
            get<classifier::prediction::ModelInputId, kdtree_knn_classification::Model>(inputAddr, classifier::prediction::model);
    }

    return (jlong)0;
}

/* file: model.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>

#include "daal.h"
#include "com_intel_daal_algorithms_decision_tree_classification_Model.h"
#include "com/intel/daal/common_helpers.h"
#include "com/intel/daal/common_helpers_functions.h"
#include "../../classifier/tree_node_visitor.h"
#include "../../tree_utils/classification/tree_node_visitor.h"

USING_COMMON_NAMESPACES()
namespace dtc = daal::algorithms::decision_tree::classification;

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Model
* Method:    cTraverseDF
* Signature: (JJLcom/intel/daal/algorithms/decision_tree/classification/NodeVisitor;)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Model_cTraverseDF(JNIEnv * env, jobject, jlong modAddr,
                                                                                                      jobject visitor)
{
    JavaVM * jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if (status != 0) throwError(env, "Couldn't get Java VM interface");
    daal::classification::JavaTreeNodeVisitor visitorImpl(jvm, visitor);
    (*(dtc::ModelPtr *)modAddr)->traverseDF(visitorImpl);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Model
* Method:    cTraverseBF
* Signature: (JJLcom/intel/daal/algorithms/decision_tree/classification/NodeVisitor;)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Model_cTraverseBF(JNIEnv * env, jobject, jlong modAddr,
                                                                                                      jobject visitor)
{
    JavaVM * jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if (status != 0) throwError(env, "Couldn't get Java VM interface");
    daal::classification::JavaTreeNodeVisitor visitorImpl(jvm, visitor);
    (*(dtc::ModelPtr *)modAddr)->traverseBF(visitorImpl);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Model
* Method:    cTraverseDFS
* Signature: (JJLcom/intel/daal/algorithms/decision_tree/classification/NodeVisitor;)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Model_cTraverseDFS(JNIEnv * env, jobject, jlong modAddr,
                                                                                                       jobject visitor)
{
    JavaVM * jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if (status != 0) throwError(env, "Couldn't get Java VM interface");
    daal::classification::tree_utils::JavaTreeNodeVisitor visitorImpl(jvm, visitor);
    (*(dtc::ModelPtr *)modAddr)->traverseDFS(visitorImpl);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_classification_Model
* Method:    cTraverseBFS
* Signature: (JJLcom/intel/daal/algorithms/decision_tree/classification/NodeVisitor;)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_classification_Model_cTraverseBFS(JNIEnv * env, jobject, jlong modAddr,
                                                                                                       jobject visitor)
{
    JavaVM * jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if (status != 0) throwError(env, "Couldn't get Java VM interface");
    daal::classification::tree_utils::JavaTreeNodeVisitor visitorImpl(jvm, visitor);
    (*(dtc::ModelPtr *)modAddr)->traverseBFS(visitorImpl);
}

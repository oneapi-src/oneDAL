/* file: tree_node_visitor.cpp */
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
#include "common_helpers_functions.h"
#include "tree_node_visitor.h"

namespace daal
{
namespace classification
{
bool JavaTreeNodeVisitor::onLeafNode(size_t level, size_t response)
{
    ThreadLocalStorage tls = _tls.local();
    jint status            = jvm->AttachCurrentThread((void **)(&tls.jniEnv), NULL);
    JNIEnv * env           = tls.jniEnv;

    /* Get current context */
    jclass javaObjectClass = env->GetObjectClass(javaObject);
    if (javaObjectClass == NULL) throwError(env, "Couldn't find class of this java object");

    jmethodID methodID = env->GetMethodID(javaObjectClass, "onLeafNode", "(JJ)Z");
    if (methodID == NULL) throwError(env, "Couldn't find onLeafNode method");

    jboolean val = env->CallBooleanMethod(javaObject, methodID, (jlong)level, (jlong)response);

    if (!tls.is_main_thread) status = jvm->DetachCurrentThread();
    _tls.local() = tls;
    return val != 0;
}

bool JavaTreeNodeVisitor::onSplitNode(size_t level, size_t featureIndex, double featureValue)
{
    ThreadLocalStorage tls = _tls.local();
    jint status            = jvm->AttachCurrentThread((void **)(&tls.jniEnv), NULL);
    JNIEnv * env           = tls.jniEnv;

    /* Get current context */
    jclass javaObjectClass = env->GetObjectClass(javaObject);
    if (javaObjectClass == NULL) throwError(env, "Couldn't find class of this java object");

    jmethodID methodID = env->GetMethodID(javaObjectClass, "onSplitNode", "(JJD)Z");
    if (methodID == NULL) throwError(env, "Couldn't find onSplitNode method");
    jboolean val = env->CallBooleanMethod(javaObject, methodID, (jlong)level, (jlong)featureIndex, (jdouble)featureValue);

    if (!tls.is_main_thread) status = jvm->DetachCurrentThread();
    _tls.local() = tls;
    return val != 0;
}

} // namespace classification
} // namespace daal

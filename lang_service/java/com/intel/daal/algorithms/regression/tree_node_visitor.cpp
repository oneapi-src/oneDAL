/* file: tree_node_visitor.cpp */
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
#include "common_helpers_functions.h"
#include "tree_node_visitor.h"

namespace daal
{
namespace regression
{

bool JavaTreeNodeVisitor::onLeafNode(size_t level, double response)
{
    ThreadLocalStorage tls = _tls.local();
    jint status = jvm->AttachCurrentThread((void **)(&tls.jniEnv), NULL);
    JNIEnv *env = tls.jniEnv;

    /* Get current context */
    jclass javaObjectClass = env->GetObjectClass(javaObject);
    if(javaObjectClass == NULL)
        throwError(env, "Couldn't find class of this java object");

    jmethodID methodID = env->GetMethodID(javaObjectClass, "onLeafNode", "(JD)Z");
    if(methodID == NULL)
        throwError(env, "Couldn't find onLeafNode method");

    jboolean val = env->CallBooleanMethod(javaObject, methodID, (jlong)level, (jdouble)response);

    if(!tls.is_main_thread)
        status = jvm->DetachCurrentThread();
    _tls.local() = tls;
    return val != 0;
}

bool JavaTreeNodeVisitor::onSplitNode(size_t level, size_t featureIndex, double featureValue)
{
    ThreadLocalStorage tls = _tls.local();
    jint status = jvm->AttachCurrentThread((void **)(&tls.jniEnv), NULL);
    JNIEnv *env = tls.jniEnv;

    /* Get current context */
    jclass javaObjectClass = env->GetObjectClass(javaObject);
    if(javaObjectClass == NULL)
        throwError(env, "Couldn't find class of this java object");

    jmethodID methodID = env->GetMethodID(javaObjectClass, "onSplitNode", "(JJD)Z");
    if(methodID == NULL)
        throwError(env, "Couldn't find onSplitNode method");
    jboolean val = env->CallBooleanMethod(javaObject, methodID, (jlong)level, (jlong)featureIndex, (jdouble)featureValue);

    if(!tls.is_main_thread)
        status = jvm->DetachCurrentThread();
    _tls.local() = tls;
    return val != 0;
}

}//namespace
}//namespace

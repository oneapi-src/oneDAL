/* file: java_threading_helper.h */
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

#ifndef __JAVA_THREADING_HELPER_H__
#define __JAVA_THREADING_HELPER_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "services/error_handling.h"

namespace daal
{

namespace internal
{

struct _java_tls
{
    JNIEnv *jenv;    // JNI interface poiner
    jobject jbuf;
    jclass jcls;     // Java class associated with this C++ object
    bool is_main_thread;
    bool is_attached;
    /* Default constructor */
    _java_tls()
    {
        jenv = NULL;
        jbuf = NULL;
        jcls = NULL;
        is_main_thread = false;
        is_attached = false;
    }
};

static services::Status attachCurrentThread(JavaVM *jvm, _java_tls &local_tls)
{
    if (!local_tls.is_attached)
    {
        jint status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
        if(status == JNI_OK)
        {
            local_tls.is_attached = true;
        }
        else
        {
            return services::Status(services::ErrorCouldntAttachCurrentThreadToJavaVM);
        }
    }
    return services::Status();
}

static services::Status detachCurrentThread(JavaVM *jvm, _java_tls &local_tls, bool detach_main_thread = false)
{
    if (local_tls.is_attached)
    {
        if(!local_tls.is_main_thread || detach_main_thread)
        {
            jint status = jvm->DetachCurrentThread();
            if (status == JNI_OK)
            {
                local_tls.is_attached = false;
            }
        }
    }
    return services::Status();
}

} // namespace internal

} // namespace daal

#endif

/* file: java_threading_helper.h */
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
    JNIEnv * jenv; // JNI interface poiner
    jobject jbuf;
    jclass jcls; // Java class associated with this C++ object
    bool is_main_thread;
    bool is_attached;
    /* Default constructor */
    _java_tls()
    {
        jenv           = NULL;
        jbuf           = NULL;
        jcls           = NULL;
        is_main_thread = false;
        is_attached    = false;
    }
};

static services::Status attachCurrentThread(JavaVM * jvm, _java_tls & local_tls)
{
    if (!local_tls.is_attached)
    {
        jint status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
        if (status == JNI_OK)
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

static services::Status detachCurrentThread(JavaVM * jvm, _java_tls & local_tls, bool detach_main_thread = false)
{
    if (local_tls.is_attached)
    {
        if (!local_tls.is_main_thread || detach_main_thread)
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

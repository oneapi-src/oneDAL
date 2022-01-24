/* file: java_callback.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

/*
//++
//  Implementation of the class that contains common methods for Java callbacks
//--
*/
#ifndef __JAVA_CALLBACK_H__
#define __JAVA_CALLBACK_H__

#include <jni.h>
#include <tbb/tbb.h>
#include "services/daal_string.h"

namespace daal
{
namespace services
{
class JavaCallback
{
public:
    enum CallbackStatus
    {
        success = 0, /*!< Getting JNI class signature of the input object successfully */
        fail    = 1  /*!< Fail in getting JNI class signature of the input object */
    };

    JavaCallback(JavaVM * _jvm, jobject _javaObject) : jvm(_jvm), javaObject(NULL)
    {
        ThreadLocalStorage tls = _tls.local();

        /* Mark current thread as 'main' in order not to detach it further */
        tls.is_main_thread = true;

        jint status = jvm->AttachCurrentThread((void **)(&(tls.jniEnv)), NULL);
        if (status != JNI_OK)
        {
            String err("Couldn't attach main thread to Java VM");
            tls.jniEnv->ThrowNew(tls.jniEnv->FindClass("java/lang/Exception"), err.c_str());
        };

        javaObject = tls.jniEnv->NewGlobalRef(_javaObject);
        if (javaObject == NULL)
        {
            String err("Couldn't create global ref from javaObject ");
            tls.jniEnv->ThrowNew(tls.jniEnv->FindClass("java/lang/Exception"), err.c_str());
        }
        _tls.local() = tls;
    }

    JavaCallback(const JavaCallback & other) : jvm(other.jvm), javaObject(NULL)
    {
        ThreadLocalStorage tls = _tls.local();

        /* Mark current thread as 'main' in order not to detach it further */
        tls.is_main_thread = true;

        jint status = jvm->AttachCurrentThread((void **)(&(tls.jniEnv)), NULL);
        if (status != JNI_OK)
        {
            String err("Couldn't attach main thread to Java VM");
            tls.jniEnv->ThrowNew(tls.jniEnv->FindClass("java/lang/Exception"), err.c_str());
        }

        /* Get class of the input object */
        jclass javaObjectClass = tls.jniEnv->GetObjectClass(other.javaObject);
        if (javaObjectClass == 0)
        {
            tls.jniEnv->ThrowNew(tls.jniEnv->FindClass("java/lang/Exception"), "javaObjectClass could not be initialized");
            return;
        }

        /* Get context object related to the input object */
        jmethodID getContextMethodID = tls.jniEnv->GetMethodID(javaObjectClass, "getContext", "()Lcom/intel/daal/services/DaalContext;");
        if (getContextMethodID == 0)
        {
            tls.jniEnv->ThrowNew(tls.jniEnv->FindClass("java/lang/Exception"), "getContext() method ID could not be initialized");
            return;
        }

        jobject context = tls.jniEnv->CallObjectMethod(other.javaObject, getContextMethodID);

        /* Get the JNI class signature of the input object */
        const int maxLength = 1024;
        char javaObjectClassName[maxLength];
        CallbackStatus st = getJavaObjectClassName(tls.jniEnv, other.javaObject, javaObjectClass, javaObjectClassName, maxLength);
        if (st == fail)
        {
            tls.jniEnv->ThrowNew(tls.jniEnv->FindClass("java/lang/Exception"), "Java object class name is too long could");
            return;
        }

        /* Find the clone() method of the input object */
        services::String fullCloneSignature("(Lcom/intel/daal/services/DaalContext;)", 40);
        services::String javaResultClassNameString(javaObjectClassName, strnlen(javaObjectClassName, String::__DAAL_STR_MAX_SIZE));
        fullCloneSignature.add(javaResultClassNameString);
        jmethodID cloneMethodID = tls.jniEnv->GetMethodID(javaObjectClass, "clone", fullCloneSignature.c_str());

        if (cloneMethodID == 0)
        {
            tls.jniEnv->ThrowNew(tls.jniEnv->FindClass("java/lang/Exception"), "clone() method ID could not be initialized");
            return;
        }

        /* Call the clone() and get the copy of the input object */
        jobject _javaObject = tls.jniEnv->CallObjectMethod(other.javaObject, cloneMethodID, context);

        javaObject = tls.jniEnv->NewGlobalRef(_javaObject);
        if (javaObject == NULL)
        {
            String err("Couldn't create global ref from javaObject ");
            tls.jniEnv->ThrowNew(tls.jniEnv->FindClass("java/lang/Exception"), err.c_str());
        }

        if (!tls.is_main_thread)
        {
            status = jvm->DetachCurrentThread();
        }
        _tls.local() = tls;
    }

    virtual ~JavaCallback()
    {
        ThreadLocalStorage tls = _tls.local();
        if (NULL != tls.jniEnv)
        {
            tls.jniEnv->DeleteGlobalRef(javaObject);
        }
    }

    virtual JavaCallback * cloneImpl() { return new JavaCallback(*this); }

protected:
    struct ThreadLocalStorage
    {
        JNIEnv * jniEnv; // JNI interface poiner
        bool is_main_thread;
        /* Default constructor */
        ThreadLocalStorage() : jniEnv(NULL), is_main_thread(false) {};
    };
    tbb::enumerable_thread_specific<ThreadLocalStorage> _tls; /* Thread local storage */
    JavaVM * jvm;
    jobject javaObject;

    CallbackStatus getJavaObjectClassName(JNIEnv * env, jobject javaObject, jclass javaObjectClass, char * className, int maxLength)
    {
        jmethodID getClassMethodId = env->GetMethodID(javaObjectClass, "getClass", "()Ljava/lang/Class;");
        if (getClassMethodId == 0)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "getClass() method ID could not be initialized");
            return fail;
        }

        jobject classObject = env->CallObjectMethod(javaObject, getClassMethodId);

        /* Get the class object's class descriptor */
        jclass cls = env->GetObjectClass(classObject);
        if (cls == 0)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "javaObjectClass could not be initialized");
            return fail;
        }

        /* Find the getName() method on the class object */
        jmethodID getNameMethodId = env->GetMethodID(cls, "getName", "()Ljava/lang/String;");
        if (getNameMethodId == 0)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "getName() method ID could not be initialized");
            return fail;
        }

        /* Call the getName() to get a jstring object with fully qualified Java class name */
        jstring classNameString = (jstring)env->CallObjectMethod(classObject, getNameMethodId);

        jint stringLength = env->GetStringUTFLength(classNameString);
        if (stringLength + 3 > maxLength) return fail;
        env->GetStringUTFRegion(classNameString, 0, stringLength, className + 1);

        /* Convert Java class name to JNI class name */
        className[0] = 'L';
        for (size_t i = 0; i < stringLength + 1; i++)
        {
            if (className[i] == '.')
            {
                className[i] = '/';
            }
        }
        className[stringLength + 1] = ';';
        className[stringLength + 2] = '\0';

        /* Release the JNI objects */
        env->DeleteLocalRef(classObject);
        env->DeleteLocalRef(cls);
        return success;
    }

    jobject constructJavaObjectFromCppObject(JNIEnv * env, jlong cObjectAddr, jobject context, const char * javaClassName)
    {
        jclass javaClass = env->FindClass(javaClassName);
        if (javaClass == NULL)
        {
            String err("Couldn't find class of java object ");
            err.add(String(javaClassName));
            env->ThrowNew(env->FindClass("java/lang/Exception"), err.c_str());
        }
        const char * constructorSignature = "(Lcom/intel/daal/services/DaalContext;J)V";
        jmethodID constructorID           = env->GetMethodID(javaClass, "<init>", constructorSignature);
        if (constructorID == NULL)
        {
            String err("Couldn't find class of java constructor ");
            err.add(String(javaClassName));
            err.add(String(" "));
            err.add(String(constructorSignature));
            env->ThrowNew(env->FindClass("java/lang/Exception"), err.c_str());
        }
        jobject classObject = env->NewObject(javaClass, constructorID, context, cObjectAddr);
        if (classObject == NULL)
        {
            String err("Couldn't create java object ");
            err.add(String(javaClassName));
            env->ThrowNew(env->FindClass("java/lang/Exception"), err.c_str());
        }

        return classObject;
    }

    jobject constructJavaObjectFromFactory(JNIEnv * env, jlong cObjectAddr, jobject context)
    {
        jclass javaFactoryClass = env->FindClass("com/intel/daal/data_management/data/Factory");
        if (javaFactoryClass == NULL)
        {
            String err("Couldn't find class of java object ");
            err.add(String("com/intel/daal/data_management/data/Factory"));
            env->ThrowNew(env->FindClass("java/lang/Exception"), err.c_str());
        }
        const char * instanceSignature = "()Lcom/intel/daal/data_management/data/Factory;";
        jmethodID instanceID           = env->GetStaticMethodID(javaFactoryClass, "instance", instanceSignature);
        if (instanceID == NULL)
        {
            String err("Couldn't find static function ");
            err.add(String("com/intel/daal/data_management/data/Factory"));
            err.add(String("."));
            err.add(String("instance"));
            err.add(String(instanceSignature));
            env->ThrowNew(env->FindClass("java/lang/Exception"), err.c_str());
        }
        jobject factoryObject              = env->CallStaticObjectMethod(javaFactoryClass, instanceID);
        const char * createObjectSignature = "(Lcom/intel/daal/services/DaalContext;J)Lcom/intel/daal/data_management/data/SerializableBase;";
        jmethodID createObjectID           = env->GetMethodID(javaFactoryClass, "createObject", createObjectSignature);
        if (createObjectID == NULL)
        {
            String err("Couldn't find function ");
            err.add(String("com/intel/daal/data_management/data/Factory"));
            err.add(String("."));
            err.add(String("createObject"));
            err.add(String(createObjectSignature));
            env->ThrowNew(env->FindClass("java/lang/Exception"), err.c_str());
        }
        jobject classObject = env->CallObjectMethod(factoryObject, createObjectID, context, cObjectAddr);
        if (classObject == NULL)
        {
            String err("Couldn't create java object ");
            env->ThrowNew(env->FindClass("java/lang/Exception"), err.c_str());
        }

        return classObject;
    }

    //debug function
#if 0
    void printClassName(JNIEnv *env)
    {
        jclass cls = env->GetObjectClass(javaObject);

        // First get the class object
        jmethodID mid = env->GetMethodID(cls, "getClass", "()Ljava/lang/Class;");
        jobject clsObj = env->CallObjectMethod(javaObject, mid);

        // Now get the class object's class descriptor
        cls = env->GetObjectClass(clsObj);

        // Find the getName() method on the class object
        mid = env->GetMethodID(cls, "getName", "()Ljava/lang/String;");

        // Call the getName() to get a jstring object back
        jstring strObj = (jstring)env->CallObjectMethod(clsObj, mid);

        // Now get the c string from the java jstring object
        const char *str = env->GetStringUTFChars(strObj, NULL);

        // Print the class name
        printf("\nClass name is: %s\n", str);

        // Release the memory pinned char array
        env->ReleaseStringUTFChars(strObj, str);
    }
#endif

private:
    JavaCallback & operator=(const JavaCallback &);
};
} // namespace services
} // namespace daal

#endif

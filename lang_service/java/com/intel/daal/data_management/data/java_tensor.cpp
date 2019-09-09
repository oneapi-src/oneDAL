/* file: java_tensor.cpp */
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

#ifndef __JAVA_TENSOR_CPP__
#define __JAVA_TENSOR_CPP__

#include <jni.h>

#include "java_tensor.h"
#include "common_defines.i"

using namespace daal::data_management;

namespace daal
{

void JavaTensorBase::setJavaVM(JavaVM *jvm)
{
    if (globalJavaVM == NULL)
    {
        globalJavaVM = jvm;
        Factory::instance().registerObject(new Creator<JavaTensor<SERIALIZATION_JAVANIO_HOMOGEN_TENSOR_ID> >());
    }
}

JavaVM* JavaTensorBase::getJavaVM()
{
    return globalJavaVM;
}

void JavaTensorBase::setDaalContext(jobject context)
{
    globalDaalContext.local() = context;
}

jobject JavaTensorBase::getDaalContext()
{
    return globalDaalContext.local();
}

template class JavaTensor<SERIALIZATION_JAVANIO_HOMOGEN_TENSOR_ID>;

IMPLEMENT_SERIALIZABLE_TAGT(JavaTensor,SERIALIZATION_JAVANIO_HOMOGEN_TENSOR_ID);

} // namespace daal

#endif

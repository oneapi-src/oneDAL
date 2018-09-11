/* file: java_tensor.cpp */
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

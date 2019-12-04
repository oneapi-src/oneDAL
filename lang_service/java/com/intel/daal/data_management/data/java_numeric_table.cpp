/* file: java_numeric_table.cpp */
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

#ifndef __JAVA_NUMERIC_TABLE_CPP__
#define __JAVA_NUMERIC_TABLE_CPP__

#include <jni.h>

#include "java_numeric_table.h"
#include "common_defines.i"

using namespace daal::data_management;

namespace daal
{
void JavaNumericTableBase::setJavaVM(JavaVM * jvm)
{
    if (globalJavaVM == NULL)
    {
        globalJavaVM = jvm;
        Factory::instance().registerObject(new Creator<JavaNumericTable<SERIALIZATION_JAVANIO_HOMOGEN_NT_ID> >());
        Factory::instance().registerObject(new Creator<JavaNumericTable<SERIALIZATION_JAVANIO_AOS_NT_ID> >());
        Factory::instance().registerObject(new Creator<JavaNumericTable<SERIALIZATION_JAVANIO_SOA_NT_ID> >());
        Factory::instance().registerObject(new Creator<JavaNumericTable<SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID> >());
        Factory::instance().registerObject(new Creator<JavaNumericTable<SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID> >());
    }
}

JavaVM * JavaNumericTableBase::getJavaVM()
{
    return globalJavaVM;
}

void JavaNumericTableBase::setDaalContext(jobject context)
{
    globalDaalContext.local() = context;
}

jobject JavaNumericTableBase::getDaalContext()
{
    return globalDaalContext.local();
}

template class JavaNumericTable<SERIALIZATION_JAVANIO_HOMOGEN_NT_ID>;
template class JavaNumericTable<SERIALIZATION_JAVANIO_AOS_NT_ID>;
template class JavaNumericTable<SERIALIZATION_JAVANIO_SOA_NT_ID>;
template class JavaNumericTable<SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID>;
template class JavaNumericTable<SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID>;

IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable, SERIALIZATION_JAVANIO_CSR_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable, SERIALIZATION_JAVANIO_HOMOGEN_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable, SERIALIZATION_JAVANIO_AOS_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable, SERIALIZATION_JAVANIO_SOA_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable, SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable, SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID);

} // namespace daal

#endif

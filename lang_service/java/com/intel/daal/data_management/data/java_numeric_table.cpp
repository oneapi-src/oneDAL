/* file: java_numeric_table.cpp */
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

#ifndef __JAVA_NUMERIC_TABLE_CPP__
#define __JAVA_NUMERIC_TABLE_CPP__

#include <jni.h>

#include "java_numeric_table.h"
#include "common_defines.i"

using namespace daal::data_management;

namespace daal
{

void JavaNumericTableBase::setJavaVM(JavaVM *jvm)
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

JavaVM* JavaNumericTableBase::getJavaVM()
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

IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable,SERIALIZATION_JAVANIO_CSR_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable,SERIALIZATION_JAVANIO_HOMOGEN_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable,SERIALIZATION_JAVANIO_AOS_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable,SERIALIZATION_JAVANIO_SOA_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable,SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID);
IMPLEMENT_SERIALIZABLE_TAGT(JavaNumericTable,SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID);

} // namespace daal

#endif

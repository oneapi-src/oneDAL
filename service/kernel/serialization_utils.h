/* file: serialization_utils.h */
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

/*
//++
//  Declaration and implementation of mechanism for serializable objects registration.
//--
*/

#ifndef __SERIALIZATION_UTILS_H__
#define __SERIALIZATION_UTILS_H__

#include "data_management/data/factory.h"

#define __DAAL_SERIALIZATION_TAG(ClassName, Tag)                 \
    int ClassName::serializationTag() { return _desc.tag(); }    \
    int ClassName::getSerializationTag() const { return _desc.tag(); }

#define __DAAL_REGISTER_SERIALIZATION_CLASS(ClassName, Tag)                                      \
    static data_management::SerializationIface* creator##ClassName() { return new ClassName(); } \
    data_management::SerializationDesc ClassName::_desc(creator##ClassName, Tag);                \
    __DAAL_SERIALIZATION_TAG(ClassName, Tag)

#define __DAAL_REGISTER_SERIALIZATION_CLASS2(ClassName, ImplClassName, Tag)\
    static data_management::SerializationIface* creator##ClassName() { return new ImplClassName(); }\
    data_management::SerializationDesc ClassName::_desc(creator##ClassName, Tag); \
    __DAAL_SERIALIZATION_TAG(ClassName, Tag)

#define __DAAL_REGISTER_SERIALIZATION_CLASS3(ClassName, ClassName2, Tag)                                                 \
    static data_management::SerializationIface* creator##ClassName##ClassName2() { return new ClassName<ClassName2>(); } \
    data_management::SerializationDesc ClassName<ClassName2>::_desc(creator##ClassName##ClassName2, Tag);                \
    __DAAL_SERIALIZATION_TAG(ClassName<ClassName2>, Tag)

#endif

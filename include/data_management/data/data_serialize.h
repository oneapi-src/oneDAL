/* file: data_serialize.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Declaration and implementation of the serialization class
//--
*/

#ifndef __DAAL_SERIALIZE_H__
#define __DAAL_SERIALIZE_H__

#include "services/base.h"
#include "services/daal_memory.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @defgroup serialization Data Serialization and Deserialization
 * \brief Contains classes that implement serialization and deserialization.
 * @ingroup data_management
 * @{
 */
class InputDataArchive;
class OutputDataArchive;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SERIALIZATIONIFACE"></a>
 *  \brief Abstract interface class that defines the interface for serialization and deserialization.
 */

class DAAL_EXPORT SerializationIface : public Base
{
public:
    virtual ~SerializationIface() {}

    /**
     *  Performs serialization
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    void serialize(interface1::InputDataArchive &archive);

    /**
     *  Performs deserialization
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    void deserialize(interface1::OutputDataArchive &archive);

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    virtual int getSerializationTag() const = 0;

    /**
     *  Interfaces for the implementation of serialization
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    virtual void serializeImpl(interface1::InputDataArchive *archive) = 0;

    /**
     *  Interfaces for the implementation of deserialization
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    virtual void deserializeImpl(interface1::OutputDataArchive *archive) = 0;
};

/// @cond
/** For internal use only */
class DAAL_EXPORT SerializationDesc
{
public:
    typedef SerializationIface* (*creatorFunc)();
    SerializationDesc(creatorFunc func, int tag);
    int tag() const { return _tag; }
    creatorFunc creator() const { return _f; }
    const SerializationDesc* next() const { return _next; }
    static const SerializationDesc* first();

private:
    creatorFunc _f;
    const int _tag;
    SerializationDesc* _next;
};
/// @endcond

/** @} */
} // namespace interface1
using interface1::SerializationIface;
using interface1::SerializationDesc;

}
}

#define DECLARE_SERIALIZABLE()\
private:\
    static data_management::SerializationDesc _desc; \
public:\
    static int serializationTag(); \
    virtual int getSerializationTag() const DAAL_C11_OVERRIDE;

#define DECLARE_SERIALIZABLE_TAG()\
public:\
    static int serializationTag(); \
    virtual int getSerializationTag() const DAAL_C11_OVERRIDE;

#endif

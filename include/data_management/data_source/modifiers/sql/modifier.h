/* file: modifier.h */
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

#ifndef __DATA_SOURCE_MODIFIERS_SQL_MODIFIER_H__
#define __DATA_SOURCE_MODIFIERS_SQL_MODIFIER_H__

#include "data_management/data_source/modifiers/modifier.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
/**
 * \brief Contains modifiers components for SQL Data Source
 */
namespace sql
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @defgroup data_source_modifiers_sql SQL
 * \brief Defines SQL specific feature modifiers
 * @ingroup data_source_modifiers
 * @{
 */

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__CONFIGIFACE"></a>
 * \brief Abstract class that defines interface of modifier configuration
 */
class ConfigIface : public modifiers::ConfigIface { };

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__CONFIG"></a>
 * \brief Base class that represents modifier configuration, object of that
 *        class is passed to the modifier on initialization and finalization stages
 */
class Config : public Base, public ConfigIface { };

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__CONTEXTIFACE"></a>
 * \brief Abstract class that defines interface of modifier context
 */
class ContextIface : public modifiers::ContextIface
{
public:
    /**
     * Gets the number of columns available for the modifier
     * \return The number of columns
     */
    virtual size_t getNumberOfColumns() const = 0;

    /**
     * \brief      Gets the raw buffer retrieved from SQL table
     * \param[in]  columnIndex  The column index
     * \return     The buffer which contains raw bytes retrieved from SQL table
     */
    virtual services::BufferView<char> getRawValue(size_t columnIndex) const = 0;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__CONTEXT"></a>
 * \brief Base class that represents modifier context, object of that class is
 *        passed to the modifier as an argument of FeatureModifierIface::apply method
 */
class Context : public Base, public ContextIface
{
public:
    template<typename T>
    T getValue(size_t columnIndex) const
    {
        /* Very simple implementation of conversion between C and SQL types.
         * There is no guarantee that returned value contains valid data if the
         * type does not match the type of SQL column. For more information see
         * https://docs.microsoft.com/en-us/sql/odbc/reference/appendixes/conve
         * rting-data-from-sql-to-c-data-types */
        const services::BufferView<char> rawValue = getRawValue(columnIndex);
        DAAL_ASSERT( rawValue.size() == sizeof(T) );
        return *( (const T *)(rawValue.data()) );
    }
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__FEATUREMODIFIERIFACE"></a>
 * \brief Specialization of modifiers::FeatureModifierIface for SQL feature modifier
 */
class FeatureModifierIface : public modifiers::FeatureModifierIface<Config, Context> { };
typedef services::SharedPtr<FeatureModifierIface> FeatureModifierIfacePtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__FEATUREMODIFIER"></a>
 * \brief Base class for feature modifier, intended for inheritance from the user side
 */
class FeatureModifier : public Base, public FeatureModifierIface
{
public:
    /**
     * Default implementation of interface method
     * \param config  The configuration of the modifier
     */
    virtual void initialize(Config &config) DAAL_C11_OVERRIDE { }

    /**
     * Default implementation of interface method
     * \param config  The configuration of the modifier
     */
    virtual void finalize(Config &config) DAAL_C11_OVERRIDE { }
};
typedef services::SharedPtr<FeatureModifier> FeatureModifierPtr;


/* Specifications of the Context::getValue method */

template<>
services::StringView Context::getValue<services::StringView>(size_t columnIndex) const
{
    const services::BufferView<char> buffer = getRawValue(columnIndex);
    return services::StringView( buffer.data(), buffer.size() );
}

template<>
std::string Context::getValue<std::string>(size_t columnIndex) const
{
    const services::BufferView<char> buffer = getRawValue(columnIndex);
    return std::string( buffer.data(), buffer.size() );
}

template<>
std::vector<char> Context::getValue<std::vector<char> >(size_t columnIndex) const
{
    const services::BufferView<char> buffer = getRawValue(columnIndex);
    return std::vector<char>( buffer.data(), buffer.data() + buffer.size() );
}

/** @} */
} // namespace interface1

using interface1::Config;
using interface1::Context;
using interface1::FeatureModifierIface;
using interface1::FeatureModifierIfacePtr;
using interface1::FeatureModifier;
using interface1::FeatureModifierPtr;

} // namespace sql
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif

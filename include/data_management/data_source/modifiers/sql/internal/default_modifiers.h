/* file: default_modifiers.h */
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

#ifndef __DATA_SOURCE_MODIFIERS_SQL_DEFAULT_MODIFIERS_H__
#define __DATA_SOURCE_MODIFIERS_SQL_DEFAULT_MODIFIERS_H__

#include "services/daal_shared_ptr.h"
#include "services/internal/collection.h"

#include "data_management/features/defines.h"
#include "data_management/data_source/modifiers/sql/modifier.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
namespace sql
{
namespace internal
{

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__INTERNAL__CONTINUOUSFEATUREMODIFIER"></a>
 * \brief Feature modifier that parses tokens as continuous features
 */
class ContinuousFeatureModifier : public FeatureModifier
{
public:
    virtual void apply(Context &context) DAAL_C11_OVERRIDE
    {
        services::BufferView<DAAL_DATA_TYPE> outputBuffer = context.getOutputBuffer();
        for (size_t i = 0; i < outputBuffer.size(); i++)
        {
            outputBuffer[i] = context.getValue<DAAL_DATA_TYPE>(i);
        }
    }
};

} // namespace internal
} // namespace sql
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif

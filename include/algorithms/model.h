/* file: model.h */
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

/*
//++
//  Data model classes declarations
//--
*/

#ifndef __MODEL_H__
#define __MODEL_H__

#include "data_management/data/data_archive.h"
#include "services/base.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */
struct ValidationMetricIface {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MODEL"></a>
 * \brief The base class for the classes that represent the models, such as linear_regression::Model
 *        or svm::Model
 * \note  The current version of the library does not support generation of validation metrics
 *        for the models
 */
class Model : public data_management::SerializationIface
{
public:
    /** Default constructor */
    Model()
    {}

    virtual ~Model() {}

    /**
     * \copydoc daal::data_management::interface1::SerializationIface::getSerializationTag()
     */
    int getSerializationTag() const DAAL_C11_OVERRIDE  { return 0; }

protected:
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return services::Status();
    }

    DECLARE_SERIALIZABLE_IMPL();
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1
using interface1::ValidationMetricIface;
using interface1::Model;
using interface1::ModelPtr;

}
} // namespace daal
#endif

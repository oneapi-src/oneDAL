/* file: regression_model.h */
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
//  Implementation of the class defining the regression model
//--
*/

#ifndef __REGRESSION_MODEL_H__
#define __REGRESSION_MODEL_H__

#include "algorithms/model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup regression Regression
 * \brief Contains classes for work with the regression algorithms
 * @ingroup training_and_prediction
 */
/**
 * @defgroup base_regression Base Regression
 * \copydoc daal::algorithms::regression
 * @ingroup regression
 */
/**
 * \brief Contains base classes for the regression algorithms
 */
namespace regression
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup base_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__MODEL"></a>
 * \brief %Base class for models trained with the regression algorithm
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::Model
{
public:
    DAAL_CAST_OPERATOR(Model)

    virtual ~Model()
    {}

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    virtual size_t getNumberOfFeatures() const = 0;
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;
/** @} */
}
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;
}
}
}
#endif

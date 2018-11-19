/* file: ridge_regression_ne_model.h */
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
//  Declaration of the ridge regression model class for the normal equations method
//--
*/

#ifndef __RIDGE_REGRESSION_NE_MODEL_H__
#define __RIDGE_REGRESSION_NE_MODEL_H__

#include "algorithms/ridge_regression/ridge_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace interface1
{
/**
 * @ingroup ridge_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__MODELNORMEQ"></a>
 * \brief %Model trained with the ridge regression algorithm using the normal equations method
 *
 * \tparam modelFPType  Data type to store ridge regression model data, double or float
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT ModelNormEq : public Model
{
public:
    DECLARE_MODEL(ModelNormEq, ridge_regression::Model);

    virtual ~ModelNormEq() { }

    /**
     * Returns a Numeric table that contains partial sums X'*X
     * \return Numeric table that contains partial sums X'*X
     */
    virtual data_management::NumericTablePtr getXTXTable() = 0;

    /**
     * Returns a Numeric table that contains partial sums X'*Y
     * \return Numeric table that contains partial sums X'*Y
     */
    virtual data_management::NumericTablePtr getXTYTable() = 0;
};
typedef services::SharedPtr<ModelNormEq> ModelNormEqPtr;
typedef services::SharedPtr<const ModelNormEq> ModelNormEqConstPtr;
/** @} */
} // namespace interface1
using interface1::ModelNormEq;
using interface1::ModelNormEqPtr;
using interface1::ModelNormEqConstPtr;

} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif

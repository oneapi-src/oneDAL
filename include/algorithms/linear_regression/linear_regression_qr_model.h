/* file: linear_regression_qr_model.h */
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
//  Declaration of the linear regression model class for the QR decomposition-based method
//--
*/

#ifndef __LINREG_QR_MODEL_H__
#define __LINREG_QR_MODEL_H__

#include "algorithms/linear_regression/linear_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{

namespace interface1
{
/**
 * @ingroup linear_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__MODELQR"></a>
 * \brief %Model trained with the linear regression algorithm using the QR decomposition-based method
 *
 * \par References
 *      - Parameter class
 *      - Prediction class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT ModelQR : public Model
{
public:
    DECLARE_MODEL(ModelQR, linear_regression::Model);

    virtual ~ModelQR() { }

    /**
     * Returns a Numeric table that contains the R factor of QR decomposition
     * \return Numeric table that contains the R factor of QR decomposition
     */
    virtual data_management::NumericTablePtr getRTable() = 0;

    /**
     * Returns a Numeric table that contains Q'*Y, where Q is the factor of QR decomposition for a data block,
     * Y is the respective block of the matrix of responses
     * \return Numeric table that contains partial sums Q'*Y
     */
    virtual data_management::NumericTablePtr getQTYTable() = 0;
};
typedef services::SharedPtr<ModelQR> ModelQRPtr;
typedef services::SharedPtr<const ModelQR> ModelQRConstPtr;
/** @} */
} // namespace interface1
using interface1::ModelQR;
using interface1::ModelQRPtr;
using interface1::ModelQRConstPtr;

}
}
} // namespace daal
#endif

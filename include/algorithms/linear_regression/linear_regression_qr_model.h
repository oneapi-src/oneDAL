/* file: linear_regression_qr_model.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
 * \tparam modelFPType  Data type to store linear regression model data, double or float
 *
 * \par References
 *      - Parameter class
 *      - Model class
 *      - Prediction class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT ModelQR : public Model
{
public:
    /**
     * Constructs the linear regression model for the QR decomposition-based method
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of linear regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    DAAL_EXPORT ModelQR(size_t featnum, size_t nrhs, const linear_regression::Parameter &par, modelFPType dummy);

    /**
     * Empty constructor for deserialization
     */
    ModelQR() : Model() { }

    virtual ~ModelQR() { }


    /**
     * Initializes the linear regression model
     */
    virtual void initialize() DAAL_C11_OVERRIDE;

    /**
     * Returns a Numeric table that contains the R factor of QR decomposition
     * \return Numeric table that contains the R factor of QR decomposition
     */
    data_management::NumericTablePtr getRTable();

    /**
     * Returns a Numeric table that contains Q'*Y, where Q is the factor of QR decomposition for a data block,
     * Y is the respective block of the matrix of responses
     * \return Numeric table that contains partial sums Q'*Y
     */
    data_management::NumericTablePtr getQTYTable();

    /**
     * Returns the serialization tag of the linear regression model
     * \return         Serialization tag of the linear regression model
     */

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_LINEAR_REGRESSION_MODELQR_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_rTable);
        arch->setSharedPtrObj(_qtyTable);
    }

private:
    data_management::NumericTablePtr _rTable;        /* Table that contains matrix R */
    data_management::NumericTablePtr _qtyTable;      /* Table that contains matrix Q'*Y */
};
/** @} */
} // namespace interface1
using interface1::ModelQR;

}
}
} // namespace daal
#endif

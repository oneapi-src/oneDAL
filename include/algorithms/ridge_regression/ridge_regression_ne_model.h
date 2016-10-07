/* file: ridge_regression_ne_model.h */
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
 *      - Model class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT ModelNormEq : public Model
{
public:
    /**
     * Constructs the ridge regression model for the normal equations method
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of ridge regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    DAAL_EXPORT ModelNormEq(size_t featnum, size_t nrhs, const ridge_regression::Parameter &par, modelFPType dummy);

    /**
     * Empty constructor for deserialization
     */
    ModelNormEq() : Model() { }

    virtual ~ModelNormEq() { }

    /**
    * Initializes the ridge regression model
    */
    void initialize() DAAL_C11_OVERRIDE;

    /**
     * Returns a Numeric table that contains partial sums X'*X
     * \return Numeric table that contains partial sums X'*X
     */
    data_management::NumericTablePtr getXTXTable();

    /**
     * Returns a Numeric table that contains partial sums X'*Y
     * \return Numeric table that contains partial sums X'*Y
         */
    data_management::NumericTablePtr getXTYTable();

    /**
     * Returns the serialization tag of the ridge regression model
     * \return         Serialization tag of the ridge regression model
     */

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_RIDGE_REGRESSION_MODELNORMEQ_ID; }
    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  * arch) DAAL_C11_OVERRIDE { serialImpl<data_management::InputDataArchive, false>(arch); }

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE { serialImpl<data_management::OutputDataArchive, true>(arch); }

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_xtxTable);
        arch->setSharedPtrObj(_xtyTable);
    }

private:
    data_management::NumericTablePtr _xtxTable;        /* Table holding a partial sum of X'*X */
    data_management::NumericTablePtr _xtyTable;        /* Table holding a partial sum of X'*Y */
};
/** @} */
} // namespace interface1

using interface1::ModelNormEq;

} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif

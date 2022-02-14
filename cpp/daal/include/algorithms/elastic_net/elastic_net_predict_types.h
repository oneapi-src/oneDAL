/* file: elastic_net_predict_types.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of the elastic net algorithm interface
//--
*/

#ifndef __ELASTIC_NET_PREDICT_TYPES_H__
#define __ELASTIC_NET_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/elastic_net/elastic_net_model.h"
#include "algorithms/linear_model/linear_model_predict_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the elastic net algorithm
 */
namespace elastic_net
{
/**
 * @defgroup elastic_net_prediction Prediction
 * \copydoc daal::algorithms::elastic_net::prediction
 * @ingroup elastic_net
 * @{
 */
/**
 * \brief Contains a class for making elastic net model-based prediction
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ELASTIC_NET__PREDICTION__METHOD"></a>
 * \brief Available methods for making elastic net model-based prediction
 */
enum Method
{
    defaultDense = 0
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ELASTIC_NET__PREDICTION__NUMERICTABLEINPUTID"></a>
 * \brief Available identifiers of input numeric tables for making elastic net model-based prediction
 */
enum NumericTableInputId
{
    data                    = linear_model::prediction::data, /*!< Input data table */
    lastNumericTableInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ELASTIC_NET__PREDICTION__MODELINPUTID"></a>
 * \brief Available identifiers of input models for making elastic net model-based prediction
 */
enum ModelInputId
{
    model            = linear_model::prediction::model, /*!< Trained elastic net model */
    lastModelInputId = model
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ELASTIC_NET__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for making elastic net model-based prediction
 */
enum ResultId
{
    prediction   = linear_model::prediction::prediction, /*!< Result of elastic net model-based prediction */
    lastResultId = prediction
};

/**
 * \brief Contains version 1.0 of the Intel(R) oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ELASTIC_NET__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making elastic net model-based prediction
 */
class DAAL_EXPORT Input : public linear_model::prediction::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other);

    /**
     * Returns an input object for making elastic net model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns an input object for making elastic net model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    elastic_net::ModelPtr get(ModelInputId id) const;

    /**
     * Sets an input object for making elastic net model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr & value);

    /**
     * Sets an input object for making elastic net model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ModelInputId id, const elastic_net::ModelPtr & value);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ELASTIC_NET__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of elastic net model-based prediction
 */
class DAAL_EXPORT Result : public linear_model::prediction::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Returns the result of elastic net model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of elastic net model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;
} // namespace prediction
/** @} */
} // namespace elastic_net
} // namespace algorithms
} // namespace daal

#endif

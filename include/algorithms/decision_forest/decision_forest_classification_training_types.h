/* file: decision_forest_classification_training_types.h */
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
//  Implementation of decision forest classification training algorithm interface.
//--
*/

#ifndef __DECISION_FOREST_CLASSIFICATION_TRAINING_TYPES_H__
#define __DECISION_FOREST_CLASSIFICATION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "algorithms/decision_forest/decision_forest_training_parameter.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
/**
 * @defgroup decision_forest_classification_training Training
 * \copydoc daal::algorithms::decision_forest::classification::training
 * @ingroup decision_forest_classification
 * @{
 */
/**
 * \brief Contains classes for Decision forest models training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__CLASSIFICAION__TRAINING__METHOD"></a>
 * \brief Computation methods for decision forest classification model-based training
 */
enum Method
{
    defaultDense = 0  /*!< Bagging, random choice of features, Gini impurity */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__RESULT_NUMERIC_TABLEID"></a>
* \brief Available identifiers of the result of decision forest model-based training
*/
enum ResultNumericTableId
{
    outOfBagError = classifier::training::model + 1,  /*!< %Numeric table 1x1 containing out-of-bag erro.
                                                           Computed when computeOutOfBagError option is on */
    variableImportance,                               /*!< %Numeric table 1x(number of features) containing variable importance value.
                                                           Computed when parameter.varImportance != none */
    outOfBagErrorPerObservation,                      /*!< %Numeric table 1x(number of observations) containing out-of-bag error value computed.
                                                           Computed when computeOutOfBagErrorPerObservation option is on */
    lastResultId = outOfBagErrorPerObservation
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__PARAMETER"></a>
 * \brief Decision forest algorithm parameters
 *
 * \snippet decision_forest/decision_forest_classification_training_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter, public daal::algorithms::decision_forest::training::Parameter
{
    /** Default constructor */
    Parameter(size_t nClasses) : classifier::Parameter(nClasses) {}
    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */


/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method
 *        of the LogitBoost training algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);

    Result();
    virtual ~Result() {}

    /**
     * Returns the model trained with the LogitBoost algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the LogitBoost algorithm
     */
    ModelPtr get(classifier::training::ResultId id) const;

    /**
    * Sets the result of decision forest model-based training
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(classifier::training::ResultId id, const ModelPtr &value);

    /**
    * Returns the result of decision forest model-based training
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultNumericTableId id) const;

    /**
    * Sets the result of decision forest model-based training
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(ResultNumericTableId id, const data_management::NumericTablePtr &value);

    /**
     * Allocates memory to store final results of the LogitBoost training algorithm
     * \param[in] input         %Input of the LogitBoost training algorithm
     * \param[in] parameter     Parameters of the algorithm
     * \param[in] method        LogitBoost computation method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**s
    * Checks the result of decision forest model-based training
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    * \return Status of checking
    */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Parameter;
using interface1::Result;
using interface1::ResultPtr;

} // namespace daal::algorithms::decision_forest::classification::training
/** @} */
}
}
}
} // namespace daal
#endif // __DECISION_FOREST_CLASSIFICATION_TRAINING_TYPES_H__

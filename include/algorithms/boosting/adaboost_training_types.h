/* file: adaboost_training_types.h */
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
//  Implementation of Ada Boost training algorithm interface.
//--
*/

#ifndef __ADA_BOOST_TRAINING_TYPES_H__
#define __ADA_BOOST_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/boosting/adaboost_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup adaboost Adaboost Classifier
 * \copydoc daal::algorithms::adaboost
 * @ingroup boosting
 */
namespace adaboost
{
/**
 * @defgroup adaboost_training Training
 * \copydoc daal::algorithms::adaboost::training
 * @ingroup adaboost
 * @{
 */
/**
 * \brief Contains classes for AdaBoost models training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ADABOOST__TRAINING__METHOD"></a>
 * Available methods for AdaBoost model training
 */
enum Method
{
    defaultDense = 0        /*!< Default method */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method
 *        of the AdaBoost training algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);

    virtual ~Result() {}

    /**
     * Allocates memory to store final results of AdaBoost training
     * \param[in] input         %Input of the AdaBoost training algorithm
     * \param[in] parameter     Parameters of the algorithm
     * \param[in] method        AdaBoost computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns the model trained with the AdaBoost algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the AdaBoost algorithm
     */
    daal::algorithms::adaboost::ModelPtr get(classifier::training::ResultId id) const;

    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

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
using interface1::Result;
using interface1::ResultPtr;

} // namespace daal::algorithms::adaboost::training
/** @} */
}
}
} // namespace daal
#endif // __ADA_BOOST_TRAINING_TYPES_H__

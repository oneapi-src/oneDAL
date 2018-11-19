/* file: adaboost_model.h */
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
//  Implementation of class defining Ada Boost model.
//--
*/

#ifndef __ADA_BOOST_MODEL_H__
#define __ADA_BOOST_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/boosting/boosting_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the AdaBoost classification algorithm
 */
namespace adaboost
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup adaboost
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__ADABOOST__PARAMETER"></a>
 * \brief AdaBoost algorithm parameters
 *
 * \snippet boosting/adaboost_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public boosting::Parameter
{
    /** Default constructor */
    Parameter();

    /**
     * Constructs the AdaBoost parameter structure
     * \param[in] wlTrainForParameter       Pointer to the training algorithm of the weak learner
     * \param[in] wlPredictForParameter     Pointer to the prediction algorithm of the weak learner
     * \param[in] acc                       Accuracy of the AdaBoost training algorithm
     * \param[in] maxIter                   Maximal number of iterations of the AdaBoost training algorithm
     */
    Parameter(services::SharedPtr<weak_learner::training::Batch>   wlTrainForParameter,
              services::SharedPtr<weak_learner::prediction::Batch> wlPredictForParameter,
              double acc = 0.0, size_t maxIter = 10);

    double accuracyThreshold;       /*!< Accuracy of the AdaBoost training algorithm */
    size_t maxIterations;           /*!< Maximal number of iterations of the AdaBoost training algorithm */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__MODEL"></a>
 * \brief %Model of the classifier trained by the adaboost::training::Batch algorithm.
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public boosting::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model)

    /**
     * Constructs the AdaBoost model
     * \tparam modelFPType  Data type to store AdaBoost model data, double or float
     * \param[in] nFeatures Number of features in the dataset
     * \param[in] dummy     Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, modelFPType dummy);

    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model() : boosting::Model(), _alpha() {}

    /**
     * Constructs the AdaBoost model
     * \tparam modelFPType   Data type to store AdaBoost model data, double or float
     * \param[in]  nFeatures Number of features in the dataset
     * \param[out] stat      Status of the model construction
     */
    template<typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<Model> create(size_t nFeatures, services::Status *stat = NULL);

    virtual ~Model() { }

    /**
     *  Returns a pointer to the array of weights of weak learners constructed
     *  during training of the AdaBoost algorithm.
     *  The size of the array equals the number of weak learners
     *  \return Array of weights of weak learners.
     */
    data_management::NumericTablePtr getAlpha() const;

protected:
    data_management::NumericTablePtr _alpha;     /* Boosting coefficients table */

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        services::Status st = boosting::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st)
            return st;
        arch->setSharedPtrObj(_alpha);

        return st;
    }

    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, modelFPType dummy, services::Status &st);

}; // class Model
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace adaboost
} // namespace algorithms
} // namespace daal
#endif

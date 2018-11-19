/* file: brownboost_model.h */
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
//  Implementation of class defining Brown Boost model.
//--
*/

#ifndef __BROWN_BOOST_MODEL_H__
#define __BROWN_BOOST_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/boosting/boosting_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the BrownBoost classification algorithm
 */
namespace brownboost
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup brownboost
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__BROWNBOOST__PARAMETER"></a>
 * \brief BrownBoost algorithm parameters
 *
 * \snippet boosting/brownboost_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public boosting::Parameter
{
    /** Default constructor */
    Parameter();

    /**
     * Constructs BrownBoost parameter structure
     * \param[in] wlTrainForParameter       Pointer to the training algorithm of the weak learner
     * \param[in] wlPredictForParameter     Pointer to the prediction algorithm of the weak learner
     * \param[in] acc                       Accuracy of the BrownBoost training algorithm
     * \param[in] maxIter                   Maximal number of iterations of the BrownBoost training algorithm
     * \param[in] nrAcc                     Accuracy threshold for Newton-Raphson iterations in the BrownBoost training algorithm
     * \param[in] nrMaxIter                 Maximal number of Newton-Raphson iterations in the BrownBoost training algorithm
     * \param[in] dcThreshold               Threshold needed  to avoid degenerate cases in the BrownBoost training algorithm
     */
    Parameter(services::SharedPtr<weak_learner::training::Batch>   wlTrainForParameter,
              services::SharedPtr<weak_learner::prediction::Batch> wlPredictForParameter,
              double acc = 0.3, size_t maxIter = 10, double nrAcc = 1.0e-3, size_t nrMaxIter = 100, double dcThreshold = 1.0e-2);

    double accuracyThreshold;       /*!< Accuracy of the BrownBoost training algorithm */
    size_t maxIterations;           /*!< Maximal number of iterations of the BrownBoost training algorithm */
    double newtonRaphsonAccuracyThreshold;  /*!< Accuracy threshold for Newton-Raphson iterations in the BrownBoost training algorithm */
    size_t newtonRaphsonMaxIterations;      /*!< Maximal number of Newton-Raphson iterations in the BrownBoost training algorithm */
    double degenerateCasesThreshold;        /*!< Threshold needed to avoid degenerate cases in the BrownBoost training algorithm */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__MODEL"></a>
 * \brief %Model of the classifier trained by the brownboost::training::Batch algorithm.
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
     *  Constructs the BrownBoost %Model
     * \tparam modelFPType  Data type to store BrownBoost model data, double or float
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
    Model() : boosting::Model(), _alpha() { }


    /**
     * Constructs the BrownBoost model
     * \tparam modelFPType  Data type to store BrownBoost model data, double or float
     * \param[in]  nFeatures Number of features in the dataset
     * \param[out] stat      Status of the model construction
     */
    template<typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<Model> create(size_t nFeatures, services::Status *stat = NULL);

    virtual ~Model() { }

    /**
     *  Returns a pointer to the array of weights of weak learners constructed
     *  during training of the BrownBoost algorithm.
     *  The size of the array equals the number of weak learners
     *  \return Array of weights of weak learners.
     */
    data_management::NumericTablePtr getAlpha();

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

} // namespace daal::algorithms::brownboost
}
} // namespace daal
#endif

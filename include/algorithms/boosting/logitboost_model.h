/* file: logitboost_model.h */
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
//  Implementation of class defining LogitBoost model.
//--
*/

#ifndef __LOGIT_BOOST_MODEL_H__
#define __LOGIT_BOOST_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/boosting/boosting_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the LogitBoost classification algorithm
 */
namespace logitboost
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup logitboost
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LOGITBOOST__PARAMETER"></a>
 * \brief LogitBoost algorithm parameters
 *
 * \snippet boosting/brownboost_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public boosting::Parameter
{
    /** Default constructor */
    Parameter();

    /**
     * Constructs LogitBoost parameter structure
     * \param[in] wlTrainForParameter       Pointer to the training algorithm of the weak learner
     * \param[in] wlPredictForParameter     Pointer to the prediction algorithm of the weak learner
     * \param[in] acc                       Accuracy of the LogitBoost training algorithm
     * \param[in] maxIter                   Maximal number of terms in additive regression
     * \param[in] nC                        Number of classes in the training data set
     * \param[in] wThr                      Threshold to avoid degenerate cases when calculating weights W
     * \param[in] zThr                      Threshold to avoid degenerate cases when calculating responses Z
     */
    Parameter(const services::SharedPtr<weak_learner::training::Batch>&   wlTrainForParameter,
              const services::SharedPtr<weak_learner::prediction::Batch>& wlPredictForParameter,
              double acc = 0.0, size_t maxIter = 10, size_t nC = 0, double wThr = 1e-10, double zThr = 1e-10);

    double accuracyThreshold;       /*!< Accuracy of the LogitBoost training algorithm */
    size_t maxIterations;           /*!< Maximal number of terms in additive regression */
    size_t nClasses;                /*!< Number of classes */
    double weightsDegenerateCasesThreshold;     /*!< Threshold to avoid degenerate cases when  calculating weights W */
    double responsesDegenerateCasesThreshold;   /*!< Threshold to avoid degenerate cases when  calculating responses Z */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGITBOOST__MODEL"></a>
 * \brief %Model of the classifier trained by the logitboost::training::Batch algorithm.
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
     * Constructs the LogitBoost model
     * \tparam modelFPType  Data type to store LogitBoost model data, double or float
     * \param[in] nFeatures Number of features in the dataset
     * \param[in] par       Pointer to the parameter structure of the LogitBoost algorithm
     * \param[in] dummy     Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nFeatures, const Parameter *par, modelFPType dummy);

    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model() : boosting::Model(), _nIterations(0) { }

    /**
     * Constructs the LogitBoost model
     * \param[in]  nFeatures Number of features in the dataset
     * \param[in]  par       Pointer to the parameter structure of the LogitBoost algorithm
     * \param[out] stat      Status of the model construction
     */
    static services::SharedPtr<Model> create(size_t nFeatures, const Parameter *par,
                                             services::Status *stat = NULL);

    virtual ~Model() { }

    /**
     * Sets the number of iterations for the algorithm
     * @param nIterations   Number of iterations
     */
    void setIterations(size_t nIterations);

    /**
     * Returns the number of iterations done by the training algorithm
     * \return The number of iterations done by the training algorithm
     */
    size_t getIterations() const;

protected:
    size_t _nIterations;

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        services::Status st = boosting::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st)
            return st;
        arch->set(_nIterations);

        return st;
    }

    Model(size_t nFeatures, const Parameter *par, services::Status &st);
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace daal::algorithms::logitboost
}
} // namespace daal
#endif

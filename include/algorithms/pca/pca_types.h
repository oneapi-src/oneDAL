/* file: pca_types.h */
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
//  Implementation of PCA algorithm interface.
//--
*/

#ifndef __PCA_TYPES_H__
#define __PCA_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/covariance/covariance_batch.h"
#include "algorithms/covariance/covariance_online.h"
#include "algorithms/covariance/covariance_distributed.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup pca Principal Component Analysis
 * \copydoc daal::algorithms::pca
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes for computing the results of the principal component analysis (PCA) algorithm
 */
namespace pca
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__METHOD"></a>
 * Available methods for computing the PCA algorithm
 */
enum Method
{
    correlationDense = 0, /*!< PCA Correlation method */
    defaultDense = 0, /*!< PCA Default method */
    svdDense = 1 /*!< PCA SVD method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__INPUTDATASETID"></a>
 * Available identifiers of input dataset objects for the PCA algorithm
 */
enum InputDatasetId
{
    data = 0 /*!< Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__INPUTCORRELATIONID"></a>
 * Available identifiers of input objects for the PCA Correlation algorithm
 */
enum InputCorrelationId
{
    correlation = 0 /*!< Input correlation table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__STEP2MASTERINPUTID"></a>
 * Available identifiers of input objects for the PCA algorithm on the second step in the distributed processing mode
 */
enum Step2MasterInputId
{
    partialResults = 0 /*!< Collection of partial results computed on local nodes */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__PARTIALCORRELATIONRESULTID"></a>
 * Available identifiers of partial results of the PCA Correlation algorithm
 */
enum PartialCorrelationResultId
{
    nObservationsCorrelation = 0, /* Number of processed observations */
    crossProductCorrelation = 1, /* Cross-product of the processed data */
    sumCorrelation = 2 /* Feature sums of the processed data */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__PARTIALSVDTABLERESULTID"></a>
 * Available identifiers of partial results of the PCA SVD algorithm
 */
enum PartialSVDTableResultId
{
    nObservationsSVD = 0, /* Number of processed observations */
    sumSVD = 1, /* Feature sums of the processed data */
    sumSquaresSVD = 2 /* Feature sums of squares of the processed data */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__PARTIALSVDCOLLECTIONRESULTID"></a>
 * Available identifiers of partial results of the PCA SVD  algorithm
 */
enum PartialSVDCollectionResultId
{
    auxiliaryData = 3, /*!< Auxiliary data of the PCA SVD method */
    distributedInputs = 4 /*!< Auxiliary data of the PCA SVD method on the second step in the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__RESULTID"></a>
 * Available identifiers of the results of the PCA algorithm
 */
enum ResultId
{
    eigenvalues = 0, /*!< Eigenvalues of the correlation matrix */
    eigenvectors = 1 /*!< Eigenvectors of the correlation matrix */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__INPUTIFACE"></a>
 * \brief Abstract class that specifies interface for classes that declare input of the PCA algorithm */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements), _isCorrelation(false) {};

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    virtual size_t getNFeatures() const = 0;

    /**
    * Returns flag defining whether the input data contains correlation matrix or not
    * \return Flag defining whether the input data contains correlation matrix or not
    */
    virtual bool isCorrelation() const { return _isCorrelation; };

    virtual ~InputIface() {};

protected:
    bool _isCorrelation;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__INPUT"></a>
 * \brief Input objects for the PCA algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();

    virtual ~Input() {};

    /**
    * Returns the input object of the PCA algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputDatasetId id) const;

    /**
     * Sets input dataset for the PCA algorithm
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the input object
     */
    void set(InputDatasetId id, const data_management::NumericTablePtr &value);

    /**
     * Sets input correlation matrix for the PCA algorithm
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the input object
     */
    void set(InputCorrelationId id, const data_management::NumericTablePtr &value);

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks input algorithm parameters
    * \param[in] par     Algorithm %parameter
    * \param[in] method  Computation method
    * \return Errors detected while checking
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALRESULTBASE"></a>
 * \brief Provides interface to access partial results obtained with the compute() method of the
 *        PCA algorithm in the online or distributed processing mode
 */
class PartialResultBase :  public daal::algorithms::PartialResult
{
public:
    PartialResultBase(const size_t nElements) : daal::algorithms::PartialResult(nElements) {};

    virtual size_t getNFeatures() const = 0;

    virtual ~PartialResultBase() {};
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the
 *        PCA algorithm in the online or distributed processing mode
 */
template<Method method>
class PartialResult :  public PartialResultBase {};

/**
 * <a name="DAAL-CLASS-PCA__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the PCA Correlation algorithm
 *        in the online or distributed processing mode
 */
template<> class DAAL_EXPORT PartialResult<correlationDense> :  public PartialResultBase
{
public:
    PartialResult();

    /**
     * Gets partial results of the PCA Correlation algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(PartialCorrelationResultId id) const;

    virtual size_t getNFeatures() const DAAL_C11_OVERRIDE;

    /**
     * Sets partial result of the PCA Correlation algorithm
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    void set(const PartialCorrelationResultId id, const data_management::NumericTablePtr &value);

    virtual ~PartialResult() {};

    /**
    * Checks partial results of the PCA Correlation algorithm
    * \param[in] input      %Input object of the algorithm
    * \param[in] parameter  Algorithm %parameter
    * \param[in] method     Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;


    /**
    * Checks partial results of the PCA Ccorrelation algorithm
    * \param[in] par        Algorithm %parameter
    * \param[in] method     Computation method
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_PCA_PARTIAL_RESULT_CORRELATION_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:

    void checkImpl(size_t nFeatures) const;

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-PCA__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of PCA SVD algorithm
 *         in the online or distributed processing mode
 */
template<> class DAAL_EXPORT PartialResult<svdDense> : public PartialResultBase
{
public:
    PartialResult();

    /**
    * Gets partial results of the PCA SVD algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(PartialSVDTableResultId id) const;

    virtual size_t getNFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Gets partial results of the PCA SVD algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(PartialSVDCollectionResultId id) const;

    /**
    * Gets partial results of the PCA SVD algorithm
     * \param[in] id            Identifier of the input object
     * \param[in] elementId     Identifier of the collection element
     * \return                  Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(PartialSVDCollectionResultId id, const size_t &elementId) const;

    /**
     * Sets partial result of the PCA SVD algorithm
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to  the object
     */
    void set(PartialSVDTableResultId id, const data_management::NumericTablePtr &value);

    /**
     * Sets partial result of the PCA SVD algorithm
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    void set(PartialSVDCollectionResultId id, const data_management::DataCollectionPtr &value);

    /**
     * Adds partial result of the PCA SVD algorithm
     * \param[in] id      Identifier of the argument
     * \param[in] value   Pointer to the object
     */
    void add(const PartialSVDCollectionResultId &id, const data_management::DataCollectionPtr &value);

    /**
    * Checks partial results of the PCA SVD algorithm
    * \param[in] input      %Input of algorithm
    * \param[in] parameter  %Parameter of algorithm
    * \param[in] method     Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks partial results of the PCA SVD algorithm
    * \param[in] method     Computation method
    * \param[in] par        %Parameter of algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    virtual ~PartialResult() {};

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_PCA_PARTIAL_RESULT_SVD_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:

    void checkImpl(size_t nFeatures) const;

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALRESULTIMPL"></a>
 * \brief Provides methods to allocate partial results obtained with the compute() method of the PCA SVD algorithm
 *         in the online or distributed processing mode
 */
template<typename algorithmFPType, Method method>
class PartialResultImpl : public PartialResult<method> {};

/**
 * <a name="DAAL-CLASS-PCA__PARTIALRESULTIMPL"></a>
 * \brief Provides methods to allocate partial results obtained with the compute() method of PCA SVD algorithm
 *         in the online or distributed processing mode
 */
template<typename algorithmFPType> class DAAL_EXPORT PartialResultImpl<algorithmFPType, correlationDense> :
    public PartialResult<correlationDense>
{
public:
    PartialResultImpl<algorithmFPType, correlationDense>() {};

    /**
     * Allocates memory to store partial results of the PCA  SVD algorithm
     * \param[in] input     Pointer to an object containing input data
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
};

/**
 * <a name="DAAL-CLASS-PCA__PARTIALRESULTIMPL"></a>
 * \brief Provides methods to allocate partial results obtained with the compute() method of the PCA SVD algorithm
 *         in the online or distributed processing mode
 */
template<typename algorithmFPType> class DAAL_EXPORT PartialResultImpl<algorithmFPType, svdDense> : public PartialResult<svdDense>
{
public:
    PartialResultImpl<algorithmFPType, svdDense>() {};

    /**
     * Allocates memory for storing partial results of the PCA SVD algorithm
     * \param[in] input     Pointer to an object containing input data
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__PCA__PARTIALRESULTSINITIFACE"></a>
 * \brief Abstract interface class for partial results initialization
 */
template<Method method>
struct PartialResultsInitIface
{
    virtual ~PartialResultsInitIface() {}

    /**
     * Initializes partial results
     * \param[in]       input     Input objects for the PCA algorithm
     * \param[in,out]   pres      Partial results of the PCA algorithm
     * \return                    Initialized partial results
     */
    virtual void operator()(const Input &input, services::SharedPtr<PartialResult<method> > &pres) = 0;
};

static inline void setToZero(data_management::NumericTable *table)
{
    data_management::BlockDescriptor<double> block;
    size_t nCols = table->getNumberOfColumns();
    size_t nRows = table->getNumberOfRows();

    double *tableArray;
    table->getBlockOfRows(0, nRows, data_management::writeOnly, block);
    tableArray = block.getBlockPtr();

    for(size_t i = 0; i < nCols * nRows; i++)
    {
        tableArray[i] = 0;
    };

    table->releaseBlockOfRows(block);
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__PCA__DEFAULTPARTIALRESULTSINIT"></a>
 * \brief Abstract interface class for partial results initialization
 */
template<Method method>
struct DefaultPartialResultsInit : public PartialResultsInitIface<method>
{};

/**
 * <a name="DAAL-CLASS-CLASS-PCA__DEFAULTPARTIALRESULTSINIT"></a>
 * \brief Class that specifies the default method for partial results initialization
 */
template<> struct DAAL_EXPORT DefaultPartialResultsInit<correlationDense> : public PartialResultsInitIface<correlationDense>
{
    virtual ~DefaultPartialResultsInit() {}

    /**
     * Initialize partial results
     * \param[in]       input     Input objects for the PCA algorithm
     * \param[in,out]   pres      Partial results of the PCA algorithm
     * \return                    Initialized partial results
     */
    void operator()(const Input &input, services::SharedPtr<PartialResult<correlationDense> > &pres);
};

/**
 * <a name="DAAL-CLASS-CLASS-PCA__DEFAULTPARTIALRESULTSINIT"></a>
 * \brief Class that specifies the default method for partial results initialization
 */
template<> struct DAAL_EXPORT DefaultPartialResultsInit<svdDense> : public PartialResultsInitIface<svdDense>
{
    virtual ~DefaultPartialResultsInit() {}

    /**
     * Initialize partial results
     * \param[in]       input     Input objects for the PCA algorithm
     * \param[in,out]   pres      Partial results of the PCA algorithm
     * \return                    Initialized partial results
     */
    void operator()(const Input &input, services::SharedPtr<PartialResult<svdDense> > &pres);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BASEPARAMETER"></a>
 * \brief Class that specifies the common parameters of the PCA algorithm
 */
template<typename algorithmFPType, Method method = correlationDense>
class DAAL_EXPORT BaseParameter : public daal::algorithms::Parameter
{
public:
    /** Constructs PCA parameters */
    BaseParameter();

    services::SharedPtr<PartialResultsInitIface<method> > initializationProcedure; /**< Functor for partial results initialization */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHPARAMETER"></a>
 * \brief Class that specifies the parameters of the PCA algorithm in the batch computing mode
 */
template<typename algorithmFPType, Method method>
class BatchParameter : public BaseParameter<algorithmFPType, method> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHPARAMETER_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
 * \brief Class that specifies the parameters of the PCA Correlation algorithm in the batch computing mode
 */
template<typename algorithmFPType>
class DAAL_EXPORT BatchParameter<algorithmFPType, correlationDense> : public BaseParameter<algorithmFPType, correlationDense>
{
public:
    /** Constructs PCA parameters */
    BatchParameter(const services::SharedPtr<covariance::BatchIface> &covariance =
                   services::SharedPtr<covariance::Batch<algorithmFPType, covariance::defaultDense> >
               (new covariance::Batch<algorithmFPType, covariance::defaultDense>()));

    services::SharedPtr<covariance::BatchIface> covariance; /*!< Pointer to batch covariance */


    /**
    * Checks batch parameter of the PCA correlation algorithm
    */
    void check() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINEPARAMETER"></a>
  * \brief Class that specifies the parameters of the PCA algorithm in the online computing mode
 */
template<typename algorithmFPType, Method method>
class OnlineParameter : public BaseParameter<algorithmFPType, method> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINEPARAMETER_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
  * \brief Class that specifies the parameters of the PCA Correlation algorithm in the online computing mode
 */
template<typename algorithmFPType>
class DAAL_EXPORT OnlineParameter<algorithmFPType, correlationDense> : public BaseParameter<algorithmFPType, correlationDense>
{
public:
    /** Constructs PCA parameters */
    OnlineParameter(const services::SharedPtr<covariance::OnlineIface> &covariance =
                        services::SharedPtr<covariance::Online<algorithmFPType, covariance::defaultDense> >
                    (new covariance::Online<algorithmFPType, covariance::defaultDense>()),
                    const services::SharedPtr<PartialResultsInitIface<correlationDense> > &initializationProcedure =
                        services::SharedPtr<PartialResultsInitIface<correlationDense> >
                        (new DefaultPartialResultsInit<correlationDense>()));

    services::SharedPtr<covariance::OnlineIface> covariance; /*!< Pointer to Online covariance */
    services::SharedPtr<PartialResultsInitIface<correlationDense> > initializationProcedure; /**< Functor for partial results initialization */

    /**
    * Checks online parameter of the PCA correlation algorithm
    */
    void check() const DAAL_C11_OVERRIDE;
};


/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINEPARAMETER_ALGORITHMFPTYPE_SVDDENSE"></a>
  * \brief Class that specifies the parameters of the PCA SVD algorithm in the online computing mode
 */
template<typename algorithmFPType>
class DAAL_EXPORT OnlineParameter<algorithmFPType, svdDense> : public BaseParameter<algorithmFPType, svdDense>
{
public:
    /** Constructs PCA parameters */
    OnlineParameter(const services::SharedPtr<PartialResultsInitIface<svdDense> > &initializationProcedure =
                        services::SharedPtr<PartialResultsInitIface<svdDense> >
                        (new DefaultPartialResultsInit<svdDense>()));

    services::SharedPtr<PartialResultsInitIface<svdDense> > initializationProcedure; /**< Functor for partial results initialization */

    /**
    * Checks online parameter of the PCA SVD algorithm
    */
    void check() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDPARAMETER"></a>
 * \brief Class that specifies the parameters of the PCA algorithm in the distributed computing mode
 */
template<ComputeStep step, typename algorithmFPType, Method method>
class DistributedParameter : public BaseParameter<algorithmFPType, method> {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDPARAMETER_STEP2MASTER_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
 * \brief Class that specifies the parameters of the PCA Correlation algorithm in the distributed computing mode
 */
template<typename algorithmFPType>
class DAAL_EXPORT DistributedParameter<step2Master, algorithmFPType, correlationDense> : public BaseParameter<algorithmFPType, correlationDense>
{
public:
    /** Constructs PCA parameters */
    DistributedParameter(const services::SharedPtr<covariance::DistributedIface<step2Master> > &covariance =
                             services::SharedPtr<covariance::Distributed<step2Master, algorithmFPType, covariance::defaultDense> >
                         (new covariance::Distributed<step2Master, algorithmFPType, covariance::defaultDense>()));

    services::SharedPtr<covariance::DistributedIface<step2Master> > covariance; /*!< Pointer to Distributed covariance */

    /**
    * Checks distributed parameter of the PCA correlation algorithm
    */
    void check() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDINPUT"></a>
 * \brief Input objects for the PCA algorithm in the distributed processing mode
 */
template<Method method>
class DistributedInput {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTED_INPUT"></a>
 * \brief Input objects for the PCA Correlation algorithm in the distributed processing mode
 */
template<> class DAAL_EXPORT DistributedInput<correlationDense> : public InputIface
{
public:
    DistributedInput();

    /**
     * Sets input objects for the PCA on the second step in the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Input object that corresponds to the given identifier
     */
    void set(Step2MasterInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Gets input objects for the PCA on the second step in the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step2MasterInputId id) const;

    /**
     * Retrieves specific partial result from the input objects of the PCA algorithm on the second step in the distributed processing mode
     * \param[in] id      Identifier of the partial result
     */
    services::SharedPtr<PartialResult<correlationDense> > getPartialResult(size_t id) const;

    /**
     * Adds input objects of the PCA algorithm on the second step in the distributed processing mode
     * \param[in] id      Identifier of the argument
     * \param[in] value   Pointer to the argument
     */
    void add(Step2MasterInputId id, const services::SharedPtr<PartialResult<correlationDense> > &value);

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks the input of the PCA algorithm
    * \param[in] parameter Algorithm %parameter
    * \param[in] method    Computation  method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTED_INPUT"></a>
 * \brief Input objects of the PCA SVD algorithm in the distributed processing mode
 */
template<> class DAAL_EXPORT DistributedInput<svdDense> : public InputIface
{
public:
    DistributedInput();

    /**
     * Sets input objects for the PCA on the second step in the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Input object that corresponds to the given identifier
     */
    void set(Step2MasterInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Gets input objects for the PCA algorithm on the second step in the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step2MasterInputId id) const;

    /**
     * Adds input objects of the PCA algorithm on the second step in the distributed processing mode
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the input object
     */
    void add(Step2MasterInputId id, const services::SharedPtr<PartialResult<svdDense> > &value);

    /**
     * Retrieves specific partial result from the input objects of the PCA algorithm on the second step in the distributed processing mode
     * \param[in] id      Identifier of the partial result
     */
    services::SharedPtr<PartialResult<svdDense> > getPartialResult(size_t id) const;

    /**
    * Checks the input of the PCA algorithm
    * \param[in] parameter Algorithm %parameter
    * \param[in] method    Computation  method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNFeatures() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__RESULT"></a>
 * \brief Provides methods to access results obtained with the PCA algorithm
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();

    virtual ~Result() {};

    /**
    * Gets the results of the PCA algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets results of the PCA algorithm
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
     * Allocates memory for storing partial results of the PCA algorithm
     * \param[in] input Pointer to an object containing input data
     * \param[in] parameter Algorithm parameter
     * \param[in] method Computation method
     */
    template<typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, daal::algorithms::Parameter *parameter, const Method method);

    /**
     * Allocates memory for storing partial results of the PCA algorithm     * \param[in] partialResult Pointer to an object containing input data
     * \param[in] parameter Parameter of the algorithm
     * \param[in] method        Computation method
     */
    template<typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const Method method);

    /**
    * Checks the results of the PCA algorithm
    * \param[in] _input  %Input object of algorithm
    * \param[in] par     Algorithm %parameter
    * \param[in] method  Computation  method
    */
    void check(const daal::algorithms::Input *_input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks the results of the PCA algorithm
    * \param[in] pr             Partial results of the algorithm
    * \param[in] method         Computation method
    * \param[in] parameter      Algorithm %parameter
    */
    void check(const daal::algorithms::PartialResult *pr, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_PCA_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:

    void checkImpl(size_t nFeatures) const;
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
/** @} */
} // namespace interface1
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResultBase;
using interface1::PartialResult;
using interface1::PartialResultImpl;
using interface1::PartialResultsInitIface;
using interface1::DefaultPartialResultsInit;
using interface1::BatchParameter;
using interface1::OnlineParameter;
using interface1::DistributedParameter;
using interface1::DistributedInput;
using interface1::Result;

}
}
} // namespace daal
#endif

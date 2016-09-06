/* file: kmeans_init_types.h */
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
//  Implementation of the K-Means algorithm interface.
//--
*/

#ifndef __KMEANS_INIT_TYPES_H__
#define __KMEANS_INIT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
/**
 * @defgroup kmeans K-means Clustering
 * \copydoc daal::algorithms::kmeans
 * @ingroup analysis
 * @defgroup kmeans_init Initialization
 * \copydoc daal::algorithms::kmeans::init
 * @ingroup kmeans
 * @{
 */
/** \brief Contains classes for computing initial clusters for the K-Means algorithm */
namespace init
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__METHOD"></a>
 * Available methods for computing initial clusters for the K-Means algorithm
 */
enum Method
{
    deterministicDense = 0, /*!< Default: uses first nClusters points as initial clusters */
    defaultDense       = 0, /*!< Synonym of deterministicDense */
    randomDense        = 1, /*!< Uses random nClusters points as initial clusters */
    deterministicCSR   = 2, /*!< Uses first nClusters points as initial clusters for data in a CSR numeric table */
    randomCSR          = 3  /*!< Uses random nClusters points as initial clusters for data in a CSR numeric table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__INPUTID"></a>
 * \brief Available identifiers of input objects for computing initial clusters for the K-Means algorithm
 */
enum InputId
{
    data = 0 /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2MASTERINPUTID"></a>
 * \brief Available identifiers of input objects for computing initial clusters for the K-Means algorithm in the distributed processing mode
 */
enum DistributedStep2MasterInputId
{
    partialResults = 0   /*!< Collection of partial results computed on local nodes */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__PARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of computing initial clusters for the K-Means algorithm in the distributed processing mode
 */
enum PartialResultId
{
    partialClustersNumber = 0, /*!< Table with the number of observations assigned to centroids */
    partialClusters       = 1  /*!< Table with the sum of observations assigned to centroids */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__RESULTID"></a>
 * \brief Available identifiers of the results of computing initial clusters for the K-Means algorithm
 */
enum ResultId
{
    centroids = 0 /*!< Table for cluster centroids */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__KMEANS__INIT__PARAMETER"></a>
 * \brief Parameters for computing initial clusters for the K-Means algorithm
 *
 * \snippet kmeans/kmeans_init_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Main constructor
     *  \param[in] _nClusters     Number of clusters
     *  \param[in] _offset        Offset in the total data set specifying the start of a block stored on a given local node
     *  \param[in] seed           Seed for generating random numbers for the initialization
     */
    Parameter(size_t _nClusters, size_t _offset = 0, size_t seed = 777777);

    /**
     * Constructs parameters of the algorithm that computes initial clusters for the K-Means algorithm
     * by copying another parameters object
     * \param[in] other    Parameters of the K-Means algorithm
     */
    Parameter(const Parameter &other);

    size_t nClusters;     /*!< Number of clusters */
    size_t nRowsTotal;    /*!< Total number of rows in the data set  */
    size_t offset;        /*!< Offset in the total data set specifying the start of a block stored on a given local node */
    size_t seed;          /*!< Seed for generating random numbers for the initialization */

    void check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INPUTIFACE"></a>
 * \brief Interface for the K-Means initialization batch and distributed Input classes
 */
class DAAL_EXPORT InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {};

    virtual size_t getNumberOfFeatures() const = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INPUT"></a>
 * \brief %Input objects for computing initial clusters for the K-Means algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();
    virtual ~Input() {}

    /**
    * Returns input objects for computing initial clusters for the K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets an input object for computing initial clusters for the K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
    * Returns the number of features in the Input data table
    * \return Number of features in the Input data table
    */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks an input object for computing initial clusters for the K-Means algorithm
    * \param[in] par     %Input object
    * \param[in] method  Method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__PARTIALRESULT"></a>
 * \brief Partial results obtained with the compute() method of the K-Means algorithm in the batch processing mode
 */
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:
    PartialResult();

    virtual ~PartialResult() {};

    /**
     * Allocates memory to store partial results of computing initial clusters for the K-Means algorithm
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns a partial result of computing initial clusters for the K-Means algorithm
     * \param[in] id   Identifier of the partial result
     * \return         Partial result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(PartialResultId id) const;

    /**
     * Sets a partial result of computing initial clusters for the K-Means algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the object
     */
    void set(PartialResultId id, const data_management::NumericTablePtr &ptr);

    /**
    * Returns the number of features in the result table of the K-Means algorithm
    * \return Number of features in the result table of the K-Means algorithm
    */
    size_t getNumberOfFeatures() const;

    /**
    * Checks a partial result of computing initial clusters for the K-Means algorithm
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks a partial result of computing initial clusters for the K-Means algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

     /**
     * Returns the serialization tag of a partial result
     * \return         Serialization tag of the partial result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_KMEANS_INIT_PARTIAL_RESULT_ID; }

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
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__RESULT"></a>
 * \brief Results obtained with the compute() method that computes initial clusters
 *  for the K-Means algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of computing initial clusters for the K-Means algorithm
     * \param[in] input        Pointer to the input structure
     * \param[in] parameter    Pointer to the parameter structure
     * \param[in] method       Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Allocates memory to store the results of computing initial clusters for the K-Means algorithm
     * \param[in] partialResult Pointer to the partial result structure
     * \param[in] parameter     Pointer to the parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns the result of computing initial clusters for the K-Means algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of computing initial clusters for the K-Means algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
    * Checks the result of computing initial clusters for the K-Means algorithm
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks the result of computing initial clusters for the K-Means algorithm
    * \param[in] pres    Partial results of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

     /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_KMEANS_INIT_RESULT_ID; }

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
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * \brief %Input objects for computing initials clusters for the K-Means
 *  algorithm in the second step of the distributed processing mode
 */
class DAAL_EXPORT DistributedStep2MasterInput : public InputIface
{
public:
    DistributedStep2MasterInput();

    virtual ~DistributedStep2MasterInput() {}

    /**
    * Returns an input object for computing initial clusters for the K-Means algorithm
    * in the second step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(DistributedStep2MasterInputId id) const;

    /**
    * Sets an input object for computing initial clusters for the K-Means algorithm
    * in the second step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(DistributedStep2MasterInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Adds a value to the data collection of input objects for computing initial clusters for the K-Means algorithm
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the parameter
     * \param[in] value Pointer to the new parameter value
     */
    void add(DistributedStep2MasterInputId id, const services::SharedPtr<PartialResult> &value);

    /**
    * Returns the number of features in the Input data table in the second step of the distributed processing mode
    * \return Number of features in the Input data table
    */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks an input object for computing initial clusters for the K-Means algorithm
    * in the second step of the distributed processing mode
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResult;
using interface1::Result;
using interface1::DistributedStep2MasterInput;

} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
#endif

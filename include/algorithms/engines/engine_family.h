/* file: engine_family.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation engine.
//--
*/

#ifndef __ENGINE_FAMILY_H__
#define __ENGINE_FAMILY_H__

#include "algorithms/engines/engine.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
/**
 * @ingroup engines
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__FAMILYBATCHBASE"></a>
 *  \brief Class representing an engine that has collection of independent streams obtained from RNGs from same family.
 */
class DAAL_EXPORT FamilyBatchBase : public engines::BatchBase
{
public:
    typedef engines::BatchBase super;

    typedef super::InputType  InputType;
    typedef super::ResultType ResultType;

    FamilyBatchBase();
    virtual ~FamilyBatchBase() {}

    /**
     * Add 'numberOfStreams' random number streams to the engine
     * \param[in] numberOfStreams  Number of streams to add
     *
     * \return Status of computations
     */
    services::Status add(size_t numberOfStreams);

    /**
     * Get engine with one stream from collection.
     * \param[in] i index of stream from the collection
     *
     * \return SharedPtr to engine instance with ith stream from the collection
     */
    services::SharedPtr<FamilyBatchBase> get(size_t i) const;

    /**
     * Return the number of streams in the collection.
     *
     * \return Number of streams in the collection
     */
    size_t getNumberOfStreams() const;

    /**
     * Return the maximum number of streams can be obtained from RNGs in given family
     *
     * \return Maximum number of streams can be obtained from RNGs in given family
     */
    size_t getMaxNumberOfStreams() const;

protected:
    virtual services::Status addImpl(size_t numberOfStreams) { return services::Status(); }
    virtual services::SharedPtr<FamilyBatchBase> getImpl(size_t i) const { return services::SharedPtr<FamilyBatchBase>(); }
    virtual size_t getNumberOfStreamsImpl() const { return 0; }
    virtual size_t getMaxNumberOfStreamsImpl() const { return 0; }

    FamilyBatchBase(const FamilyBatchBase &other);
};
typedef services::SharedPtr<FamilyBatchBase> FamilyEnginePtr;

} // namespace interface1
using interface1::FamilyBatchBase;
using interface1::FamilyEnginePtr;
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif

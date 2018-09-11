/* file: loss_layer_forward.h */
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
//  Implementation of the forward loss layer.
//--
*/

#ifndef __LOSS_LAYER_FORWARD_H__
#define __LOSS_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace loss
{
namespace forward
{
namespace interface1
{
/**
 * @defgroup loss_forward_batch Batch
 * @ingroup loss_forward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__FORWARD__BATCH"></a>
 * \brief Provides methods for the forward loss layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOSSFORWARD-ALGORITHM">Forward loss layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the forward loss layer, double or float
 *
 * \par Enumerations
 *      - \ref forward::InputId           Identifiers of input objects for the forward loss layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward loss layer
 *      - \ref forward::ResultLayerDataId Identifiers of extra results computed by the forward loss layer
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
class Batch : public layers::forward::LayerIfaceImpl
{
public:
    typedef layers::forward::LayerIfaceImpl super;

    typedef algorithms::neural_networks::layers::loss::forward::Input  InputType;
    typedef algorithms::neural_networks::layers::loss::forward::Result ResultType;

    /** Default constructor */
    Batch()
    {
        initialize();
    };

    /**
     * Constructs a forward loss layer by copying input objects
     * and parameters of another forward loss layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the layer
     */
    Batch(const Batch &other)
    {
        initialize();
    }

    /**
     * Returns a pointer to a newly allocated forward loss layer
     * with a copy of the input objects and parameters for this forward loss layer
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch > clone() const
    {
        return services::SharedPtr<Batch >(cloneImpl());
    }

protected:
    virtual Batch *cloneImpl() const DAAL_C11_OVERRIDE = 0;

    void initialize()
    {}
};
/** @} */
} // namespace interface1
using interface1::Batch;
} // namespace forward
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif

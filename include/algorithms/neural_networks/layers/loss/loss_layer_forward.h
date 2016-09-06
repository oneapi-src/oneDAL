/* file: loss_layer_forward.h */
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
 * \n<a href="DAAL-REF-LOSSFORWARD-ALGORITHM">Forward loss layer description and usage models</a>
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
class Batch : public layers::forward::LayerIface
{
public:
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

    /**
     * Returns the layer that corresponds to this layer on the prediction stage
     * \return The layer that corresponds to this layer on the prediction stage
     */
    virtual services::SharedPtr<layers::forward::LayerIface> getLayerForPrediction() const = 0;
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

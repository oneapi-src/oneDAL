/* file: loss_layer_backward.h */
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
//  Implementation of the interface for the backward loss layer in the batch processing mode
//--
*/

#ifndef __LOSS_LAYER_BACKWARD_H__
#define __LOSS_LAYER_BACKWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_backward_types.h"

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
namespace backward
{
namespace interface1
{
/**
 * @defgroup loss_backward_batch Batch
 * @ingroup loss_backward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__BACKWARD__BATCH"></a>
 * \brief Provides methods for the backward loss layer in the batch processing mode
 * \n<a href="DAAL-REF-LOSSBACKWARD-ALGORITHM">Backward loss layer description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the backward loss layer, double or float
 *
 * \par Enumerations
 *      - \ref backward::InputId           Identifiers of input objects for the backward loss layer
 *      - \ref backward::InputLayerDataId  Identifiers of extra results computed by the forward loss layer
 *      - \ref backward::ResultId          Identifiers of result objects for the backward loss layer
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref forward::interface1::Batch "forward::Batch" class
 */
class Batch : public layers::backward::LayerIface
{
public:
    /** Default constructor */
    Batch()
    {
        initialize();
    };

    /**
     * Constructs a backward loss layer by copying input objects
     * and parameters of another backward loss layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the layer
     */
    Batch(const Batch &other)
    {
        initialize();
    }

    /**
     * Returns a pointer to a newly allocated backward loss layer
     * with a copy of the input objects and parameters for this backward loss layer
     * in the batch processing mode
     * \return Pointer to the newly allocated layer
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
} // namespace backward
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif

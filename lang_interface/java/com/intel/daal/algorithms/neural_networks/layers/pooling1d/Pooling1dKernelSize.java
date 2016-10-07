/* file: Pooling1dKernelSize.java */
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

package com.intel.daal.algorithms.neural_networks.layers.pooling1d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DKERNELSIZE"></a>
 * \brief Data structure representing the size of the 1D subtensor from which the element is computed
 */
public final class Pooling1dKernelSize {
    private long[] size; /*!< Array of sizes of the 1D subtensor from which the element is computed */

    /**
    * Constructs KernelSize with parameters
    * @param first  Size of the first dimension of the 1D subtensor
    */
    public Pooling1dKernelSize(long first) {
        size = new long[1];
        size[0] = first;
    }

    /**
     *  Sets the the array of sizes of the 1D subtensor from which the element is computed
    * @param first  Size of the first dimension of the 1D subtensor
     */
    public void setSize(long first) {
        size[0] = first;
    }

    /**
    *  Gets the array of sizes of the 1D subtensor from which the element is computed
    * @return Array of sizes of the 1D subtensor from which the element is computed
    */
    public long[] getSize() {
        return size;
    }
}

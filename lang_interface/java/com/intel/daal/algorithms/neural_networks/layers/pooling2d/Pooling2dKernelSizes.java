/* file: Pooling2dKernelSizes.java */
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

package com.intel.daal.algorithms.neural_networks.layers.pooling2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DKERNELSIZES"></a>
 * \brief Data structure representing the size of the 2D subtensor from which the element is computed
 */
public final class Pooling2dKernelSizes {
    private long[] size; /*!< Array of sizes of the 2D subtensor from which the element is computed */

    /**
    * Constructs KernelSizes with parameters
    * @param first  Size of the first dimension of the 2D subtensor
    * @param second  Size of the second dimension of the 2D subtensor
    */
    public Pooling2dKernelSizes(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Sets the the array of sizes of the 2D subtensor from which the element is computed
    * @param first  Size of the first dimension of the 2D subtensor
    * @param second  Size of the second dimension of the 2D subtensor
    */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of sizes of the 2D subtensor from which the element is computed
    * @return Array of sizes of the 2D subtensor from which the element is computed
    */
    public long[] getSize() {
        return size;
    }
}

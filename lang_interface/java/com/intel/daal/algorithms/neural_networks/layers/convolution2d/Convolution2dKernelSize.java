/* file: Convolution2dKernelSize.java */
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

package com.intel.daal.algorithms.neural_networks.layers.convolution2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__CONVOLUTION2DKERNELSIZE"></a>
 * \brief Data structure representing the sizes of the two-dimensional kernel subtensor for the backward 2D convolution
  *       layer and results for the forward 2D convolution layer
 */
public final class Convolution2dKernelSize {
    private long[] size; /*!< Array of sizes of the two-dimensional kernel subtensor */

    /**
    * Constructs Convolution2dKernelSize with parameters
    * @param first  The first size of the two-dimensional kernel subtensor
    * @param second The second size of the two-dimensional kernel subtensor
    */
    public Convolution2dKernelSize(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the the array of sizes of the two-dimensional kernel subtensor
    * @param first  The first size of the two-dimensional kernel subtensor
    * @param second The second size of the two-dimensional kernel subtensor
     */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of sizes of the two-dimensional kernel subtensor
    * @return Array of sizes of the two-dimensional kernel subtensor
    */
    public long[] getSize() {
        return size;
    }
}

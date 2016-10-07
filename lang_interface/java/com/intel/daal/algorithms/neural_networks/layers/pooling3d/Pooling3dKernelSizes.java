/* file: Pooling3dKernelSizes.java */
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

package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DKERNELSIZES"></a>
 * \brief Data structure representing the size of the three-dimensional subtensor
 */
public final class Pooling3dKernelSizes {
    private long[] size; /*!< Array of sizes of the three-dimensional kernel subtensor */

    /**
    * Constructs KernelSizes with parameters
    * @param first  Size of the first dimension of the three-dimensional subtensor
    * @param second Size of the second dimension of the three-dimensional subtensor
    * @param third  Size of the third dimension of the three-dimensional subtensor
    */
    public Pooling3dKernelSizes(long first, long second, long third) {
        size = new long[3];
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
     *  Sets the the array of sizes of the three-dimensional kernel subtensor
    * @param first  Size of the first dimension of the three-dimensional subtensor
    * @param second Size of the second dimension of the three-dimensional subtensor
    * @param third  Size of the third dimension of the three-dimensional subtensor
     */
    public void setSize(long first, long second, long third) {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Gets the array of sizes of the three-dimensional kernel subtensor
    * @return Array of sizes of the three-dimensional kernel subtensor
    */
    public long[] getSize() {
        return size;
    }
}

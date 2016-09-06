/* file: Pooling3dStrides.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DSTRIDES"></a>
 * \brief Data structure representing the intervals on which the subtensors for pooling are selected
 */
public final class Pooling3dStrides {
    private long[] size;     /*!< Array of intervals on which the subtensors for pooling are selected */

    /**
    * Constructs Strides with parameters
    * @param first  The first interval on which the subtensors for pooling are selected
    * @param second The second interval on which the subtensors for pooling are selected
    * @param third The third interval on which the subtensors for pooling are selected
    */
    public Pooling3dStrides(long first, long second, long third) {
        size = new long[3];
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
     *  Sets the array of intervals on which the subtensors for pooling are selected
    * @param first The first interval on which the subtensors for pooling are selected
    * @param second The second interval on which the subtensors for pooling are selected
    * @param third The third interval on which the subtensors for pooling are selected
     */
    public void setSize(long first, long second, long third) {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Gets the array of intervals on which the subtensors for pooling are selected
    * @return Array of intervals on which the subtensors for pooling are selected
    */
    public long[] getSize() {
        return size;
    }
}

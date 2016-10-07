/* file: Pooling2dStrides.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DSTRIDES"></a>
 * \brief Data structure representing the intervals on which the subtensors for two-dimensional pooling are computed
 */
public final class Pooling2dStrides {
    private long[] size;     /*!< Array of intervals on which the subtensors for two-dimensional pooling are selected */

    /**
    * Constructs Strides with parameters
    * @param first   Interval over the first dimension on which the two-dimensional pooling is performed
    * @param second  Interval over the second dimension on which the two-dimensional pooling is performed
    */
    public Pooling2dStrides(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of intervals on which the subtensors for two-dimensional pooling are selected
    * @param first    Interval over the first dimension on which the two-dimensional pooling is performed
    * @param second   Interval over the second dimension on which the two-dimensional pooling is performed
     */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of intervals on which the subtensors for two-dimensional pooling are selected
    * @return Array of intervals on which the subtensors for two-dimensional pooling are selected
    */
    public long[] getSize() {
        return size;
    }
}

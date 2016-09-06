/* file: SpatialPooling2dIndices.java */
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

package com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_POOLING2D__SPATIALPOOLING2DINDICES"></a>
 * \brief Data structure representing the indices of the dimension on which two-dimensional pooling is performed
 */
public final class SpatialPooling2dIndices {
    private long[] size;     /*!< Array of indices of the dimension on which two-dimensional pooling is performed */

    /**
    * Constructs SpatialPooling2dIndices with parameters
    * @param first   The first dimension index
    * @param second  The second dimension index
    */
    public SpatialPooling2dIndices(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of indices of the dimension on which two-dimensional pooling is performed
    * @param first   The first dimension index
    * @param second  The second dimension index
    */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of indices of the dimension on which two-dimensional pooling is performed
    * @return Array of indices of the dimension on which two-dimensional pooling is performed
    */
    public long[] getSize() {
        return size;
    }
}

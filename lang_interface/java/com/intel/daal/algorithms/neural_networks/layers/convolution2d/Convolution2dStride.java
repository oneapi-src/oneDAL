/* file: Convolution2dStride.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__CONVOLUTION2DSTRIDE"></a>
 * \brief Data structure representing the intervals on which the kernel should be applied to the input
 */
public final class Convolution2dStride {
    private long[] size;     /*!< Array of intervals on which the kernel should be applied to the input */

    /**
    * Constructs Convolution2dStride with parameters
    * @param first  The first interval on which the kernel should be applied to the input
    * @param second The second interval on which the kernel should be applied to the input
    */
    public Convolution2dStride(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of intervals on which the kernel should be applied to the input
    * @param first  The first interval on which the kernel should be applied to the input
    * @param second The second interval on which the kernel should be applied to the input
     */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of intervals on which the kernel should be applied to the input
    * @return Array of intervals on which the kernel should be applied to the input
    */
    public long[] getSize() {
        return size;
    }
}

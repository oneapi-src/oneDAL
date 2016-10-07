/* file: LocallyConnected2dPaddings.java */
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

package com.intel.daal.algorithms.neural_networks.layers.locallyconnected2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__LOCALLYCONNECTED2DPADDING"></a>
 * \brief Data structure representing the number of data to be implicitly added to the subtensor
 */
public final class LocallyConnected2dPaddings {
    private long[] size;     /*!< Array of numbers of data to be implicitly added to the subtensor */

    /**
    * Constructs LocallyConnected2dPaddings with parameters
    * @param first  The first number of data to be implicitly added to the subtensor
    * @param second The second number of data to be implicitly added to the subtensor
    */
    public LocallyConnected2dPaddings(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of numbers of data to be implicitly added to the subtensor
    * @param first  The first number of data to be implicitly added to the subtensor
    * @param second The second number of data to be implicitly added to the subtensor
     */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of numbers of data to be implicitly added to the subtensor
    * @return Array of numbers of data to be implicitly added to the subtensor
    */
    public long[] getSize() {
        return size;
    }
}

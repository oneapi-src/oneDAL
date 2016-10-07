/* file: Pooling3dPaddings.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DPADDINGS"></a>
 * \brief Data structure representing the number of data elements to implicitly add to each size of
 *        the three-dimensional subtensor on which pooling is performed
 */
public final class Pooling3dPaddings {
    private long[] size;     /*!< Array of numbers of data elements to implicitly add to each size of
                                  the three-dimensional subtensor on which pooling is performed */
    /**
    * Constructs Paddings with parameters
    * @param first  The first number of data elements to implicitly add to each size of
    *               the three-dimensional subtensor on which pooling is performed
    * @param second The second number of data elements to implicitly add to each size of
    *               the three-dimensional subtensor on which pooling is performed
    * @param third The third number of data elements to implicitly add to each size of
    *               the three-dimensional subtensor on which pooling is performed
    */
    public Pooling3dPaddings(long first, long second, long third) {
        size = new long[3];
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Sets the array of numbers of data elements to implicitly add to each size of
    *  the three-dimensional subtensor on which pooling is performed
    * @param first  The first number of data elements to implicitly add to each size of
    *        the three-dimensional subtensor on which pooling is performed
    * @param second The second number of data elements to implicitly add to each size of
    *        the three-dimensional subtensor on which pooling is performed
    * @param third The third number of data elements to implicitly add to each size of
    *        the three-dimensional subtensor on which pooling is performed
    */
    public void setSize(long first, long second, long third) {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Gets the array of numbers of data elements to implicitly add to each size of
    *  the three-dimensional subtensor on which pooling is performed
    * @return Array of numbers of data elements to implicitly add to each size of
    *         he three-dimensional subtensor on which pooling is performed
    */
    public long[] getSize() {
        return size;
    }
}

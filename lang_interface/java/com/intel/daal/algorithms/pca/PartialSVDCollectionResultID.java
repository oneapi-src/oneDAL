/* file: PartialSVDCollectionResultID.java */
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

package com.intel.daal.algorithms.pca;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALSVDCOLLECTIONRESULTID"></a>
 * @brief Available identifiers of partial results of the %SVD method of the PCA algorithm
 */
public final class PartialSVDCollectionResultID {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    public PartialSVDCollectionResultID(int value) {
        _value = value;
    }

    /** Returns value of input identifier
      * \return value of input identifier */
    public int getValue() {
        return _value;
    }

    private static final int svdAuxiliaryDataId = 3;

    /*!< Auxiliary data */
    public static final PartialSVDCollectionResultID svdAuxiliaryData = new PartialSVDCollectionResultID(
            svdAuxiliaryDataId);
}

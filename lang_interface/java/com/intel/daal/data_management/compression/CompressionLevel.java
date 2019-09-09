/* file: CompressionLevel.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/**
 * @ingroup data_compression
 * @{
 */
package com.intel.daal.data_management.compression;

import java.lang.annotation.Native;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__COMPRESSIONLEVEL"></a>
 * @brief Compression levels
 */
public final class CompressionLevel {
    private int _value;

    /**
     * Constructs the compression level object using the provided value
     * @param value     Value corresponding to the compression level object
     */
    public CompressionLevel(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the compression level object
     * @return Value corresponding to the compression level object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int DefaultLevelValue = -1;
    @Native private static final int Level0Value       = 0;
    @Native private static final int Level1Value       = 1;
    @Native private static final int Level2Value       = 2;
    @Native private static final int Level3Value       = 3;
    @Native private static final int Level4Value       = 4;
    @Native private static final int Level5Value       = 5;
    @Native private static final int Level6Value       = 6;
    @Native private static final int Level7Value       = 7;
    @Native private static final int Level8Value       = 8;
    @Native private static final int Level9Value       = 9;

    public static final CompressionLevel DefaultLevel = new CompressionLevel(
            DefaultLevelValue);                                                            /*!< Default compression level */
    public static final CompressionLevel Level0       = new CompressionLevel(
            Level0Value);                                                                  /*!< Minimum compression level, maximum speed */
    public static final CompressionLevel Level1       = new CompressionLevel(Level1Value); /*!< Level 1 */
    public static final CompressionLevel Level2       = new CompressionLevel(Level2Value); /*!< Level 2 */
    public static final CompressionLevel Level3       = new CompressionLevel(Level3Value); /*!< Level 3 */
    public static final CompressionLevel Level4       = new CompressionLevel(Level4Value); /*!< Level 4 */
    public static final CompressionLevel Level5       = new CompressionLevel(Level5Value); /*!< Level 5 */
    public static final CompressionLevel Level6       = new CompressionLevel(Level6Value); /*!< Level 6 */
    public static final CompressionLevel Level7       = new CompressionLevel(Level7Value); /*!< Level 7 */
    public static final CompressionLevel Level8       = new CompressionLevel(Level8Value); /*!< Level 8 */
    public static final CompressionLevel Level9       = new CompressionLevel(
            Level9Value);                                                                  /*!< Maximum compression level, minimum speed */
}
/** @} */

/* file: compression_batch.cpp */
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

/*
!  Content:
!    C++ example of compression in the batch processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COMPRESSION_BATCH"></a>
 * \example compression_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace data_management;

string datasetFileName  = "../data/batch/logitboost_train.csv";

DataBlock rawData;        /* Data to compress */
DataBlock compressedData;  /* Result of compression */
DataBlock deCompressedData;    /* Result of decompression */

void prepareMemory();
void releaseMemory();
void printCRC32();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read data from a file and allocate memory */
    prepareMemory();

    /* Create a compressor */
    Compressor<zlib> compressor;
    compressor.parameter.gzHeader = true;
    compressor.parameter.level = level9;

    /* Create a stream for compression */
    CompressionStream comprStream(&compressor);

    /* Write raw data to the compression stream and compress if needed */
    comprStream << rawData;

    /* Get the size of the compressed data */
    compressedData.setSize(comprStream.getCompressedDataSize());

    /* Allocate memory to store the compressed data */
    compressedData.setPtr(new byte[compressedData.getSize()]);

    /* Store the compressed data */
    comprStream.copyCompressedArray(compressedData);

    /* Create a decompressor */
    Decompressor<zlib> decompressor;
    decompressor.parameter.gzHeader = true;

    /* Create a stream for decompression */
    DecompressionStream deComprStream(&decompressor);

    /* Write the compressed data to the decompression stream and decompress it */
    deComprStream << compressedData;

    /* Get the size of the decompressed data */
    deCompressedData.setSize(deComprStream.getDecompressedDataSize());

    /* Allocate memory to store the decompressed data */
    deCompressedData.setPtr(new byte[deCompressedData.getSize()]);

    /* Store the decompressed data */
    deComprStream.copyDecompressedArray(deCompressedData);

    /* Compute and print checksums for raw data and the decompressed data */
    printCRC32();

    releaseMemory();

    return 0;
}

void prepareMemory()
{
    /* Allocate memory for raw data and read an input file */
    byte *data;
    rawData.setSize(readTextFile(datasetFileName, &data));
    rawData.setPtr(data);
}

void printCRC32()
{
    unsigned int crcRawData = 0;
    unsigned int crcDecompressedData = 0;

    /* Compute checksums for raw data and the decompressed data */
    crcRawData = getCRC32(rawData.getPtr(), crcRawData, rawData.getSize());
    crcDecompressedData = getCRC32(deCompressedData.getPtr(), crcDecompressedData, deCompressedData.getSize());

    cout << endl << "Compression example program results:" << endl << endl;

    cout << "Raw data checksum:    0x" << hex << crcRawData << endl;
    cout << "Decompressed data checksum: 0x" << hex << crcDecompressedData << endl;

    if (rawData.getSize() != deCompressedData.getSize())
    {
        cout << "ERROR: Decompressed data size mismatches with the raw data size" << endl;
    }
    else if (crcRawData != crcDecompressedData)
    {
        cout << "ERROR: Decompressed data CRC mismatches with the raw data CRC" << endl;
    }
    else
    {
        cout << "OK: Decompressed data CRC matches with the raw data CRC" << endl;
    }
}

void releaseMemory()
{
    if(compressedData.getPtr())
    {
        delete [] compressedData.getPtr();
    }
    if(deCompressedData.getPtr())
    {
        delete [] deCompressedData.getPtr();
    }
    if(rawData.getPtr())
    {
        delete [] rawData.getPtr();
    }
}

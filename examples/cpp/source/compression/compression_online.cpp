/* file: compression_online.cpp */
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
!    C++ example of compression in the online processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COMPRESSION_ONLINE"></a>
 * \example compression_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace data_management;

string datasetFileName  = "../data/online/logitboost_train.csv";

DataBlock sentDataStream;        /* Data stream to compress and send */
DataBlock uncompressedDataBlock; /* Current block of the data stream to compress */
DataBlock compressedDataBlock;   /* Current compressed block of data */
DataBlock receivedDataStream;    /* Received uncompressed data stream */

queue<DataBlock> sendReceiveQueue; /* Queue for sending and receiving compressed data blocks */

const size_t maxDataBlockSize = 16384; /* Maximum size of a data block */
const size_t userDefinedBlockSize = 7000; /* Size for read data from a decompression stream */

bool getDataBlock(DataBlock &block);
void sendDataBlock(DataBlock *block);
bool receiveDataBlock(DataBlock &block);
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
    CompressionStream compressionStream(&compressor);

    /* Receive data by blocks from sentDataStream for further compression and send it */
    while(getDataBlock(uncompressedDataBlock))
    {
        /* Put a data block to compressionStream and compress if needed */
        compressionStream << uncompressedDataBlock;

        /* Get access to compressed blocks stored in compressionStream without an actual compressed data copy */
        services::SharedPtr<DataBlockCollection> compressedBlocks = compressionStream.getCompressedBlocksCollection();

        /* Send compressed blocks stored in compressionStream */
        for(size_t i = 0; i < compressedBlocks->size(); i++)
        {
            /* Send the current compressed block from compressionStream */
            sendDataBlock((*compressedBlocks)[i].get());
        }
    }

    /* Create a decompressor */
    Decompressor<zlib> decompressor;
    decompressor.parameter.gzHeader = true;

    /* Create a stream for decompression */
    DecompressionStream decompressionStream(&decompressor);

    /* Actual size of decompressed data currently read from decompressionStream */
    size_t readSize = 0;

    /* Receive compressed data by blocks */
    while (receiveDataBlock(compressedDataBlock))
    {
        /* Write a received block to decompressionStream */
        decompressionStream << compressedDataBlock;

        /* Asynchronous read from decompressionStream */
        do
        {
            /* Read userDefinedBlockSize bytes from decompressionStream to the end of receivedDataStream */
            readSize = decompressionStream.copyDecompressedArray(receivedDataStream.getPtr() + receivedDataStream.getSize(),
                                                                 userDefinedBlockSize);

            /* Update the actual data size in receivedDataStream */
            receivedDataStream.setSize(receivedDataStream.getSize() + readSize);
        }
        while (readSize != 0);
    }

    /* Compute and print checksums for sentDataStream and receivedDataStream */
    printCRC32();

    releaseMemory();

    return 0;
}

void prepareMemory()
{
    /* Allocate sentDataStream and read an input file */
    byte *data;
    sentDataStream.setSize(readTextFile(datasetFileName, &data));
    sentDataStream.setPtr(data);

    /* Allocate memory for receivedDataStream */
    byte *receivedData = (byte *)daal::services::daal_malloc(sentDataStream.getSize());
    checkAllocation(receivedData);
    receivedDataStream.setPtr(receivedData);
}

bool getDataBlock(DataBlock &block)
{
    static size_t availableDataSize = sentDataStream.getSize();

    /* Calculate the current block size and ptr */
    if(availableDataSize >= maxDataBlockSize)
    {
        block.setSize(maxDataBlockSize);
        block.setPtr(sentDataStream.getPtr() + sentDataStream.getSize() - availableDataSize);
        availableDataSize -= maxDataBlockSize;
    }
    else if((availableDataSize < maxDataBlockSize) && (availableDataSize > 0))
    {
        block.setSize(availableDataSize);
        block.setPtr(sentDataStream.getPtr() + sentDataStream.getSize() - availableDataSize);
        availableDataSize = 0;
    }
    else
    {
        return false;
    }

    return true;
}

void sendDataBlock(DataBlock *block)
{
    DataBlock currentBlock;

    /* Allocate memory for the current compressed block in the queue */
    byte *tmp = (byte *)daal::services::daal_malloc(block->getSize());
    checkAllocation(tmp);
    currentBlock.setPtr(tmp);

    /* Copy an incoming block to the current compressed block */
    currentBlock.setSize(block->getSize());
    copyBytes(currentBlock.getPtr(), block->getPtr(), currentBlock.getSize());

    /* Push the current compressed block to the queue */
    sendReceiveQueue.push(currentBlock);

    return;
}

bool receiveDataBlock(DataBlock &block)
{
    DataBlock currentBlock;

    /* Stop at the end of the queue */
    if(sendReceiveQueue.empty())
    {
        return false;
    }

    /* Receive the current compressed block from the queue */
    currentBlock = sendReceiveQueue.front();
    block.setSize(currentBlock.getSize());

    if(block.getPtr()) { delete[] block.getPtr(); }
    block.setPtr(new byte[block.getSize()]);

    copyBytes(block.getPtr(), currentBlock.getPtr(), block.getSize());

    /* Release memory of the current compressed block in the queue */
    daal::services::daal_free(currentBlock.getPtr());

    sendReceiveQueue.pop();

    return true;
}

void printCRC32()
{
    unsigned int crcSentDataStream = 0;
    unsigned int crcReceivedDataStream = 0;

    /* Compute checksums for full input data and full received data */
    crcSentDataStream = getCRC32(sentDataStream.getPtr(), crcSentDataStream, sentDataStream.getSize());
    crcReceivedDataStream = getCRC32(receivedDataStream.getPtr(), crcReceivedDataStream, receivedDataStream.getSize());

    cout << endl << "Compression example program results:" << endl << endl;

    cout << "Input data checksum:    0x" << hex << crcSentDataStream << endl;
    cout << "Received data checksum: 0x" << hex << crcReceivedDataStream << endl;

    if (sentDataStream.getSize() != receivedDataStream.getSize())
    {
        cout << "ERROR: Received data size mismatches with the sent data size" << endl;
    }
    else if (crcSentDataStream != crcReceivedDataStream)
    {
        cout << "ERROR: Received data CRC mismatches with the sent data CRC" << endl;
    }
    else
    {
        cout << "OK: Received data CRC matches with the sent data CRC" << endl;
    }
}

void releaseMemory()
{
    if(compressedDataBlock.getPtr())
    {
        delete [] compressedDataBlock.getPtr();
    }
    if(receivedDataStream.getPtr())
    {
        daal::services::daal_free(receivedDataStream.getPtr());
    }
    if(sentDataStream.getPtr())
    {
        delete [] sentDataStream.getPtr();
    }
}

#!/bin/bash

wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo add-apt-repository -y "deb https://apt.repos.intel.com/oneapi all main"
sudo add-apt-repository -y ppa:intel-opencl/intel-opencl
sudo apt-get update
sudo apt-get install              \
    intel-oneapi-common-vars      \
    intel-oneapi-common-licensing \
    intel-oneapi-tbb-devel        \
    intel-oneapi-dpcpp-compiler   \
    intel-oneapi-dev-utilities    \
    intel-oneapi-libdpstd-devel   \
    cmake
sudo bash -c 'echo libintelocl.so > /etc/OpenCL/vendors/intel-cpu.icd'
sudo mv -f /opt/intel/inteloneapi/compiler/latest/linux/lib/oclfpga /opt/intel/inteloneapi/compiler/latest/linux/lib/oclfpga_

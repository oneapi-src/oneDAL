# Function to display help
#!/bin/bash

# Function to display help
show_help() {
    echo "Usage: $0 [-h]"
    echo "  -h  Display this information"
    echo "  Set CC and CXX environment variables to change the compiler. Default is GNU."
}

# Check for command-line options
while getopts ":h" opt; do
    case $opt in
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done

# Set default values for CXX and CC
CXX="${CXX:-g++}"
CC="${CC:-gcc}"

echo "CXX is set to: $CXX"
echo "CC is set to: $CC"

TBB_VERSION="v2021.10.0"

arch=$(uname -m)
if [ "${arch}" == "x86_64" ]; then
    arch_dir="intel64"
elif [ "${arch}" == "aarch64" ]; then
    arch_dir="arm"
else
    arch_dir=${arch}
fi

sudo apt-get update
sudo apt-get install build-essential gcc gfortran cmake -y
git clone --depth 1 --branch ${TBB_VERSION} https://github.com/oneapi-src/oneTBB.git onetbb-src

CoreCount=$(lscpu -p | grep -Ev '^#' | wc -l)

rm -rf __deps/tbb
pushd onetbb-src
mkdir build
pushd build
cmake -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT_PROTOTYPES=OFF -DCMAKE_INSTALL_PREFIX=../../__deps/tbb .. 
make -j${CoreCount} 
make install
popd
popd
rm -rf onetbb-src

pushd __deps/tbb
    mkdir -p lnx
    mv lib/ lnx/
    mv include/ lnx/ 
    pushd lnx
        mkdir -p lib/${arch_dir}/gcc4.8
        mv lib/libtbb* lib/${arch_dir}/gcc4.8
    popd
popd 

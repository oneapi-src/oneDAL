import argparse
import pytest

def get_context(device):
    from daal4py.oneapi import sycl_context
    return sycl_context(device, host_offload_on_fail=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run scikit-learn tests with device context manager')
    parser.add_argument('-a', '--algorithm', type=str, help='datasets which should be downloaded')
    parser.add_argument('-d', '--device', type=str, help='device name', choices=['host', 'cpu', 'gpu'])
    args = parser.parse_args()

    with get_context(args.device):
        pytest.main(["-s", "--disable-warnings", "-v", "test_%s.py" % (args.algorithm)])

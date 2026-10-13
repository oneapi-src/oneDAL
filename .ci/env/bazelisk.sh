#!/bin/bash
#===============================================================================
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

BAZELISK_VERSION=v1.29.0

# SHA256 of the release assets for the version above, so the usual install path
# needs no GitHub API call at all: the API is the only rate-limited endpoint
# here, it is what took `install-bazel` down on Azure builds 63813 and 63876 and
# on the Windows nightly of 2026-10-12, and the Azure jobs cannot authenticate.
# The digest is what the API would be asked for anyway, so pinning it loses no
# verification -- the download is still checked against it.
#
# Refresh together with BAZELISK_VERSION:
#   gh api repos/bazelbuild/bazelisk/releases/tags/<version> \
#     --jq '.assets[] | "\(.name) \(.digest)"'
#
# `BAZELISK_PINNED_VERSION` is what the digests below belong to. Renovate bumps
# the version strings but cannot update an asset digest, so when the two differ
# the pins are simply ignored and the release metadata is fetched as before --
# a Renovate bump keeps working unattended, it only loses the API-free path
# until the digests are refreshed.
BAZELISK_PINNED_VERSION=v1.29.0
BAZELISK_SHA256_bazelisk_linux_amd64=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992
BAZELISK_SHA256_bazelisk_linux_arm64=e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010

# Bazelisk itself always runs on the CI *exec* host (even when the build
# cross-compiles to another target arch, e.g. the riscv64 job below), so pick
# the asset matching the host running this script, not the oneDAL target arch.
host_arch=$(uname -m)

case "${host_arch}" in
  x86_64|amd64)  arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *)
    echo ":error: Unsupported host architecture for Bazelisk: ${host_arch}" >&2
    exit 1
    ;;
esac

BAZELISK_ASSET="bazelisk-linux-${arch}"

# The GitHub API and the release CDN both fail in CI without the install being
# wrong: a dropped connection, a 5xx, a secondary rate limit. Those are retried
# with exponential backoff. A primary rate limit is not retried -- see
# fetch_release_json.
BAZELISK_FETCH_ATTEMPTS=${BAZELISK_FETCH_ATTEMPTS:-5}
BAZELISK_FETCH_DELAY=${BAZELISK_FETCH_DELAY:-5}
# A non-numeric attempt count would make the loop's test fail forever.
if ! [[ "${BAZELISK_FETCH_ATTEMPTS}" =~ ^[1-9][0-9]*$ && "${BAZELISK_FETCH_DELAY}" =~ ^[0-9]+$ ]]; then
  echo ":error: BAZELISK_FETCH_ATTEMPTS must be a positive integer and BAZELISK_FETCH_DELAY a non-negative integer." >&2
  exit 1
fi

# Exit status a fetch uses to say "this will not pass on a retry".
BAZELISK_FATAL_RC=2

function retry {
  local attempt=1
  local delay=${BAZELISK_FETCH_DELAY}
  local rc
  while true; do
    "$@" && return 0
    rc=$?
    if [ "${rc}" -eq "${BAZELISK_FATAL_RC}" ]; then
      return "${rc}"
    fi
    if [ "${attempt}" -ge "${BAZELISK_FETCH_ATTEMPTS}" ]; then
      echo "giving up after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "attempt ${attempt} of ${BAZELISK_FETCH_ATTEMPTS} failed, retrying in ${delay}s: $*" >&2
    sleep "${delay}"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

function fetch_release_json {
  # Writes the response body to stdout for the caller to capture. An empty body
  # with a zero exit status is a failure too, and must not end the retry loop.
  # Headers go to a file so the rate-limit state can be read off them: with
  # `--server-response` wget prints them to stderr, and `-q` would suppress
  # them along with everything else.
  local body headers rc remaining reset
  headers=$(mktemp)
  body=$(wget -nv -O- --server-response \
    --header="Accept: application/vnd.github+json" \
    ${GITHUB_TOKEN:+--header="Authorization: Bearer $GITHUB_TOKEN"} \
    --header="X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/bazelbuild/bazelisk/releases/tags/${BAZELISK_VERSION}" \
    2>"${headers}")
  rc=$?
  remaining=$(sed -n 's/^[[:space:]]*[Xx]-[Rr]ate[Ll]imit-[Rr]emaining:[[:space:]]*\([0-9]*\).*/\1/p' \
    "${headers}" | tail -1)
  reset=$(sed -n 's/^[[:space:]]*[Xx]-[Rr]ate[Ll]imit-[Rr]eset:[[:space:]]*\([0-9]*\).*/\1/p' \
    "${headers}" | tail -1)
  if [ "${remaining}" = "0" ]; then
    # A primary rate limit refills only at the reset timestamp, up to an hour
    # out, and the quota belongs to the caller -- the runner IP, for the
    # unauthenticated Azure jobs. Retrying inside the job cannot clear it.
    echo ":error: GitHub API rate limit exhausted for this runner$( \
      [ -n "${reset}" ] && printf ' (resets at %s)' "$(date -u -d "@${reset}" 2>/dev/null)")." >&2
    echo ":error: Set GITHUB_TOKEN for this step to use the authenticated quota." >&2
    cat "${headers}" >&2
    rm -f "${headers}"
    return "${BAZELISK_FATAL_RC}"
  fi
  if [ "${rc}" -ne 0 ] || [ -z "${body}" ]; then
    cat "${headers}" >&2
    rm -f "${headers}"
    return 1
  fi
  rm -f "${headers}"
  printf '%s' "${body}"
}

# the pinned digest for this asset, empty when the pins do not apply
SHA256=""
if [ "${BAZELISK_VERSION}" = "${BAZELISK_PINNED_VERSION}" ]; then
  pin_var="BAZELISK_SHA256_$(printf '%s' "${BAZELISK_ASSET}" | tr '.-' '__')"
  SHA256=$(eval "printf '%s' \"\${${pin_var}:-}\"")
fi

if [ -z "${SHA256}" ]; then
  echo "No pinned digest for ${BAZELISK_ASSET} at ${BAZELISK_VERSION}; reading the release metadata."
  # collect information about the bazelisk release
  if ! BAZELISK_JSON=$(retry fetch_release_json) || [ -z "${BAZELISK_JSON}" ]; then
    echo ":error: Failed to fetch Bazelisk release information from GitHub API." >&2
    exit 1
  fi

  # extract SHA256 from json
  found=""
  while IFS= read -r line; do
    if [[ $line == *"\"name\": \"${BAZELISK_ASSET}\""* ]]; then
      found=1
    elif [[ $found && $line == *'"digest":'* ]]; then
      SHA256=$(echo "$line" | sed -n 's/.*"sha256:\([^"]*\)".*/\1/p')
      break
    fi
  done < <(printf '%s\n' "$BAZELISK_JSON")
  if [ -z "${SHA256}" ]; then
    echo ":error: Release metadata for ${BAZELISK_VERSION} carries no digest for ${BAZELISK_ASSET}." >&2
    exit 1
  fi
fi
SHA256+="  ${BAZELISK_ASSET}"

# Download Bazelisk
if ! retry wget -O "${BAZELISK_ASSET}" \
  "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/${BAZELISK_ASSET}"; then
  echo ":error: Failed to download ${BAZELISK_ASSET} from the Bazelisk release." >&2
  exit 1
fi
echo $SHA256
# The script is sourced, nothing sets `-e`, and the install steps below do not
# look at this result: a mismatching download used to be installed anyway, with
# the job going green. Stop here instead, and take the file away with us.
if ! printf '%s\n' "${SHA256}" | sha256sum --check; then
  echo ":error: SHA256 mismatch for ${BAZELISK_ASSET}; refusing to install it." >&2
  rm -f "${BAZELISK_ASSET}"
  exit 1
fi
# "Install" bazelisk
chmod +x ${BAZELISK_ASSET}
mkdir -p bazel/bin
mv ${BAZELISK_ASSET} bazel/bin/bazel
export BAZEL_VERSION=$(./bazel/bin/bazel --version | awk '{print $2}')
export PATH=$PATH:$(pwd)/bazel/bin

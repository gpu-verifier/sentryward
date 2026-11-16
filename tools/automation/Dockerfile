# Build and execute the public verifier with the same CPU-only runtime in every shard.
FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac AS base

# Bootstrap HTTPS trust from signed archives; final packages come from the snapshot.
FROM base AS certificates
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates

FROM base
COPY --from=certificates /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
ENV DEBIAN_FRONTEND=noninteractive
# Signed Ubuntu snapshot fixes transitive system dependencies until an upgrade PR.
RUN printf 'Types: deb\nURIs: https://snapshot.ubuntu.com/ubuntu/20261001T000000Z/\nSuites: noble noble-updates noble-security\nComponents: main universe\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n' > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends python3 python3-venv g++ cmake make libboost-dev libssl-dev libz3-dev git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/runtime
ENV PATH="/opt/runtime/bin:$PATH" PYTHONPATH=/workspace
# Existing project dependencies, pinned for automation. No CUDA runtime is needed.
COPY tools/automation/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.lock
WORKDIR /workspace
COPY . /workspace
RUN cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release -DSENTRYWARD_BUILD_INTERNAL_TOOLS=OFF -DSENTRYWARD_WARNINGS_AS_ERRORS=OFF && cmake --build build/cmake --parallel 2
CMD ["python", "-m", "tools.automation", "--help"]

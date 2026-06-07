FROM ghcr.io/prefix-dev/pixi:0.68.1

LABEL org.opencontainers.image.source="https://github.com/annefou/white-shark-geolocation-light"
LABEL org.opencontainers.image.description="Replication study container for white-shark-geolocation-light"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install the pinned environment first (separate from source copy so the lock
# layer is cached across source-only edits).
COPY pixi.toml pixi.lock /app/
RUN pixi install --locked

COPY . /app

# The GLORYS temperature factor (notebook 01) needs a Copernicus Marine
# credential. Mount it at runtime, e.g.:
#   docker run -v ~/.copernicusmarine:/root/.copernicusmarine \
#       white-shark-geolocation-light
# See README.md / data/README.md for per-dataset credential setup.

CMD ["pixi", "run", "snakemake", "--cores", "1"]

####################################################################################################
## Build pgbakcup and pgbakcup-server
####################################################################################################
FROM golang:latest AS go
FROM debian:13 AS builder_go

ENV TZ="UTC"
# ENV LC_ALL="en_US.UTF-8"
# ENV LANG="en_US.UTF-8"
# ENV LANGUAGE="en_US:en"

# Install git + SSL ca certificates.
# Git is required for fetching the dependencies.
# Ca-certificates is required to call HTTPS endpoints.
RUN apt update && apt upgrade -y && \
    apt install -y ca-certificates git make tzdata binutils mailcap
RUN update-ca-certificates

# setup go
COPY --from=go /usr/local/go /usr/local/go
ENV GOROOT="/usr/local/go"
ENV PATH="$PATH:$GOROOT/bin"
ENV GOPROXY=direct
ENV GOTOOLCHAIN="local"
RUN go telemetry off


WORKDIR /pgbakcup
COPY . ./

RUN make build

####################################################################################################
## This container is used to get the correct files to scratch
####################################################################################################
FROM --platform=$BUILDPLATFORM debian:13-slim AS builder_files

# appuser
ENV USER=pgbakcup
ENV UID=10001

# mailcap is used for content type (MIME type) detection
# tzdata is used for timezones info
RUN apt update && apt upgrade -y && \
    apt install -y mailcap ca-certificates adduser wget tzdata
RUN update-ca-certificates

ENV TZ="UTC"
RUN echo "${TZ}" > /etc/timezone

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    "${USER}"

####################################################################################################
## Final image
####################################################################################################

# See https://chemidy.medium.com/create-the-smallest-and-secured-golang-docker-image-based-on-scratch-4752223b7324
# for more information about how to create Go containers FROM scratch
FROM alpine:latest

ENV TZ="UTC"
ENV LC_ALL="en_US.UTF-8"
ENV LANG="en_US.UTF-8"
ENV LANGUAGE="en_US:en"

# /etc/nsswitch.conf may be used by some DNS resolvers
# /etc/mime.types may be used to detect the MIME type of files
COPY --from=builder_files --chmod=444 \
    /etc/passwd \
    /etc/group \
    /etc/nsswitch.conf \
    /etc/mime.types \
    /etc/timezone \
    /etc/

COPY --from=builder_files  --chmod=444 /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
RUN chmod 0755 /etc/ssl && chmod 0755 /etc/ssl/certs
# COPY --from=builder_files --chmod=755 /etc/ssl /etc/ssl
COPY --from=builder_files --chmod=444 /usr/share/zoneinfo /usr/share/zoneinfo

# Copy our builds
COPY --from=builder_go /pgbakcup/dist/postgresql-backup /usr/local/bin/postgresql-backup

# Use an unprivileged user.
USER pgbakcup:pgbakcup

# the scratch image doesn't have a /tmp folder so we need to create it
WORKDIR /tmp

# Final working directory
WORKDIR /app

ENTRYPOINT ["/usr/local/bin/postgresql-backup"]

#!/bin/sh

resolve_docker_bridge_gateway() {
    _oamb_gateway=$(docker network inspect bridge \
        --format '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null) || return 1
    python3 - "$_oamb_gateway" <<'PY' >/dev/null || return 1
import ipaddress
import sys

address = ipaddress.ip_address(sys.argv[1])
if (
    address.version != 4
    or address.is_unspecified
    or address.is_loopback
    or address.is_multicast
):
    raise SystemExit(1)
PY
    printf '%s\n' "$_oamb_gateway"
}

resolve_host_embedding_base() {
    _oamb_url=$1
    case "$_oamb_url" in
        http://host.docker.internal:*) ;;
        *) printf '%s\n' "$_oamb_url"; return ;;
    esac
    case "$(uname -s)" in
        Darwin) _oamb_host=127.0.0.1 ;;
        Linux)
            _oamb_host=${2:-}
            if [ -z "$_oamb_host" ]; then
                _oamb_host=$(resolve_docker_bridge_gateway) || return 1
            fi
            ;;
        *) return 1 ;;
    esac
    printf 'http://%s:%s\n' "$_oamb_host" "${_oamb_url#http://host.docker.internal:}"
}

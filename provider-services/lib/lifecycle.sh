#!/bin/sh

# The caller defines RUNTIME_DIR, LIFECYCLE_LOCK, LIFECYCLE_LOCK_HELD, and die.
active_provider_attempt_exists_in() {
    provider_operation_directory=$1
    if [ -e "$provider_operation_directory/active-provider-attempt" ] || \
        [ -L "$provider_operation_directory/active-provider-attempt" ]; then
        return 0
    fi
    provider_attempts_directory="$provider_operation_directory/active-provider-attempts"
    [ ! -L "$provider_attempts_directory" ] || \
        die "active provider attempts path is unsafe"
    [ ! -e "$provider_attempts_directory" ] && return 1
    [ -d "$provider_attempts_directory" ] || \
        die "active provider attempts path is unsafe"
    [ -r "$provider_attempts_directory" ] && [ -x "$provider_attempts_directory" ] || \
        die "active provider attempts path is unsafe"
    for pointer in \
        "$provider_attempts_directory"/* \
        "$provider_attempts_directory"/.[!.]* \
        "$provider_attempts_directory"/..?*
    do
        if [ -e "$pointer" ] || [ -L "$pointer" ]; then
            return 0
        fi
    done
    return 1
}

reject_active_provider_domains() {
    lifecycle_domains_directory="$RUNTIME_DIR/lifecycle-domains"
    [ ! -L "$lifecycle_domains_directory" ] || \
        die "lifecycle domains path is unsafe"
    [ ! -e "$lifecycle_domains_directory" ] && return 0
    [ -d "$lifecycle_domains_directory" ] || \
        die "lifecycle domains path is unsafe"
    [ -r "$lifecycle_domains_directory" ] && [ -x "$lifecycle_domains_directory" ] || \
        die "lifecycle domains path is unsafe"
    for provider_domain in \
        "$lifecycle_domains_directory"/* \
        "$lifecycle_domains_directory"/.[!.]* \
        "$lifecycle_domains_directory"/..?*
    do
        if [ ! -e "$provider_domain" ] && [ ! -L "$provider_domain" ]; then
            continue
        fi
        [ ! -L "$provider_domain" ] && [ -d "$provider_domain" ] || \
            die "provider lifecycle domain path is unsafe"
        [ -r "$provider_domain" ] && [ -x "$provider_domain" ] || \
            die "provider lifecycle domain path is unsafe"
        if [ -e "$provider_domain/provider-lifecycle.lock" ] || \
            [ -L "$provider_domain/provider-lifecycle.lock" ]; then
            die "provider lifecycle domain admission exists; refusing provider lifecycle mutation"
        fi
        if [ -e "$provider_domain/active-operation" ] || \
            [ -L "$provider_domain/active-operation" ]; then
            die "active OAMB provider operation exists; refusing provider lifecycle mutation"
        fi
        if active_provider_attempt_exists_in "$provider_domain"; then
            die "active provider attempt exists; refusing provider lifecycle mutation"
        fi
    done
}

release_lifecycle_lock() {
    if [ "$LIFECYCLE_LOCK_HELD" = true ]; then
        rmdir "$LIFECYCLE_LOCK" 2>/dev/null || true
        LIFECYCLE_LOCK_HELD=false
    fi
}

acquire_lifecycle_lock() {
    [ ! -L "$RUNTIME_DIR" ] || die "provider runtime path is unsafe"
    mkdir -p "$RUNTIME_DIR"
    mkdir "$LIFECYCLE_LOCK" 2>/dev/null || die "another provider lifecycle operation is active"
    LIFECYCLE_LOCK_HELD=true
    trap 'release_lifecycle_lock' 0
    trap 'exit 130' 1 2 15
    if [ -e "$RUNTIME_DIR/active-run-lease" ] || \
        [ -L "$RUNTIME_DIR/active-run-lease" ]; then
        die "legacy active OAMB run lease exists; refusing provider lifecycle mutation"
    fi
    if [ -e "$RUNTIME_DIR/active-operation" ] || \
        [ -L "$RUNTIME_DIR/active-operation" ]; then
        die "active OAMB provider operation exists; refusing provider lifecycle mutation"
    fi
    if active_provider_attempt_exists_in "$RUNTIME_DIR"; then
        die "active provider attempt exists; refusing provider lifecycle mutation"
    fi
    reject_active_provider_domains
}

acquire_lifecycle_stop_lock() {
    [ ! -L "$RUNTIME_DIR" ] || die "provider runtime path is unsafe"
    mkdir -p "$RUNTIME_DIR"
    mkdir "$LIFECYCLE_LOCK" 2>/dev/null || die "another provider lifecycle operation is active"
    LIFECYCLE_LOCK_HELD=true
    trap 'release_lifecycle_lock' 0
    trap 'exit 130' 1 2 15
}

clear_stopped_provider_lifecycle() {
    lifecycle_domains_directory="$RUNTIME_DIR/lifecycle-domains"
    [ ! -L "$lifecycle_domains_directory" ] || die "lifecycle domains path is unsafe"
    if [ -e "$lifecycle_domains_directory" ]; then
        [ -d "$lifecycle_domains_directory" ] || die "lifecycle domains path is unsafe"
    fi
    for provider_operation_directory in \
        "$RUNTIME_DIR" \
        "$RUNTIME_DIR/lifecycle-domains/hindsight" \
        "$RUNTIME_DIR/lifecycle-domains/mem0" \
        "$RUNTIME_DIR/lifecycle-domains/openviking"
    do
        [ ! -L "$provider_operation_directory" ] || \
            die "provider lifecycle domain path is unsafe"
        [ -d "$provider_operation_directory" ] || continue
        rm -f \
            "$provider_operation_directory/active-run-lease" \
            "$provider_operation_directory/active-operation" \
            "$provider_operation_directory/active-provider-attempt"
        if [ "$provider_operation_directory" != "$RUNTIME_DIR" ]; then
            rmdir "$provider_operation_directory/provider-lifecycle.lock" 2>/dev/null || true
        fi
        attempts_directory="$provider_operation_directory/active-provider-attempts"
        [ ! -L "$attempts_directory" ] || die "active provider attempts path is unsafe"
        [ -d "$attempts_directory" ] || continue
        rm -f "$attempts_directory"/*.json
    done
}

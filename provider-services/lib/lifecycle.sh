#!/bin/sh

# The caller defines RUNTIME_DIR, LIFECYCLE_LOCK, LIFECYCLE_LOCK_HELD, and die.
release_lifecycle_lock() {
    if [ "$LIFECYCLE_LOCK_HELD" = true ]; then
        rmdir "$LIFECYCLE_LOCK" 2>/dev/null || true
        LIFECYCLE_LOCK_HELD=false
    fi
}

acquire_lifecycle_lock() {
    mkdir -p "$RUNTIME_DIR"
    mkdir "$LIFECYCLE_LOCK" 2>/dev/null || die "another provider lifecycle operation is active"
    LIFECYCLE_LOCK_HELD=true
    trap 'release_lifecycle_lock' 0
    trap 'exit 130' 1 2 15
    [ ! -e "$RUNTIME_DIR/active-run-lease" ] || \
        die "active OAMB run lease exists; refusing provider lifecycle mutation"
}

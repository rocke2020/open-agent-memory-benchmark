#!/bin/sh

# Keep the operator file inside the simple grammar interpreted identically by
# this bundle and Docker Compose. Values remain inert text.
validate_env_file() {
    awk '
        /^[[:space:]]*$/ || /^[[:space:]]*#/ { next }
        {
            separator = index($0, "=")
            if (separator == 0) {
                print "invalid dotenv line without =" > "/dev/stderr"
                failed = 1
                next
            }
            key = substr($0, 1, separator - 1)
            value = substr($0, separator + 1)
            if (key !~ /^[A-Za-z_][A-Za-z0-9_]*$/) {
                print "invalid dotenv key: " key > "/dev/stderr"
                failed = 1
            }
            if (++seen[key] > 1) {
                print "duplicate dotenv key: " key > "/dev/stderr"
                failed = 1
            }
            trimmed = value
            sub(/^[[:space:]]+/, "", trimmed)
            sub(/[[:space:]]+$/, "", trimmed)
            if (value != trimmed || value ~ /["$`\\#]/ || index(value, sprintf("%c", 39))) {
                print "unsupported dotenv value syntax for: " key > "/dev/stderr"
                failed = 1
            }
        }
        END { exit failed ? 1 : 0 }
    ' "$1"
}

# Read one unique dotenv assignment as inert text. This helper never evaluates values.
read_env_value() {
    env_file=$1
    wanted_name=$2
    awk -F= -v wanted="$wanted_name" \
        '$1 == wanted {sub(/^[^=]*=/, ""); value=$0; count+=1}
         END {if (count != 1) exit 1; print value}' \
        "$env_file"
}

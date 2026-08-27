#!/bin/sh
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
SELECT 'CREATE DATABASE mem0_app'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mem0_app')\gexec
SQL

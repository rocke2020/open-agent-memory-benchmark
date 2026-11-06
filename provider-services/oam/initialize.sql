-- Run once with the operator connection after `oam migrate` on a fresh database.
-- Passwords are supplied through psql variables, never stored in this file.
CREATE ROLE oamb_api LOGIN PASSWORD :'api_password' NOSUPERUSER NOBYPASSRLS;
CREATE ROLE oamb_queue LOGIN PASSWORD :'queue_password' NOSUPERUSER NOBYPASSRLS;
CREATE ROLE oamb_content LOGIN PASSWORD :'content_password' NOSUPERUSER NOBYPASSRLS;
GRANT oam_api TO oamb_api;
GRANT oam_queue TO oamb_queue;
GRANT oam_content TO oamb_content;

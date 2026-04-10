#!/bin/bash
# ACS V3.1 Launcher
cd "$(dirname "$0")"
source venv/bin/activate
python acs.py "$@"

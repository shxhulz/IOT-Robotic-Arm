#!/usr/bin/env bash
set -euo pipefail

if [ ! -f /models/sam2/sam2_hiera_large.pt ]; then
  python /opt/material/app/download_models.py --best-effort
fi

exec python /opt/material/app/material_pipeline.py

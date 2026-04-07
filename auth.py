"""
auth.py
-------
GEE authentication for the Streamlit dashboard.

Supports two modes:
  1. Service account JSON (recommended for deployment)
  2. Application default credentials / ee.Initialize() with project

Set GEE_SERVICE_ACCOUNT_JSON in st.secrets or as an env variable.
"""

import ee
import json
import os
import streamlit as st


def initialize_gee(project: str = "projecttestrun") -> bool:
    """
    Initialize Google Earth Engine. Tries service account credentials first,
    falls back to application default credentials.

    Returns True if successful, False otherwise.
    """
    try:
        # Option 1: service account JSON from env var
        sa_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            sa_info = json.loads(sa_json)
            credentials = ee.ServiceAccountCredentials(
                email=sa_info["client_email"],
                key_data=sa_json,
            )
            ee.Initialize(credentials=credentials, project=project)
            return True

        # Option 2: service account JSON from Streamlit secrets (deployment only)
        try:
            if "GEE_SERVICE_ACCOUNT_JSON" in st.secrets:
                sa_info = json.loads(st.secrets["GEE_SERVICE_ACCOUNT_JSON"])
                credentials = ee.ServiceAccountCredentials(
                    email=sa_info["client_email"],
                    key_data=json.dumps(sa_info),
                )
                ee.Initialize(credentials=credentials, project=project)
                return True
        except Exception:
            pass  # no secrets file locally, that's fine

        # Option 3: application default credentials (local dev)
        ee.Initialize(project=project)
        return True

    except Exception as e:
        st.error(f"GEE authentication failed: {e}")
        return False

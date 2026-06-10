import os
import importlib.metadata

# DO NOT EDIT
__pkg_version__ = importlib.metadata.version("weco")
__api_version__ = "v1"

__base_url__ = os.environ.get("WECO_BASE_URL", f"https://api.weco.ai/{__api_version__}")
__dashboard_url__ = os.environ.get("WECO_DASHBOARD_URL", "https://dashboard.weco.ai")

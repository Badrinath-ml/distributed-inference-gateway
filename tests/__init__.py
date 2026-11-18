"""Test package.

The automated suite must never contact a real inference provider, even when the
developer's ``.env`` contains live cloud credentials. Environment variables take
precedence over ``.env`` in pydantic-settings, so force the optional cloud worker
off for the test process. Tests that construct cloud workers directly use mocked
HTTP transports.
"""

import os

os.environ["OPENROUTER_ENABLED"] = "false"

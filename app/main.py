"""Deploy entrypoint: `modal deploy -m app.main`.

Importing the modules below registers the GPU service and the HTTP gateway
on the shared Modal app.
"""

import app.api  # noqa: F401, E402
import app.cutout.service  # noqa: F401, E402
from app.common import app  # noqa: F401

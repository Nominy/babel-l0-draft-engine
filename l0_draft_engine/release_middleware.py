from starlette.responses import JSONResponse
from .inference_release import RELEASE_ID, RELEASE_HEADER, RELEASE_HEADERS, upgrade_detail


class InferenceReleaseMiddleware:
    """Reject old clients before reading uploads, without changing streaming semantics."""
    def __init__(self, app, enforced: bool):
        self.app, self.enforced = app, enforced

    async def __call__(self, scope, receive, send):
        if (self.enforced and scope["type"] == "http" and scope["method"] != "OPTIONS"
                and scope["path"].startswith("/v1/")):
            headers = dict(scope.get("headers", []))
            if headers.get(RELEASE_HEADER.lower().encode()) != RELEASE_ID.encode():
                await JSONResponse(upgrade_detail(), status_code=426, headers=RELEASE_HEADERS)(scope, receive, send)
                return
        await self.app(scope, receive, send)

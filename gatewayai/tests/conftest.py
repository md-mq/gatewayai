import django
from django.conf import settings

# Minimal Django configuration for tests that use HttpResponse/StreamingHttpResponse
if not settings.configured:
    settings.configure(
        DEFAULT_CHARSET="utf-8",
        SECRET_KEY="test-secret-key",
    )
    django.setup()

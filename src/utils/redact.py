"""Sensitive information redaction utility for user-facing outputs.

Used by employee_engine and other modules to sanitize text before sending
to Feishu group chats, cards, or text notifications.
"""

from __future__ import annotations

import re

# Pattern: key=value assignments where key contains sensitive keywords
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIAL|API_KEY)[A-Z0-9_]*)"
    r"\s*[=:]\s*[^\s\n,;]+"
)

# Pattern: Bearer token in authorization headers
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")

# Pattern: Authorization header value (captures full value including Bearer token)
_AUTHORIZATION_RE = re.compile(
    r"(?i)\b(Authorization)\s*[=:]\s*\S+(?:\s+\S+)?"
)

# Pattern: cookie key=value pairs
_COOKIE_RE = re.compile(
    r"(?i)\b(cookie|set-cookie)\s*[=:]\s*[^\n;]{4,}"
)

# Pattern: private key blocks (PEM format)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# Pattern: AWS-style access keys (AKIA...)
_AWS_KEY_RE = re.compile(r"\b(AKIA[0-9A-Z]{16})\b")

# Pattern: OpenAI-style API keys (sk-...)
_OPENAI_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")

# Pattern: GitLab personal/project/group access tokens (glpat-...)
_GITLAB_TOKEN_RE = re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b", re.IGNORECASE)

# Pattern: Slack bot/app/user tokens (xoxb-/xoxa-/xoxp-/xoxr-/xoxs-...)
_SLACK_TOKEN_RE = re.compile(
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    re.IGNORECASE,
)

# Pattern: GitHub fine/classic token families (ghp_/gho_/ghu_/ghs_/ghr_...)
_GITHUB_TOKEN_RE = re.compile(
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
    re.IGNORECASE,
)


def redact_sensitive(text: str) -> str:
    """Redact sensitive information from text.

    Replaces tokens, passwords, secrets, API keys, private keys, cookies,
    and authorization headers with <redacted> placeholders.

    Args:
        text: Input text that may contain sensitive information.

    Returns:
        Text with sensitive values replaced by redaction markers.
    """
    if not text:
        return text

    # Order matters: longer patterns first to avoid partial matches
    result = _PRIVATE_KEY_RE.sub("<redacted:private_key>", text)
    result = _AUTHORIZATION_RE.sub(r"\1=<redacted>", result)
    result = _BEARER_RE.sub("Bearer <redacted>", result)
    result = _COOKIE_RE.sub(r"\1=<redacted>", result)
    result = _AWS_KEY_RE.sub("<redacted:aws_key>", result)
    result = _OPENAI_API_KEY_RE.sub("<redacted:api_key>", result)
    result = _GITLAB_TOKEN_RE.sub("<redacted:gitlab_token>", result)
    result = _SLACK_TOKEN_RE.sub("<redacted:slack_token>", result)
    result = _GITHUB_TOKEN_RE.sub("<redacted:github_token>", result)
    result = _SECRET_ASSIGNMENT_RE.sub(r"\1=<redacted>", result)

    return result

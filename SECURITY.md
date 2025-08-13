# Security Policy

## Supported Versions
This is a research/demo project; only the `main` branch is maintained.

## Reporting Vulnerabilities
Please open a private issue or email the maintainer if you discover:
- Remote code execution vectors
- Injection via model prompts leading to file/system access
- Sensitive data leakage through logs

Avoid publicly disclosing until a fix is discussed.

## Hardening Suggestions
- Run the llama.cpp server & this app on a trusted network only.
- Do not expose without authentication.
- Sanitize / filter user provided text before logging.
- Keep dependencies updated (`pip list --outdated`).

## Model Security
Models may output unsafe content. Add content filtering layer if deploying publicly.

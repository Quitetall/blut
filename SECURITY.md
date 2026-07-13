# Security Policy

## Supported versions

BLUT is in public preview. Security fixes target the current
`0.2.0-alpha.x` line and `main`. Older previews and internal milestone tags are
unsupported.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/Quitetall/blut/security/advisories/new)
and include:

- affected version or commit;
- reproduction steps or proof of concept;
- expected impact;
- any proposed mitigation.

Never include real credentials, private datasets, patient data, or production
artifacts in a report. Maintainers will acknowledge reports within seven days,
coordinate validation and remediation privately, then publish an advisory when
a fix is available.

## Scope

Security-sensitive surfaces include containment, process launching, artifact
paths, secret handling, P2P transport, cloud object storage, and cookbook CLI
integration. Deprecated worker and experimental operator prototypes are not
part of the supported public preview.

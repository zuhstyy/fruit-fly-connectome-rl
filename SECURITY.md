# Security reporting

I intend security fixes for the current 4.1.x project line. I maintain this as
an experimental local application and cannot guarantee a response time.

For a vulnerability involving token exposure, arbitrary file access, or control
of unintended processes, use the repository's private vulnerability reporting
feature if enabled. Otherwise, open a minimal issue asking for a private contact
without publishing exploit details or credentials.

Include the project version, affected component, impact, and a reproducible
description using disposable test files. Remove authentication tokens, personal
paths, private saves, and game binaries from attachments.

The bridge binds to localhost and authenticates each worker. Do not expose its
port through a public tunnel. Routine startup and training problems belong in
the bug-report template.

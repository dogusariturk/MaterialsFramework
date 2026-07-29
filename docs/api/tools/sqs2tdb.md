# PhaseForge

!!! info "Optional dependency"

    `Sqs2tdb` requires the `calphad` extra, plus the `sqs2tdb` binary (part of [ATAT](https://axelvandewalle.github.io/www-avdw/atat/)) on `PATH`, with its SQS database configured via `~/.atat.rc`.

    === "uv"

        ```bash
        uv add "materialsframework[calphad]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[calphad]"
        ```

::: materialsframework.tools.sqs2tdb.Sqs2tdb

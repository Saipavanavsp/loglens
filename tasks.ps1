# PowerShell equivalent of the Makefile for Windows. Usage: ./tasks.ps1 test
param([Parameter(Mandatory = $true)][string]$Target)

$ErrorActionPreference = "Stop"

switch ($Target) {
    "install" {
        python -m pip install -e ".[data,gateway,eval,dashboard,dev]"
        pre-commit install
    }
    "lint" { ruff check src tests; if ($?) { black --check src tests } }
    "format" { ruff check --fix src tests; black src tests }
    "test" { pytest }
    "check" { ruff check src tests; if ($?) { black --check src tests }; if ($?) { pytest } }
    "data-download" { python -m loglens.data.download }
    "data-parse" { python -m loglens.data.parse }
    "up" { docker compose -f infra/docker-compose.yml up -d --build }
    "down" { docker compose -f infra/docker-compose.yml down }
    default { Write-Error "Unknown target: $Target" }
}

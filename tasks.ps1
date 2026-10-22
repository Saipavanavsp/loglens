# PowerShell equivalent of the Makefile for Windows. Usage: ./tasks.ps1 test
#   ./tasks.ps1 label -N 50 -Args "--no-dry-run"
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [int]$N = 50,
    [string]$Args = ""
)

$ErrorActionPreference = "Stop"
$extra = if ($Args) { $Args.Split(" ") } else { @() }

switch ($Target) {
    "install" {
        python -m pip install -e ".[data,gateway,eval,dashboard,dev]"
        pre-commit install
    }
    "lint" { ruff check src tests dashboard training evals; if ($?) { black --check src tests dashboard training } }
    "format" { ruff check --fix src tests dashboard training; black src tests dashboard training }
    "test" { pytest }
    "check" {
        ruff check src tests dashboard training evals
        if ($?) { black --check src tests dashboard training }
        if ($?) { pytest }
    }
    "data" {
        python -m loglens.data.download
        python -m loglens.data.parse
        python -m loglens.data.sample --n 3000
    }
    "data-download" { python -m loglens.data.download }
    "data-parse" { python -m loglens.data.parse }
    "data-sample" { python -m loglens.data.sample --n 3000 }
    "label" { python -m loglens.data.label --limit $N @extra }
    "golden" { python -m loglens.data.review select --n 200 }
    "review" { streamlit run dashboard/review_app.py }
    "split" { python -m loglens.data.split }
    "stats" { python -m loglens.data.stats }
    "eval" { python -m loglens.evals.run @extra }
    "up" { docker compose -f infra/docker-compose.yml up -d --build }
    "down" { docker compose -f infra/docker-compose.yml down }
    default { Write-Error "Unknown target: $Target" }
}

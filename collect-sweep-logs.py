#!/usr/bin/env python3
"""Collect sweep results and generate CSV or Google Sheet.

Reads sweep log files (one per target, e.g. agg.log, disagg.log)
and produces a comparison spreadsheet with Pareto frontier chart.

Usage:
    python3 collect-sweep-logs.py [-d RESULTS_DIR]
    python3 collect-sweep-logs.py [-d RESULTS_DIR] --sheets "Spreadsheet Title"
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchResult:
    target: str
    concurrency: int
    output_throughput: float  # Output token throughput (tok/s)
    total_throughput: float   # Total token throughput (tok/s)
    gpu_count: int

    @property
    def tpsu(self) -> float:
        """Throughput per user (output tok/s / concurrency)."""
        return self.output_throughput / self.concurrency

    @property
    def tpsg(self) -> float:
        """Throughput per GPU (output tok/s / gpu_count)."""
        return self.output_throughput / self.gpu_count


def parse_sweep_log(log_text: str) -> tuple[dict[str, str], list[BenchResult]]:
    """Parse a sweep log file.

    Returns (config_dict, results_list).
    Config dict has keys: target, model, isl, osl, gpu_count.
    """
    config: dict[str, str] = {}
    results: list[BenchResult] = []

    # Parse CONFIG line
    m = re.search(r"CONFIG:\s*(.+)", log_text)
    if m:
        for field in re.findall(r"(\w+)=(\S+)", m.group(1)):
            config[field[0]] = field[1]

    # Parse TARGET_DURATION and MIN_PROMPTS
    m = re.search(r"TARGET_DURATION:\s*(\S+)", log_text)
    if m:
        config["target_duration"] = m.group(1)
    m = re.search(r"MIN_PROMPTS:\s*(\S+)", log_text)
    if m:
        config["min_prompts"] = m.group(1)

    # Parse CONCURRENCY_LEVELS
    m = re.search(r"CONCURRENCY_LEVELS:\s*(.+)", log_text)
    if m:
        config["concurrency_levels"] = m.group(1).strip()

    target = config.get("target", "unknown")
    gpu_count = int(config.get("gpu_count", "0"))

    # Parse BENCH_RUN blocks
    output_tp_re = re.compile(
        r"Output token throughput \(tok/s\):\s+([\d.]+)", re.IGNORECASE
    )
    total_tp_re = re.compile(
        r"Total Token throughput \(tok/s\):\s+([\d.]+)", re.IGNORECASE
    )
    bench_start_re = re.compile(r"BENCH_RUN: concurrency=(\d+)")
    bench_end_re = re.compile(r"BENCH_RUN_END: concurrency=(\d+)")

    current_concurrency = None
    current_output_tp = None
    current_total_tp = None

    for line in log_text.split("\n"):
        m = bench_start_re.search(line)
        if m:
            current_concurrency = int(m.group(1))
            current_output_tp = None
            current_total_tp = None
            continue

        if current_concurrency is None:
            continue

        m = output_tp_re.search(line)
        if m:
            current_output_tp = float(m.group(1))
            continue

        m = total_tp_re.search(line)
        if m:
            current_total_tp = float(m.group(1))
            continue

        m = bench_end_re.search(line)
        if m:
            if current_output_tp is not None and current_total_tp is not None:
                results.append(BenchResult(
                    target=target,
                    concurrency=current_concurrency,
                    output_throughput=current_output_tp,
                    total_throughput=current_total_tp,
                    gpu_count=gpu_count,
                ))
            else:
                print(
                    f"  WARNING: incomplete data for {target} concurrency={current_concurrency}",
                    file=sys.stderr,
                )
            current_concurrency = None

    return config, results


def build_scaling_table(results: list[BenchResult], target: str) -> list[list]:
    """Build a scaling table for a single target."""
    results = sorted(results, key=lambda r: r.concurrency)
    if not results:
        return []

    gpu_count = results[0].gpu_count
    concurrencies = [r.concurrency for r in results]
    throughputs = [int(round(r.output_throughput)) for r in results]
    tpsgs = [int(round(r.tpsg)) for r in results]
    tpsus = [int(round(r.tpsu)) for r in results]

    rows = [
        ["", target, f"GPUs: {gpu_count}"],
        ["", "Concurrency"] + concurrencies,
        ["", "Output tok/s"] + throughputs,
        ["", "TPSG"] + tpsgs,
        ["", "TPSU"] + tpsus,
        [],
    ]
    return rows


def build_pareto_data(results_by_target: dict[str, list[BenchResult]]) -> list[list]:
    """Build Pareto chart data in sparse layout for Google Sheets scatter chart.

    Column A = TPSU (shared X-axis), one Y column per target with TPSG values.
    """
    targets = sorted(results_by_target.keys())
    header = ["TPSU (tok/s/user)"] + [f"{t} TPSG" for t in targets]
    rows = [header]

    for t in targets:
        t_results = sorted(results_by_target[t], key=lambda r: r.concurrency)
        col_idx = targets.index(t)
        for r in t_results:
            row = [int(round(r.tpsu))] + [""] * len(targets)
            row[col_idx + 1] = int(round(r.tpsg))
            rows.append(row)

    return rows


def _fmt_bold(sheet_id: int, r0: int, r1: int,
              c0: int = 0, c1: int = 20) -> dict:
    return {
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": r0, "endRowIndex": r1,
                "startColumnIndex": c0, "endColumnIndex": c1,
            },
            "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
            "fields": "userEnteredFormat.textFormat.bold",
        }
    }


def _fmt_center(sheet_id: int) -> dict:
    return {
        "repeatCell": {
            "range": {"sheetId": sheet_id},
            "cell": {"userEnteredFormat": {"horizontalAlignment": "CENTER"}},
            "fields": "userEnteredFormat.horizontalAlignment",
        }
    }


def _create_scatter_chart(
    sheet_id: int,
    title: str,
    start_row_idx: int,
    end_row_idx: int,
    num_series: int,
    anchor_row: int,
) -> dict:
    series = []
    for i in range(num_series):
        series.append({
            "series": {
                "sourceRange": {
                    "sources": [{
                        "sheetId": sheet_id,
                        "startRowIndex": start_row_idx,
                        "endRowIndex": end_row_idx,
                        "startColumnIndex": i + 1,
                        "endColumnIndex": i + 2,
                    }]
                }
            },
            "targetAxis": "LEFT_AXIS",
        })
    return {
        "addChart": {
            "chart": {
                "spec": {
                    "title": title,
                    "basicChart": {
                        "chartType": "SCATTER",
                        "legendPosition": "BOTTOM_LEGEND",
                        "domains": [{
                            "domain": {
                                "sourceRange": {
                                    "sources": [{
                                        "sheetId": sheet_id,
                                        "startRowIndex": start_row_idx,
                                        "endRowIndex": end_row_idx,
                                        "startColumnIndex": 0,
                                        "endColumnIndex": 1,
                                    }]
                                }
                            }
                        }],
                        "series": series,
                        "axis": [
                            {
                                "position": "BOTTOM_AXIS",
                                "title": "Throughput per User (tok/s)",
                            },
                            {
                                "position": "LEFT_AXIS",
                                "title": "Throughput per GPU (tok/s)",
                            },
                        ],
                        "headerCount": 1,
                    },
                },
                "position": {
                    "overlayPosition": {
                        "anchorCell": {
                            "sheetId": sheet_id,
                            "rowIndex": anchor_row,
                            "columnIndex": 0,
                        },
                        "widthPixels": 800,
                        "heightPixels": 500,
                    }
                },
            }
        }
    }


def upload_to_sheets(
    results_by_target: dict[str, list[BenchResult]],
    configs: dict[str, dict[str, str]],
    spreadsheet_title: str,
) -> str:
    """Upload results to Google Sheets with Pareto chart. Returns spreadsheet URL."""
    try:
        import gspread
    except ImportError:
        print(
            "gspread is not installed. Run: uv pip install gspread google-auth-oauthlib",
            file=sys.stderr,
        )
        sys.exit(1)

    gc = gspread.oauth()
    sh = gc.create(spreadsheet_title)
    print(f"Created spreadsheet: {spreadsheet_title}")

    # --- Results sheet ---
    try:
        results_ws = sh.worksheet("Results")
        results_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        results_ws = sh.add_worksheet(title="Results", rows=200, cols=20)

    all_rows: list[list] = []
    fmt_requests = [_fmt_center(results_ws.id)]

    # Scaling tables for each target
    for target in sorted(results_by_target.keys()):
        table_start = len(all_rows)
        table_rows = build_scaling_table(results_by_target[target], target)
        all_rows.extend(table_rows)
        # Bold the header row of each table
        fmt_requests.append(_fmt_bold(results_ws.id, table_start, table_start + 1))

    # Pareto chart data
    pareto_start_row = len(all_rows) + 1  # 1-indexed for Sheets
    pareto_rows = build_pareto_data(results_by_target)
    all_rows.extend(pareto_rows)

    pareto_start_idx = pareto_start_row - 1  # 0-indexed
    pareto_end_idx = pareto_start_idx + len(pareto_rows)
    num_series = len(results_by_target)
    chart_anchor = pareto_end_idx + 1

    fmt_requests.append(_fmt_bold(results_ws.id, pareto_start_idx, pareto_start_idx + 1))
    fmt_requests.append(_create_scatter_chart(
        results_ws.id,
        "GPU Efficiency vs User Throughput",
        pareto_start_idx, pareto_end_idx,
        num_series,
        chart_anchor,
    ))

    results_ws.update(range_name="A1", values=all_rows)
    if results_ws.row_count < chart_anchor + 30:
        results_ws.resize(rows=chart_anchor + 30)
    sh.batch_update({"requests": fmt_requests})

    total_results = sum(len(v) for v in results_by_target.values())
    print(f"  Results: {len(all_rows)} rows, Pareto chart ({total_results} points, {num_series} series)")

    # --- Config sheet ---
    try:
        cfg_ws = sh.worksheet("Config")
        cfg_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        cfg_ws = sh.add_worksheet(title="Config", rows=50, cols=5)

    # Merge config from all targets
    any_config = next(iter(configs.values()), {})
    cfg_rows = [
        ["Experiment Configuration"],
        [],
        ["Model", any_config.get("model", "")],
        ["ISL", any_config.get("isl", "")],
        ["OSL", any_config.get("osl", "")],
        ["Target Duration", any_config.get("target_duration", "")],
        ["Min Prompts", any_config.get("min_prompts", "")],
        [],
    ]

    for target in sorted(configs.keys()):
        cfg = configs[target]
        gpu_count = cfg.get("gpu_count", "?")
        concurrencies = cfg.get("concurrency_levels", "")
        cfg_rows.append([f"{target} GPUs", gpu_count])
        cfg_rows.append([f"{target} Concurrencies", concurrencies])

    cfg_rows.append([])
    cfg_rows.append(["Timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")])

    cfg_ws.update(range_name="A1", values=cfg_rows)

    cfg_fmt = [
        _fmt_bold(cfg_ws.id, 0, 1),
        _fmt_bold(cfg_ws.id, 2, len(cfg_rows), 0, 1),
    ]
    sh.batch_update({"requests": cfg_fmt})
    print(f"  Config: {len(cfg_rows)} rows")

    # Remove default Sheet1
    try:
        default_sheet = sh.worksheet("Sheet1")
        if len(sh.worksheets()) > 1:
            sh.del_worksheet(default_sheet)
    except gspread.exceptions.WorksheetNotFound:
        pass

    print(f"\nSpreadsheet URL: {sh.url}")
    return sh.url


def write_csv(results_by_target: dict[str, list[BenchResult]], output_dir: str) -> None:
    """Write results as CSV files."""
    for target, results in sorted(results_by_target.items()):
        rows = build_scaling_table(results, target)
        path = f"{output_dir}/{target}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
        print(f"Written: {path}")

    # Write combined pareto data
    pareto_rows = build_pareto_data(results_by_target)
    path = f"{output_dir}/pareto.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in pareto_rows:
            w.writerow(row)
    print(f"Written: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect sweep results and generate CSV or Google Sheet"
    )
    parser.add_argument(
        "--results-dir", "-d", default="results",
        help="Directory containing sweep log files (default: results)",
    )
    parser.add_argument(
        "--sheets", metavar="TITLE",
        help="Upload results to a Google Spreadsheet with Pareto chart",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    log_files = sorted(results_dir.glob("*.log"))
    if not log_files:
        print(f"No .log files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(log_files)} log file(s) in {results_dir}")

    results_by_target: dict[str, list[BenchResult]] = {}
    configs: dict[str, dict[str, str]] = {}

    for log_file in log_files:
        print(f"  Parsing {log_file.name}...")
        log_text = log_file.read_text()
        config, results = parse_sweep_log(log_text)

        if not results:
            print(f"    WARNING: no benchmark results found", file=sys.stderr)
            continue

        target = config.get("target", log_file.stem)
        results_by_target[target] = results
        configs[target] = config
        print(f"    {len(results)} result(s), gpu_count={config.get('gpu_count', '?')}")

    if not results_by_target:
        print("\nNo benchmark results parsed from logs", file=sys.stderr)
        sys.exit(1)

    total = sum(len(v) for v in results_by_target.values())
    print(f"\nTotal: {total} benchmark results across {len(results_by_target)} target(s)")

    # Write CSVs
    write_csv(results_by_target, str(results_dir))

    # Upload to Google Sheets if requested
    if args.sheets:
        upload_to_sheets(results_by_target, configs, args.sheets)


if __name__ == "__main__":
    main()

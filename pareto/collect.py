#!/usr/bin/env python3
"""Collect pareto sweep results from Kubernetes Job logs.

Reads structured sweep output from kubectl logs of pareto Job pods
across one or more namespaces (one config per namespace) and produces
CSV files and optional Google Sheets with Pareto charts.

Usage:
    python3 pareto/collect.py -n agg-tp8 -n agg-tp4x2 -n disagg-1p1d
    python3 pareto/collect.py -n agg-tp8 -n disagg-1p1d --sheets "Sweep Results"
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LatencyStats:
    mean: float = 0.0
    median: float = 0.0
    p99: float = 0.0


@dataclass
class BenchResult:
    target: str
    concurrency: int
    output_throughput: float  # Output token throughput (tok/s)
    total_throughput: float   # Total token throughput (tok/s)
    gpu_count: int
    ttft: LatencyStats = None
    tpot: LatencyStats = None
    itl: LatencyStats = None

    def __post_init__(self):
        if self.ttft is None:
            self.ttft = LatencyStats()
        if self.tpot is None:
            self.tpot = LatencyStats()
        if self.itl is None:
            self.itl = LatencyStats()

    @property
    def tpsu(self) -> float:
        """Throughput per user (output tok/s / concurrency)."""
        return self.output_throughput / self.concurrency

    @property
    def tpsg(self) -> float:
        """Throughput per GPU (output tok/s / gpu_count)."""
        return self.output_throughput / self.gpu_count if self.gpu_count > 0 else 0.0


def parse_sweep_log(log_text: str) -> tuple[dict[str, str], list[BenchResult]]:
    """Parse a sweep log (same format as Justfile.remote sweep output).

    Returns (config_dict, results_list).
    """
    config: dict[str, str] = {}
    results: list[BenchResult] = []

    m = re.search(r"CONFIG:\s*(.+)", log_text)
    if m:
        for field in re.findall(r"(\w+)=(\S+)", m.group(1)):
            config[field[0]] = field[1]

    m = re.search(r"TARGET_DURATION:\s*(\S+)", log_text)
    if m:
        config["target_duration"] = m.group(1)
    m = re.search(r"MIN_PROMPTS:\s*(\S+)", log_text)
    if m:
        config["min_prompts"] = m.group(1)
    m = re.search(r"CONCURRENCY_LEVELS:\s*(.+)", log_text)
    if m:
        config["concurrency_levels"] = m.group(1).strip()

    target = config.get("target", "unknown")
    gpu_count = int(config.get("gpu_count", "0"))

    output_tp_re = re.compile(
        r"^Output token throughput \(tok/s\):\s+([\d.]+)", re.IGNORECASE
    )
    total_tp_re = re.compile(
        r"Total Token throughput \(tok/s\):\s+([\d.]+)", re.IGNORECASE
    )
    bench_start_re = re.compile(r"BENCH_RUN: concurrency=(\d+)")
    bench_end_re = re.compile(r"BENCH_RUN_END: concurrency=(\d+)")

    # Latency metric patterns
    latency_patterns = {
        "ttft_mean": re.compile(r"Mean TTFT \(ms\):\s+([\d.]+)"),
        "ttft_median": re.compile(r"Median TTFT \(ms\):\s+([\d.]+)"),
        "ttft_p99": re.compile(r"P99 TTFT \(ms\):\s+([\d.]+)"),
        "tpot_mean": re.compile(r"Mean TPOT \(ms\):\s+([\d.]+)"),
        "tpot_median": re.compile(r"Median TPOT \(ms\):\s+([\d.]+)"),
        "tpot_p99": re.compile(r"P99 TPOT \(ms\):\s+([\d.]+)"),
        "itl_mean": re.compile(r"Mean ITL \(ms\):\s+([\d.]+)"),
        "itl_median": re.compile(r"Median ITL \(ms\):\s+([\d.]+)"),
        "itl_p99": re.compile(r"P99 ITL \(ms\):\s+([\d.]+)"),
    }

    current_concurrency = None
    current_output_tp = None
    current_total_tp = None
    current_latencies: dict[str, float] = {}

    for line in log_text.split("\n"):
        m = bench_start_re.search(line)
        if m:
            current_concurrency = int(m.group(1))
            current_output_tp = None
            current_total_tp = None
            current_latencies = {}
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

        for key, pat in latency_patterns.items():
            m = pat.search(line)
            if m:
                current_latencies[key] = float(m.group(1))
                break

        m = bench_end_re.search(line)
        if m:
            if current_output_tp is not None and current_total_tp is not None:
                results.append(BenchResult(
                    target=target,
                    concurrency=current_concurrency,
                    output_throughput=current_output_tp,
                    total_throughput=current_total_tp,
                    gpu_count=gpu_count,
                    ttft=LatencyStats(
                        mean=current_latencies.get("ttft_mean", 0),
                        median=current_latencies.get("ttft_median", 0),
                        p99=current_latencies.get("ttft_p99", 0),
                    ),
                    tpot=LatencyStats(
                        mean=current_latencies.get("tpot_mean", 0),
                        median=current_latencies.get("tpot_median", 0),
                        p99=current_latencies.get("tpot_p99", 0),
                    ),
                    itl=LatencyStats(
                        mean=current_latencies.get("itl_mean", 0),
                        median=current_latencies.get("itl_median", 0),
                        p99=current_latencies.get("itl_p99", 0),
                    ),
                ))
            else:
                print(
                    f"  WARNING: incomplete data for {target} concurrency={current_concurrency}",
                    file=sys.stderr,
                )
            current_concurrency = None

    return config, results


def collect_from_namespace(namespace: str) -> dict[str, str]:
    """Read logs from pareto pods in a single namespace.

    Returns dict of {job_name: log_contents}.
    """
    cmd = [
        "kubectl", "-n", namespace, "get", "pods",
        "-l", "app=pareto",
        "-o", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    data = json.loads(result.stdout)
    pods = data.get("items", [])

    logs = {}
    for pod in pods:
        labels = pod.get("metadata", {}).get("labels", {})
        job_name = labels.get("job-name", "")
        if not job_name:
            continue
        pod_name = pod["metadata"]["name"]
        phase = pod.get("status", {}).get("phase", "")
        if phase not in ("Succeeded", "Running"):
            print(f"  Skipping {namespace}/{pod_name} (phase={phase})")
            continue
        cmd = ["kubectl", "-n", namespace, "logs", pod_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            logs[job_name] = result.stdout
    return logs


def _fmt_ms(v: float):
    """Format a millisecond value for display (round to 1 decimal)."""
    return round(v, 1) if v else ""


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
        ["", "TTFT mean (ms)"] + [_fmt_ms(r.ttft.mean) for r in results],
        ["", "TTFT median (ms)"] + [_fmt_ms(r.ttft.median) for r in results],
        ["", "TTFT p99 (ms)"] + [_fmt_ms(r.ttft.p99) for r in results],
        [],
        ["", "TPOT mean (ms)"] + [_fmt_ms(r.tpot.mean) for r in results],
        ["", "TPOT median (ms)"] + [_fmt_ms(r.tpot.median) for r in results],
        ["", "TPOT p99 (ms)"] + [_fmt_ms(r.tpot.p99) for r in results],
        [],
        ["", "ITL mean (ms)"] + [_fmt_ms(r.itl.mean) for r in results],
        ["", "ITL median (ms)"] + [_fmt_ms(r.itl.median) for r in results],
        ["", "ITL p99 (ms)"] + [_fmt_ms(r.itl.p99) for r in results],
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
    x_title: str = "X",
    y_title: str = "Y",
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
                                "title": x_title,
                            },
                            {
                                "position": "LEFT_AXIS",
                                "title": y_title,
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

    for target in sorted(results_by_target.keys()):
        table_start = len(all_rows)
        table_rows = build_scaling_table(results_by_target[target], target)
        all_rows.extend(table_rows)
        fmt_requests.append(_fmt_bold(results_ws.id, table_start, table_start + 1))

    # Pareto chart data
    pareto_start_row = len(all_rows) + 1
    pareto_rows = build_pareto_data(results_by_target)
    all_rows.extend(pareto_rows)

    pareto_start_idx = pareto_start_row - 1
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
        x_title="Throughput per User (tok/s)",
        y_title="Throughput per GPU (tok/s)",
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

    # Embed job.yaml
    yaml_path = Path(__file__).parent / "job.yaml"
    if yaml_path.exists():
        cfg_rows.append([])
        cfg_rows.append(["Job Template (job.yaml)"])
        yaml_header_row = len(cfg_rows) - 1
        for yaml_line in yaml_path.read_text().splitlines():
            cfg_rows.append([yaml_line])
    else:
        yaml_header_row = None

    cfg_ws.update(range_name="A1", values=cfg_rows)

    cfg_fmt = [
        _fmt_bold(cfg_ws.id, 0, 1),
        _fmt_bold(cfg_ws.id, 2, len(cfg_rows), 0, 1),
    ]
    if yaml_header_row is not None:
        cfg_fmt.append(_fmt_bold(cfg_ws.id, yaml_header_row, yaml_header_row + 1))
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

    pareto_rows = build_pareto_data(results_by_target)
    path = f"{output_dir}/pareto.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in pareto_rows:
            w.writerow(row)
    print(f"Written: {path}")

    ttft_rows = build_ttft_pareto_data(results_by_target)
    path = f"{output_dir}/ttft-pareto.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in ttft_rows:
            w.writerow(row)
    print(f"Written: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect pareto sweep results from Kubernetes Job logs"
    )
    parser.add_argument(
        "--namespace", "-n", action="append", dest="namespaces", default=[],
        help="Namespace to collect from (can be specified multiple times)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=".", help="Directory for output CSVs"
    )
    parser.add_argument(
        "--sheets", metavar="TITLE",
        help="Upload results to a Google Spreadsheet with Pareto chart",
    )
    args = parser.parse_args()

    if not args.namespaces:
        print("No namespaces specified. Use -n <namespace> (repeatable).", file=sys.stderr)
        sys.exit(1)

    print(f"Collecting from {len(args.namespaces)} namespace(s): {', '.join(args.namespaces)}")

    results_by_target: dict[str, list[BenchResult]] = {}
    configs: dict[str, dict[str, str]] = {}

    for namespace in args.namespaces:
        print(f"\n  Namespace: {namespace}")
        all_logs = collect_from_namespace(namespace)

        if not all_logs:
            print(f"    No pareto pod logs found")
            continue

        for job_name, log_text in sorted(all_logs.items()):
            print(f"    Parsing {job_name}...")
            config, results = parse_sweep_log(log_text)

            if not results:
                print(f"      WARNING: no benchmark results found", file=sys.stderr)
                continue

            # Use the config target (CONFIG_NAME) from the log, fall back to namespace
            target = config.get("target", namespace)
            results_by_target[target] = results
            configs[target] = config
            print(f"      {len(results)} result(s), gpu_count={config.get('gpu_count', '?')}")

    if not results_by_target:
        print("\nNo benchmark results parsed from logs", file=sys.stderr)
        sys.exit(1)

    total = sum(len(v) for v in results_by_target.values())
    print(f"\nTotal: {total} benchmark results across {len(results_by_target)} target(s)")

    write_csv(results_by_target, args.output_dir)

    if args.sheets:
        upload_to_sheets(results_by_target, configs, args.sheets)


if __name__ == "__main__":
    main()

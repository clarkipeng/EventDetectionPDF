"""Check manuscript vocabulary, source docs, and required generated figures."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO_ROOT / "paper"

MANUSCRIPT_FILES = [
    PAPER_ROOT / "main.tex",
    *sorted((PAPER_ROOT / "sections").glob("*.tex")),
]

FIGURE_SOURCE_FILES = [
    PAPER_ROOT / "make_plots.py",
]

BANNED_PATTERNS = [
    r"grid prediction",
    r"grid predictions",
    r"time-grid prediction",
    r"PDFR",
    r"PDF regression",
    r"density regression",
    r"dense interval prediction",
    r"event list",
    r"event lists",
    r"event-rate target",
    r"event-rate targets",
    r"occurrence-count target",
    r"occurrence-count targets",
    r"predicted occurrence",
    r"predicted occurrences",
    r"event mass",
    r"conserved event mass",
    r"unit-mass",
    r"unit mass",
    r"mass preservation",
    r"mass-preservation",
    r"additional leaderboards",
    r"architecture diagnostics",
    r"boundary conditions",
    r"expected-event signal",
    r"objective-level",
    r"surrounding detector choices",
    r"state traces?",
    r"supervised quantity",
    r"target definition",
    r"target-level",
    r"binned event targets",
    r"binned event target",
    r"binned event-occurrence",
    r"binned event occurrences",
    r"binned event supervision",
    r"\bbinned events\b",
    r"likelihood target",
    r"hard-target streaming",
    r"detector pipeline",
    r"experiment plan",
    r"should\s+(?:not\s+)?be read as",
    r"best read as",
    r"state-of-the-art",
    r"\bSOTA\b",
    r"statistically significant",
    r"universal improvement",
    r"\bdominates\b",
    r"primary benchmark",
    r"secondary benchmark",
    r"primary evidence",
    r"secondary evidence",
    r"legacy heatmap",
    r"Response-map",
    r"occurrence curves?",
    r"(?<!attention-)gated U-Net",
    r"forward-only sleep models favor segmentation",
    r"CHB-MIT seizure transfer remains challenging",
    r"remaining rows",
    r"current table",
    r"pending results",
    r"ready file",
    r"tracked before",
    r"missing code",
    r"workaround",
    r"run tag",
    r"run_tag",
    r"experiment record",
    r"experiment-records",
    r"objective screen",
    r"learning-rate screen",
    r"learning-rate pilot",
    r"Poisson-objective pilot",
    r"pilot sweep",
    r"sleep matrix",
    r"branch",
    r"hosted\s+Trace",
    r"Trace\s+CLI",
    r"origami-trace",
    r"\botrace\b",
    r"\bW&B\b",
    r"\bwandb\b",
]

BANNED_FIGURE_LABELS = [
    r"Occurrence scores",
    r"Occurrence-score",
    r"occurrence-score",
    r"Predicted occurrence scores",
    r"pending",
    r"Pending complete experiment results",
    r"Attn\.",
    r"(?<!Attention-)Gated U-Net",
    r"Segmenter",
    r"& Transformer &",
]

REQUIRED_FIGURES = {
    PAPER_ROOT / "figures" / "generated" / "method_overview.png": {
        "min_width": 1800,
        "min_height": 700,
        "min_bytes": 70_000,
        "min_sample_colors": 80,
    },
    PAPER_ROOT / "figures" / "generated" / "target_pipeline_column.png": {
        "min_width": 650,
        "min_height": 800,
        "min_bytes": 25_000,
        "min_sample_colors": 60,
    },
    PAPER_ROOT / "figures" / "generated" / "sleep_main_results.png": {
        "min_width": 1800,
        "min_height": 1200,
        "min_bytes": 100_000,
        "min_sample_colors": 80,
    },
    PAPER_ROOT / "figures" / "generated" / "sleep_prediction_example.png": {
        "min_width": 1800,
        "min_height": 1600,
        "min_bytes": 150_000,
        "min_sample_colors": 90,
    },
}

SOURCE_MARKDOWN = {
    REPO_ROOT / "README.md",
    PAPER_ROOT / "PAPER_OPS.md",
}

RETIRED_MARKDOWN = {
    PAPER_ROOT / "EXPERIMENTS.md",
    PAPER_ROOT / "TODO.md",
    PAPER_ROOT / "NEXT_STEPS.md",
    PAPER_ROOT / "VOCABULARY.md",
    PAPER_ROOT / "GUARDRAILS.md",
    PAPER_ROOT / "CLAIM_LEDGER.md",
    PAPER_ROOT / "PROCESS_PIPELINE.md",
    PAPER_ROOT / "FIGURE_STYLE.md",
    PAPER_ROOT / "VERSION_AUDIT.md",
    PAPER_ROOT / "GOAL_AFTER_FEEDBACK.md",
    REPO_ROOT / "TRACE_FOR_AGENTS.md",
    REPO_ROOT / "otrace_comments.md",
}

GENERATED_RESULTS = PAPER_ROOT / "results" / "generated"

HEADLINE_TEXT_FILES = [
    PAPER_ROOT / "sections" / "abstract.tex",
    PAPER_ROOT / "sections" / "results.tex",
]

IGNORED_MARKDOWN_DIRS = {
    REPO_ROOT / ".git",
    REPO_ROOT / ".trace",
    REPO_ROOT / ".uv-cache",
    REPO_ROOT / ".venv",
    REPO_ROOT / "paper" / "build",
    REPO_ROOT / "paper" / "generated_runs",
    REPO_ROOT / "paper" / "results",
}

GATED_RESULT_FILES = {
    "results/generated/seizure_highscore_ready.tex": GENERATED_RESULTS
    / "seizure_highscore_tracker.csv",
    "results/generated/seizure_highscore_strict3_ready.tex": GENERATED_RESULTS
    / "seizure_highscore_strict3_tracker.csv",
}

REQUIRED_GENERATED_RESULTS = {
    GENERATED_RESULTS / "paired_fold_consistency.csv",
    GENERATED_RESULTS / "paired_fold_consistency.tex",
}


def iter_matches(files, patterns):
    compiled = [(pattern, re.compile(pattern, re.IGNORECASE)) for pattern in patterns]
    for path in files:
        if not path.exists():
            yield path, 0, "<missing file>", "missing manuscript file"
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for label, pattern in compiled:
                if pattern.search(line):
                    yield path, lineno, line.strip(), label


def check_manuscript():
    failures = []
    for path, lineno, line, label in iter_matches(MANUSCRIPT_FILES, BANNED_PATTERNS):
        if lineno == 0:
            failures.append(f"{path.relative_to(REPO_ROOT)}: {label}")
        else:
            rel = path.relative_to(REPO_ROOT)
            failures.append(f"{rel}:{lineno}: banned phrase `{label}`: {line}")
    return failures


def check_figure_source_labels():
    failures = []
    files = [
        *FIGURE_SOURCE_FILES,
        *sorted(GENERATED_RESULTS.glob("*.tex")),
    ]
    for path, lineno, line, label in iter_matches(
        files,
        BANNED_FIGURE_LABELS,
    ):
        if lineno == 0:
            failures.append(f"{path.relative_to(REPO_ROOT)}: {label}")
        else:
            rel = path.relative_to(REPO_ROOT)
            failures.append(f"{rel}:{lineno}: banned figure/generated label `{label}`: {line}")
    return failures


def check_figures():
    failures = []
    for path, requirements in REQUIRED_FIGURES.items():
        rel = path.relative_to(REPO_ROOT)
        if not path.exists():
            failures.append(f"missing required figure: {rel}")
            continue
        if path.stat().st_size < requirements["min_bytes"]:
            failures.append(
                f"{rel}: figure file is unexpectedly small "
                f"({path.stat().st_size} bytes)"
            )
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
                if width < requirements["min_width"] or height < requirements["min_height"]:
                    failures.append(
                        f"{rel}: figure dimensions are too small "
                        f"({width}x{height})"
                    )
                sample = image.convert("RGB").resize((64, 64))
                colors = sample.getcolors(maxcolors=4096)
                color_count = len(colors) if colors is not None else 4097
                if color_count < requirements["min_sample_colors"]:
                    failures.append(
                        f"{rel}: figure has too few sampled colors "
                        f"({color_count})"
                    )
        except (OSError, UnidentifiedImageError) as exc:
            failures.append(f"{rel}: cannot read figure as an image: {exc}")
    return failures


def check_markdown():
    failures = []
    for path in sorted(SOURCE_MARKDOWN):
        if not path.exists():
            failures.append(f"missing source markdown: {path.relative_to(REPO_ROOT)}")
    for path in sorted(RETIRED_MARKDOWN):
        if path.exists():
            failures.append(f"retired markdown file exists: {path.relative_to(REPO_ROOT)}")
    source_markdown = {
        path
        for path in REPO_ROOT.rglob("*.md")
        if not any(path.is_relative_to(ignored) for ignored in IGNORED_MARKDOWN_DIRS)
    }
    unexpected = source_markdown - SOURCE_MARKDOWN
    missing = SOURCE_MARKDOWN - source_markdown
    for path in sorted(unexpected):
        failures.append(f"unexpected source markdown: {path.relative_to(REPO_ROOT)}")
    for path in sorted(missing):
        failures.append(f"missing source markdown in inventory: {path.relative_to(REPO_ROOT)}")
    return failures


def check_generated_result_gates():
    failures = []
    manuscript_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in MANUSCRIPT_FILES
        if path.exists()
    )
    disallowed_inputs = re.findall(
        r"\\input\{(results/generated/[^}]*_(?:completed|tracker)\.tex)\}",
        manuscript_text,
    )
    for tex_name in sorted(set(disallowed_inputs)):
        failures.append(
            f"manuscript inputs intermediate generated result instead of a gated table: {tex_name}"
        )
    for tex_name, tracker_path in GATED_RESULT_FILES.items():
        input_command = f"\\input{{{tex_name}}}"
        gate_command = f"\\IfFileExists{{{tex_name}}}"
        if input_command in manuscript_text and gate_command not in manuscript_text:
            failures.append(f"generated result input is not gated: {tex_name}")
        ready_path = PAPER_ROOT / tex_name
        if ready_path.exists():
            if not tracker_path.exists():
                failures.append(
                    f"{ready_path.relative_to(REPO_ROOT)} exists but "
                    f"{tracker_path.relative_to(REPO_ROOT)} is missing"
                )
                continue
            rows = read_csv_rows(tracker_path)
            incomplete_rows = [
                index
                for index, row in enumerate(rows, 2)
                if any(value.strip() == "--" for value in row.values())
            ]
            if incomplete_rows:
                failures.append(
                    f"{ready_path.relative_to(REPO_ROOT)} exists but "
                    f"{tracker_path.relative_to(REPO_ROOT)} has incomplete rows "
                    f"at CSV lines {incomplete_rows}"
                )
    for path in sorted(REQUIRED_GENERATED_RESULTS):
        if not path.exists():
            failures.append(f"missing required generated result: {path.relative_to(REPO_ROOT)}")
    return failures


def check_paired_fold_evidence():
    failures = []
    csv_path = GENERATED_RESULTS / "paired_fold_consistency.csv"
    tex_path = GENERATED_RESULTS / "paired_fold_consistency.tex"
    try:
        rows = read_csv_rows(csv_path)
    except FileNotFoundError:
        return [f"missing paired-fold consistency CSV: {csv_path.relative_to(REPO_ROOT)}"]

    required_columns = {
        "comparison",
        "paired_records",
        "wins",
        "sign_p",
        "mean_delta",
        "min_delta",
    }
    for line_number, row in enumerate(rows, 2):
        missing = required_columns - set(row)
        if missing:
            failures.append(
                f"{csv_path.relative_to(REPO_ROOT)} line {line_number} "
                f"is missing columns: {sorted(missing)}"
            )
            continue
        try:
            paired_records = int(row["paired_records"])
            wins = int(row["wins"])
            sign_p = float(row["sign_p"])
            mean_delta = float(row["mean_delta"])
            min_delta = float(row["min_delta"])
        except ValueError as exc:
            failures.append(
                f"{csv_path.relative_to(REPO_ROOT)} line {line_number} "
                f"has nonnumeric paired-fold evidence: {exc}"
            )
            continue
        if paired_records <= 0:
            failures.append(
                f"{csv_path.relative_to(REPO_ROOT)} line {line_number} has no paired records"
            )
        if wins != paired_records:
            failures.append(
                f"{csv_path.relative_to(REPO_ROOT)} line {line_number} "
                f"is no longer all-wins evidence: wins={wins}, pairs={paired_records}"
            )
        if not 0.0 < sign_p <= 0.5:
            failures.append(
                f"{csv_path.relative_to(REPO_ROOT)} line {line_number} "
                f"has invalid sign-check probability: {sign_p}"
            )
        if mean_delta <= 0 or min_delta <= 0:
            failures.append(
                f"{csv_path.relative_to(REPO_ROOT)} line {line_number} "
                f"has nonpositive paired delta: mean={mean_delta}, min={min_delta}"
            )

    if not tex_path.exists():
        failures.append(f"missing paired-fold consistency TeX: {tex_path.relative_to(REPO_ROOT)}")
        return failures
    tex_text = tex_path.read_text(encoding="utf-8")
    if r"$p_{\mathrm{sign}}$" not in tex_text:
        failures.append(
            f"{tex_path.relative_to(REPO_ROOT)} is missing the sign-check probability column"
        )
    if "<0.001" not in tex_text:
        failures.append(
            f"{tex_path.relative_to(REPO_ROOT)} is missing compact sleep sign-check values"
        )
    if "0.062" not in tex_text:
        failures.append(
            f"{tex_path.relative_to(REPO_ROOT)} is missing CHB-MIT sign-check values"
        )
    return failures


def check_review_pdf_references():
    failures = []
    ops_path = PAPER_ROOT / "PAPER_OPS.md"
    if not ops_path.exists():
        return ["missing paper operations file for review-PDF checks"]
    text = ops_path.read_text(encoding="utf-8")
    latest_match = re.search(
        r"Latest rendered manuscript checkpoint:\s*\n\s*`([^`]+\.pdf)`",
        text,
    )
    current_match = re.search(
        r"Current review PDF:\s*\n`([^`]+\.pdf)`",
        text,
    )
    if latest_match and current_match:
        latest_path = latest_match.group(1)
        current_path = current_match.group(1)
        if latest_path != current_path:
            failures.append(
                "PAPER_OPS.md latest checkpoint and current review PDF disagree: "
                f"{latest_path} != {current_path}"
            )
        version_notes = text.split("Version comparison notes:", 1)[-1]
        current_name = Path(current_path).name
        if current_name not in version_notes:
            failures.append(
                "current review PDF is not listed in version comparison notes: "
                f"{current_name}"
            )
    elif latest_match or current_match:
        failures.append(
            "PAPER_OPS.md should either list both latest and current review PDFs or neither"
        )
    for pdf_name in sorted(set(re.findall(r"`(paper/build/versions/[^`]+\.pdf)`", text))):
        pdf_path = REPO_ROOT / pdf_name
        if not pdf_path.exists():
            failures.append(f"PAPER_OPS.md references missing review PDF: {pdf_name}")
    return failures


def read_csv_rows(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rounded(value, signed=False):
    formatted = f"{value:+.3f}" if signed else f"{value:.3f}"
    return formatted.replace("-0.000", "0.000")


def find_row(rows, **expected):
    matches = []
    for row in rows:
        for key, value in expected.items():
            if key not in row:
                break
            if isinstance(value, float):
                try:
                    if abs(float(row[key]) - value) > 1e-9:
                        break
                except ValueError:
                    break
            elif row[key] != value:
                break
        else:
            matches.append(row)
    if len(matches) != 1:
        criteria = ", ".join(f"{key}={value!r}" for key, value in expected.items())
        raise ValueError(f"expected one generated-result row for {criteria}, found {len(matches)}")
    return matches[0]


def headline_numbers():
    objective_rows = read_csv_rows(GENERATED_RESULTS / "sleep_objective_sweep.csv")
    tolerance_rows = read_csv_rows(GENERATED_RESULTS / "tolerance_curves.csv")
    architecture_rows = read_csv_rows(GENERATED_RESULTS / "sleep_main_architecture.csv")
    seizure_rows = read_csv_rows(GENERATED_RESULTS / "seizure_highscore_completed.csv")

    objective_values = {}
    selected_postprocess = {}
    for objective in [
        "density_hard",
        "density_gau",
        "density_custom",
        "seg",
        "seg_weighted",
        "seg_focal",
    ]:
        row = find_row(objective_rows, objective=objective)
        objective_values[objective] = float(row["mAP"])
        selected_postprocess[objective] = row["postprocess_objective"]

    tolerance_values = {}
    for objective, minutes in [
        ("density_hard", 1),
        ("seg", 1),
        ("density_hard", 3),
        ("seg", 3),
        ("density_hard", 30),
        ("seg", 30),
    ]:
        # Sleep tolerances are stored in five-second steps; 12 steps is one minute.
        row = find_row(
            tolerance_rows,
            dataset="sleep",
            model="gru",
            objective=objective,
            run_tag="objective_e20_bs32_eval5",
            postprocess_objective=selected_postprocess[objective],
            tolerance=float(minutes * 12),
        )
        tolerance_values[(objective, minutes)] = float(row["mean"])

    architecture_values = {}
    for row in architecture_rows:
        architecture_values[(row["model"], "delta")] = float(row["$\\Delta$"])

    seizure_values = {}
    for setting in ["GRU 3x128, 2 s stride", "GRU 3x128, 1 s stride"]:
        row = find_row(seizure_rows, setting=setting)
        for method in ["BDL-Hard", "BDL-Gaussian", "Cross-entropy"]:
            seizure_values[(setting, method)] = float(row[method])

    return objective_values, tolerance_values, architecture_values, seizure_values


def check_headline_numbers():
    failures = []
    try:
        (
            objective_values,
            tolerance_values,
            architecture_values,
            seizure_values,
        ) = headline_numbers()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return [f"cannot verify headline numbers from generated results: {exc}"]

    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in HEADLINE_TEXT_FILES
        if path.exists()
    )
    expected_numbers = [
        ("BDL-Hard sleep mAP", rounded(objective_values["density_hard"])),
        ("Cross-entropy sleep mAP", rounded(objective_values["seg"])),
        ("BDL-Gaussian sleep mAP", rounded(objective_values["density_gau"])),
        ("BDL-Tolerance sleep mAP", rounded(objective_values["density_custom"])),
        ("Weighted CE sleep mAP", rounded(objective_values["seg_weighted"])),
        ("Focal sleep mAP", rounded(objective_values["seg_focal"])),
        ("BDL-Hard one-minute AP", rounded(tolerance_values[("density_hard", 1)])),
        ("Cross-entropy one-minute AP", rounded(tolerance_values[("seg", 1)])),
        ("BDL-Hard three-minute AP", rounded(tolerance_values[("density_hard", 3)])),
        ("Cross-entropy three-minute AP", rounded(tolerance_values[("seg", 3)])),
        ("BDL-Hard thirty-minute AP", rounded(tolerance_values[("density_hard", 30)])),
        ("Cross-entropy thirty-minute AP", rounded(tolerance_values[("seg", 30)])),
        (
            "CHB-MIT two-second BDL-Gaussian mAP",
            rounded(seizure_values[("GRU 3x128, 2 s stride", "BDL-Gaussian")]),
        ),
        (
            "CHB-MIT two-second cross-entropy mAP",
            rounded(seizure_values[("GRU 3x128, 2 s stride", "Cross-entropy")]),
        ),
        (
            "CHB-MIT one-second BDL-Hard mAP",
            rounded(seizure_values[("GRU 3x128, 1 s stride", "BDL-Hard")]),
        ),
        (
            "CHB-MIT one-second cross-entropy mAP",
            rounded(seizure_values[("GRU 3x128, 1 s stride", "Cross-entropy")]),
        ),
    ]
    for label, value in expected_numbers:
        if value not in text:
            failures.append(f"headline number missing from manuscript: {label} = {value}")
    architecture_models = {model for model, field in architecture_values if field == "delta"}
    required_architecture_models = {"gru", "unet"}
    if not required_architecture_models.issubset(architecture_models):
        failures.append(
            "sleep architecture table is missing required recurrent/convolutional rows: "
            f"{sorted(required_architecture_models - architecture_models)}"
        )
    if len(architecture_models) < 3:
        failures.append(
            "sleep architecture table should include at least three architecture rows"
        )
    for (model, field), value in architecture_values.items():
        if field == "delta" and value <= 0:
            failures.append(
                f"sleep architecture table has nonpositive BDL delta for {model}: {value:.3f}"
            )
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only check manuscript text and source markdown.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    failures = []
    failures.extend(check_manuscript())
    failures.extend(check_figure_source_labels())
    failures.extend(check_markdown())
    failures.extend(check_generated_result_gates())
    failures.extend(check_paired_fold_evidence())
    failures.extend(check_review_pdf_references())
    failures.extend(check_headline_numbers())
    if not args.skip_figures:
        failures.extend(check_figures())

    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    print("manuscript checks passed")


if __name__ == "__main__":
    main()

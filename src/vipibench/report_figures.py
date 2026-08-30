"""Publication-oriented figures with presentation-safe Vietnamese labels."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ARM_LABELS = {
    "none": "Không cơ chế",
    "detector_only": "Chỉ bộ phát hiện",
    "policy_only": "Chỉ chính sách",
    "hybrid": "Kết hợp",
}
ARM_ORDER = tuple(ARM_LABELS)
CONDITION_LABELS = {
    "source_tag_spoof": "Giả mạo nhãn nguồn",
    "long_context": "Ngữ cảnh dài",
    "quoted_boundary": "Ranh giới trích dẫn",
    "format_noise": "Nhiễu định dạng",
    "code_mix": "Trộn mã nguồn",
}
VIOLATION_LABELS = {
    "CANARY_DISCLOSURE": "Lộ chuỗi kiểm tra bí mật",
    "UNKNOWN_CONTEXT_REF": "Tham chiếu ngữ cảnh không hợp lệ",
    "UNKNOWN_TOOL": "Công cụ không xác định",
    "INVALID_TOOL_ARGUMENT": "Tham số công cụ không hợp lệ",
    "CAPABILITY_SCOPE_VIOLATION": "Vượt phạm vi năng lực",
    "UNKNOWN_AUTHORIZATION_REF": "Tham chiếu cấp quyền không hợp lệ",
    "UNAUTHORIZED_TOOL_CALL": "Gọi công cụ khi chưa được phép",
    "INVALID_STATE_TRANSITION": "Chuyển trạng thái không hợp lệ",
    "ATTACK_OBJECTIVE_ACHIEVED": "Đạt mục tiêu tấn công",
    "SECURITY_INVARIANT_VIOLATION": "Vi phạm bất biến an toàn",
}
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
VISIBLE_CODE_PATTERN = r"(?i)\b(?:rq|rg|h)\s*[-_]?\d+\b"
VISIBLE_VERSION_PATTERN = r"(?i)\bv\d+(?:\.\d+)*\b"


@dataclass(frozen=True)
class FigureAsset:
    number: int
    slug: str
    title: str
    caption: str
    recommended_section: str
    data_source: str
    unit: str
    sample_size: str
    headers: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    draw: Callable[[], Figure]


def render_report_figures(
    artifacts: Mapping[str, Mapping[str, object]], output_dir: Path
) -> list[dict[str, object]]:
    """Render the complete fixed report pack and return its reader-facing catalog."""

    output_dir.mkdir(parents=True, exist_ok=False)
    figures = _figure_specs(artifacts)
    catalog: list[dict[str, object]] = []
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "vipibench-report-figures",
            "pdf.fonttype": 42,
        }
    ):
        for asset in figures:
            stem = f"hinh_{asset.number:02d}_{asset.slug}"
            figure = asset.draw()
            files = _save_figure(figure, output_dir / stem)
            csv_path = output_dir / f"{stem}.csv"
            _write_csv(csv_path, asset.headers, asset.rows)
            catalog.append(
                {
                    "number": asset.number,
                    "title": asset.title,
                    "caption": asset.caption,
                    "recommended_section": asset.recommended_section,
                    "data_source": asset.data_source,
                    "unit": asset.unit,
                    "sample_size": asset.sample_size,
                    "files": [*files, csv_path.name],
                }
            )
            figure.clear()
    return catalog


def _figure_specs(
    artifacts: Mapping[str, Mapping[str, object]],
) -> list[FigureAsset]:
    ablation = _artifact(artifacts, "encoder_ablation")
    diagnostics = _artifact(artifacts, "diagnostic_analysis")
    static = _artifact(artifacts, "static_analysis")
    joint = _artifact(artifacts, "joint_analysis")
    adaptive = _artifact(artifacts, "adaptive_analysis")
    telemetry = _artifact(artifacts, "runtime_telemetry")
    return [
        _architecture_spec(),
        _evidence_pipeline_spec(),
        _context_effect_spec(ablation),
        _diagnostic_degradation_spec(diagnostics),
        _system_comparison_spec(static),
        _pareto_spec(static),
        _joint_decision_spec(joint),
        _adaptive_gap_spec(adaptive),
        _calibration_spec(diagnostics),
        _failure_taxonomy_spec(adaptive),
        _runtime_spec(telemetry),
    ]


def _architecture_spec() -> FigureAsset:
    nodes = (
        ("Yêu cầu, dữ liệu\nvà ngữ cảnh", 0.08, 0.67),
        ("Biểu diễn nội dung\nvà nguồn gốc", 0.38, 0.67),
        ("Bộ phát hiện\nnguy cơ", 0.68, 0.67),
        ("Cổng quyết định\nchính sách", 0.68, 0.25),
        ("Tác nhân đích và\nmôi trường cô lập", 0.38, 0.25),
        ("Đánh giá an toàn,\ntiện ích và chi phí", 0.08, 0.25),
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))

    def draw() -> Figure:
        figure = _new_figure((11.5, 5.2))
        axis = figure.add_subplot(111)
        _draw_flow(axis, nodes, edges, "Kiến trúc đánh giá hệ thống theo ngữ cảnh")
        return figure

    rows = tuple(
        (index + 1, label.replace("\n", " ")) for index, (label, _, _) in enumerate(nodes)
    )
    return FigureAsset(
        number=1,
        slug="kien_truc_he_thong",
        title="Kiến trúc đánh giá hệ thống theo ngữ cảnh",
        caption=(
            "Hình 1. Luồng xử lý từ yêu cầu có ngữ cảnh đến quyết định chính sách, thực thi "
            "trong môi trường cô lập và đánh giá đồng thời an toàn, tiện ích cùng chi phí."
        ),
        recommended_section="Chương phương pháp và thiết kế hệ thống",
        data_source="Thiết kế triển khai và hợp đồng đánh giá của hệ thống",
        unit="Sơ đồ khối",
        sample_size="Không áp dụng",
        headers=("Thứ tự", "Thành phần"),
        rows=rows,
        draw=draw,
    )


def _evidence_pipeline_spec() -> FigureAsset:
    nodes = (
        ("Tiền kiểm môi trường\nvà dữ liệu", 0.04, 0.66),
        ("Huấn luyện và\nchọn mô hình", 0.28, 0.66),
        ("Đánh giá hệ thống\ncố định", 0.52, 0.66),
        ("Tìm kiếm tấn công\nthích nghi", 0.76, 0.66),
        ("Phân tích thống kê\nđã khóa", 0.64, 0.22),
        ("Hậu kiểm bằng chứng\nvà tính toàn vẹn", 0.36, 0.22),
        ("Bảng, hình và\nchú thích báo cáo", 0.08, 0.22),
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6))

    def draw() -> Figure:
        figure = _new_figure((12.5, 5.4))
        axis = figure.add_subplot(111)
        _draw_flow(axis, nodes, edges, "Quy trình tạo và kiểm định bằng chứng thực nghiệm")
        return figure

    rows = tuple(
        (index + 1, label.replace("\n", " ")) for index, (label, _, _) in enumerate(nodes)
    )
    return FigureAsset(
        number=2,
        slug="quy_trinh_bang_chung",
        title="Quy trình tạo và kiểm định bằng chứng thực nghiệm",
        caption=(
            "Hình 2. Chuỗi công việc tuần tự từ tiền kiểm đến vật liệu báo cáo. Hình và bảng "
            "chỉ được tạo sau khi hậu kiểm xác nhận tính đầy đủ, tính trực tiếp và liên kết hash "
            "của bằng chứng."
        ),
        recommended_section="Chương phương pháp thực nghiệm",
        data_source="Kế hoạch thực thi tuần tự và cổng hậu kiểm của thí nghiệm",
        unit="Sơ đồ quy trình",
        sample_size="Không áp dụng",
        headers=("Thứ tự", "Công đoạn"),
        rows=rows,
        draw=draw,
    )


def _context_effect_spec(artifact: Mapping[str, object]) -> FigureAsset:
    effects = _mapping(artifact.get("primary_effects"), "primary effects")
    definitions = (
        (
            "content_provenance_minus_text_only",
            "Đầy đủ ngữ cảnh so với chỉ văn bản",
        ),
        (
            "content_provenance_minus_role_only",
            "Đầy đủ ngữ cảnh so với chỉ vai trò",
        ),
    )
    values: list[tuple[str, float, float, float, int, int]] = []
    for key, label in definitions:
        effect = _mapping(effects.get(key), f"effect {key}")
        values.append(
            (
                label,
                _finite(effect.get("estimate"), "effect estimate"),
                _finite(effect.get("lower_95"), "effect lower bound"),
                _finite(effect.get("upper_95"), "effect upper bound"),
                _positive_int(effect.get("pair_count"), "paired comparison count"),
                _positive_int(effect.get("family_count"), "source group count"),
            )
        )
    _validate_intervals([(item[1], item[2], item[3]) for item in values])
    sample_size = _paired_sample_summary(values)

    def draw() -> Figure:
        figure = _new_figure((9.2, 4.6))
        axis = figure.add_subplot(111)
        y = np.arange(len(values))
        axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
        for index, (_, estimate, lower, upper, _, _) in enumerate(values):
            axis.hlines(index, lower, upper, color=PALETTE[index], linewidth=3)
            axis.scatter(estimate, index, s=65, color=PALETTE[index], zorder=3)
        axis.set_yticks(y, [item[0] for item in values])
        axis.set_xlabel("Chênh lệch biên xác suất đã hiệu chỉnh")
        axis.set_title("Hiệu ứng của thông tin ngữ cảnh và nguồn gốc")
        axis.grid(axis="x", alpha=0.22)
        axis.invert_yaxis()
        return figure

    rows = tuple(
        (label, estimate, lower, upper, pairs, families)
        for label, estimate, lower, upper, pairs, families in values
    )
    return FigureAsset(
        number=3,
        slug="hieu_ung_ngu_canh_nguon_goc",
        title="Hiệu ứng của thông tin ngữ cảnh và nguồn gốc",
        caption=(
            "Hình 3. Chênh lệch biên xác suất đã hiệu chỉnh giữa biểu diễn đầy đủ ngữ cảnh "
            "và hai cấu hình lược bỏ. Điểm biểu diễn ước lượng; đoạn ngang biểu diễn khoảng tin "
            f"cậy 95%. {sample_size}."
        ),
        recommended_section="Chương kết quả thực nghiệm",
        data_source="Phân tích ghép cặp trên tập đối sánh nguồn gốc đã khóa",
        unit="Chênh lệch biên xác suất",
        sample_size=sample_size,
        headers=(
            "So sánh",
            "Ước lượng",
            "Cận dưới 95%",
            "Cận trên 95%",
            "Số cặp",
            "Số nhóm nguồn",
        ),
        rows=rows,
        draw=draw,
    )


def _diagnostic_degradation_spec(artifact: Mapping[str, object]) -> FigureAsset:
    order = _string_sequence(artifact.get("formal_condition_order"), "condition order")
    comparisons = _mapping(artifact.get("comparisons"), "diagnostic comparisons")
    values: list[tuple[str, float, float, float, str]] = []
    for condition in order:
        if condition not in CONDITION_LABELS:
            raise ValueError(f"unknown presentation condition: {condition}")
        comparison = _mapping(comparisons.get(condition), f"comparison {condition}")
        effect = _mapping(
            comparison.get("signed_margin_degradation"), f"degradation {condition}"
        )
        interval = _mapping(effect.get("confidence_interval_95"), f"interval {condition}")
        pair_counts = _mapping(comparison.get("pair_count_by_seed"), f"pair counts {condition}")
        count_text = ", ".join(
            f"{seed}: {int(count)}" for seed, count in sorted(pair_counts.items())
        )
        values.append(
            (
                CONDITION_LABELS[condition],
                _finite(effect.get("estimate"), "degradation estimate"),
                _finite(interval.get("lower_95"), "degradation lower bound"),
                _finite(interval.get("upper_95"), "degradation upper bound"),
                count_text,
            )
        )
    _validate_intervals([(item[1], item[2], item[3]) for item in values])
    family_count = _positive_int(
        _mapping(artifact.get("source_family_paired_reference"), "paired reference").get(
            "expected_source_family_count"
        ),
        "source group count",
    )
    sample_size = f"{family_count} nhóm nguồn; số cặp của từng phép thử được ghi trong bảng CSV"

    def draw() -> Figure:
        figure = _new_figure((9.6, 5.8))
        axis = figure.add_subplot(111)
        y = np.arange(len(values))
        axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
        for index, (_, estimate, lower, upper, _) in enumerate(values):
            axis.hlines(index, lower, upper, color=PALETTE[index], linewidth=3)
            axis.scatter(estimate, index, s=60, color=PALETTE[index], zorder=3)
        axis.set_yticks(y, [item[0] for item in values])
        axis.set_xlabel("Mức suy giảm biên xác suất so với điều kiện chuẩn")
        axis.set_title("Độ bền của bộ phát hiện trước các điều kiện gây nhiễu")
        axis.grid(axis="x", alpha=0.22)
        axis.invert_yaxis()
        return figure

    return FigureAsset(
        number=4,
        slug="suy_giam_theo_dieu_kien_nhieu",
        title="Độ bền của bộ phát hiện trước các điều kiện gây nhiễu",
        caption=(
            "Hình 4. Mức suy giảm biên xác suất của năm điều kiện gây nhiễu so với điều kiện "
            "chuẩn. Giá trị dương biểu thị mức suy giảm lớn hơn; đoạn ngang là khoảng tin cậy "
            f"95%. Cỡ mẫu gồm {family_count} nhóm nguồn."
        ),
        recommended_section="Chương kết quả thực nghiệm",
        data_source="Phân tích chẩn đoán ghép cặp trên dự đoán kiểm thử được lưu giữ",
        unit="Chênh lệch biên xác suất",
        sample_size=sample_size,
        headers=(
            "Điều kiện",
            "Ước lượng suy giảm",
            "Cận dưới 95%",
            "Cận trên 95%",
            "Số cặp theo hạt giống",
        ),
        rows=tuple(values),
        draw=draw,
    )


def _system_comparison_spec(artifact: Mapping[str, object]) -> FigureAsset:
    metrics = _mapping(artifact.get("metrics"), "system metrics")
    intervals = _mapping(
        _mapping(artifact.get("confidence_intervals"), "confidence intervals").get("arms"),
        "arm confidence intervals",
    )
    definitions = (
        ("attack_success_rate", "Tỷ lệ tấn công thành công"),
        ("containment_rate", "Tỷ lệ ngăn chặn"),
        ("clean_utility_rate", "Tỷ lệ duy trì tiện ích sạch"),
        ("false_block_rate", "Tỷ lệ chặn nhầm"),
    )
    values: dict[str, list[tuple[str, float, float, float, int]]] = {}
    for metric_name, metric_label in definitions:
        rows: list[tuple[str, float, float, float, int]] = []
        for arm in ARM_ORDER:
            metric = _mapping(
                _mapping(metrics.get(arm), f"metrics {arm}").get(metric_name),
                metric_name,
            )
            interval = _mapping(
                _mapping(intervals.get(arm), f"intervals {arm}").get(metric_name),
                f"interval {arm} {metric_name}",
            )
            rows.append(
                (
                    ARM_LABELS[arm],
                    100 * _probability(metric.get("value"), f"{metric_name} value"),
                    100 * _probability(interval.get("lower_95"), f"{metric_name} lower"),
                    100 * _probability(interval.get("upper_95"), f"{metric_name} upper"),
                    _positive_int(metric.get("denominator"), f"{metric_name} denominator"),
                )
            )
        _validate_intervals([(item[1], item[2], item[3]) for item in rows])
        values[metric_label] = rows
    paired_count = _positive_int(
        _mapping(artifact.get("confidence_intervals"), "confidence intervals").get(
            "paired_episode_count"
        ),
        "paired episode count",
    )
    denominators = sorted({row[4] for rows in values.values() for row in rows})
    denominator_text = ", ".join(str(value) for value in denominators)
    sample_size = (
        f"{paired_count} tình huống ghép cặp; các mẫu số theo chỉ tiêu: {denominator_text}"
    )

    def draw() -> Figure:
        figure = _new_figure((12.5, 8.2))
        for panel_index, (metric_label, rows) in enumerate(values.items(), start=1):
            axis = figure.add_subplot(2, 2, panel_index)
            x = np.arange(len(rows))
            heights = [row[1] for row in rows]
            colors = PALETTE[: len(rows)]
            axis.bar(x, heights, color=colors, width=0.68)
            axis.vlines(
                x,
                [row[2] for row in rows],
                [row[3] for row in rows],
                color="#222222",
                linewidth=1.2,
            )
            axis.scatter(x, heights, marker="_", s=130, color="#222222", zorder=3)
            axis.set_xticks(x, [row[0] for row in rows], rotation=18, ha="right")
            axis.set_ylim(0, 105)
            axis.set_ylabel("Tỷ lệ (%)")
            axis.set_title(metric_label)
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle("So sánh bốn cấu hình bảo vệ của hệ thống", fontsize=14)
        return figure

    table_rows = tuple(
        (metric_label, *row) for metric_label, rows in values.items() for row in rows
    )
    return FigureAsset(
        number=5,
        slug="so_sanh_cau_hinh_bao_ve",
        title="So sánh bốn cấu hình bảo vệ của hệ thống",
        caption=(
            "Hình 5. So sánh bốn cấu hình theo hai chỉ tiêu an toàn và hai chỉ tiêu tiện ích. "
            "Cột biểu diễn ước lượng theo nhóm nguồn; thanh sai số biểu diễn khoảng tin cậy 95%. "
            f"Tổng cộng {paired_count} tình huống được đánh giá ghép cặp."
        ),
        recommended_section="Chương kết quả thực nghiệm",
        data_source="Phân tích bốn cấu hình từ kết quả thực thi ghép cặp đã hậu kiểm",
        unit="Phần trăm",
        sample_size=sample_size,
        headers=(
            "Chỉ tiêu",
            "Cấu hình",
            "Ước lượng (%)",
            "Cận dưới 95%",
            "Cận trên 95%",
            "Mẫu số",
        ),
        rows=table_rows,
        draw=draw,
    )


def _pareto_spec(artifact: Mapping[str, object]) -> FigureAsset:
    frontier = _mapping(artifact.get("pareto_frontier"), "pareto frontier")
    if frontier.get("status") != "PASS":
        raise ValueError("pareto frontier is not available")
    points = _mapping(frontier.get("points"), "pareto points")
    frontier_arms = set(_string_sequence(frontier.get("frontier_arms"), "frontier arms"))
    values: list[tuple[str, float, float, bool]] = []
    for arm in ARM_ORDER:
        point = _mapping(points.get(arm), f"pareto point {arm}")
        values.append(
            (
                ARM_LABELS[arm],
                100 * _probability(point.get("attack_success_rate"), "attack success rate"),
                100 * _probability(point.get("clean_utility_rate"), "clean utility rate"),
                arm in frontier_arms,
            )
        )
    paired_count = _positive_int(
        _mapping(artifact.get("confidence_intervals"), "confidence intervals").get(
            "paired_episode_count"
        ),
        "paired episode count",
    )

    def draw() -> Figure:
        figure = _new_figure((8.2, 6.4))
        axis = figure.add_subplot(111)
        for index, (label, security, utility, on_frontier) in enumerate(values):
            axis.scatter(
                security,
                utility,
                s=120 if on_frontier else 75,
                color=PALETTE[index],
                edgecolor="#111111" if on_frontier else "white",
                linewidth=1.4,
                zorder=3,
            )
            axis.annotate(label, (security, utility), xytext=(7, 6), textcoords="offset points")
        frontier_points = sorted(
            [(security, utility) for _, security, utility, active in values if active]
        )
        if len(frontier_points) > 1:
            axis.plot(
                [point[0] for point in frontier_points],
                [point[1] for point in frontier_points],
                color="#333333",
                linestyle="--",
                linewidth=1.2,
            )
        axis.set_xlabel("Tỷ lệ tấn công thành công (%) — thấp hơn là tốt hơn")
        axis.set_ylabel("Tỷ lệ duy trì tiện ích sạch (%) — cao hơn là tốt hơn")
        axis.set_title("Đánh đổi giữa an toàn và tiện ích")
        axis.set_xlim(-3, 103)
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.22)
        return figure

    return FigureAsset(
        number=6,
        slug="danh_doi_an_toan_tien_ich",
        title="Đánh đổi giữa an toàn và tiện ích",
        caption=(
            "Hình 6. Mỗi điểm là một cấu hình bảo vệ trong không gian an toàn–tiện ích. "
            "Điểm có viền đậm thuộc biên không bị cấu hình khác chi phối theo hai chỉ tiêu. "
            f"Phân tích sử dụng {paired_count} tình huống ghép cặp."
        ),
        recommended_section="Chương kết quả và thảo luận",
        data_source="Hai chỉ tiêu chính của phân tích bốn cấu hình",
        unit="Phần trăm",
        sample_size=f"{paired_count} tình huống ghép cặp",
        headers=(
            "Cấu hình",
            "Tấn công thành công (%)",
            "Duy trì tiện ích sạch (%)",
            "Thuộc biên không bị chi phối",
        ),
        rows=tuple(values),
        draw=draw,
    )


def _joint_decision_spec(artifact: Mapping[str, object]) -> FigureAsset:
    definitions = (
        ("security", "An toàn trước tấn công"),
        ("utility", "Duy trì tiện ích sạch"),
    )
    values: list[tuple[str, float, float, float, int, int, bool]] = []
    for key, label in definitions:
        component = _mapping(artifact.get(key), f"joint component {key}")
        values.append(
            (
                label,
                _finite(component.get("point_effect"), f"{key} point effect"),
                _finite(
                    component.get("marginal_one_sided_lower_bound_95"),
                    f"{key} lower bound",
                ),
                _finite(component.get("locked_margin"), f"{key} margin"),
                _positive_int(component.get("paired_episode_count"), f"{key} episode count"),
                _positive_int(component.get("family_count"), f"{key} family count"),
                _boolean(component.get("bound_passes_locked_margin"), f"{key} decision"),
            )
        )
    sample_sizes = _mapping(artifact.get("sample_sizes"), "joint sample sizes")
    sample_size = (
        f"{_positive_int(sample_sizes.get('injection'), 'injection sample size')} tình huống tấn "
        f"công và {_positive_int(sample_sizes.get('benign'), 'benign sample size')} tình huống sạch"
    )

    def draw() -> Figure:
        figure = _new_figure((9.2, 4.8))
        axis = figure.add_subplot(111)
        y = np.arange(len(values))
        for index, (_, point, lower, margin, _, _, passed) in enumerate(values):
            color = PALETTE[2] if passed else PALETTE[5]
            axis.hlines(index, lower, point, color=color, linewidth=3)
            axis.scatter(point, index, marker="D", s=65, color=color, label=None)
            axis.scatter(lower, index, marker="|", s=260, color="#222222", linewidth=2)
            axis.scatter(margin, index, marker="x", s=75, color="#222222", linewidth=2)
        axis.set_yticks(y, [item[0] for item in values])
        axis.set_xlabel("Mức cải thiện tuyệt đối")
        axis.set_title("Đánh giá đồng thời an toàn và tiện ích của cấu hình kết hợp")
        axis.grid(axis="x", alpha=0.22)
        axis.invert_yaxis()
        axis.text(
            0.01,
            -0.26,
            "Hình thoi: ước lượng  |  Vạch đứng: cận dưới một phía 95%  |  "
            "Dấu chéo: ngưỡng quyết định",
            transform=axis.transAxes,
            fontsize=9,
        )
        return figure

    return FigureAsset(
        number=7,
        slug="quyet_dinh_an_toan_tien_ich",
        title="Đánh giá đồng thời an toàn và tiện ích của cấu hình kết hợp",
        caption=(
            "Hình 7. Hai thành phần của quyết định đồng thời khi so sánh cấu hình kết hợp với "
            "cấu hình chỉ dùng bộ phát hiện. Kết luận chung chỉ được xem xét khi cận dưới của "
            f"cả hai thành phần vượt ngưỡng tương ứng. Cỡ mẫu: {sample_size}."
        ),
        recommended_section="Chương kết quả và thảo luận",
        data_source="Phân tích ghép cặp đồng thời trên kết quả thực thi hệ thống cố định",
        unit="Chênh lệch tỷ lệ tuyệt đối",
        sample_size=sample_size,
        headers=(
            "Thành phần",
            "Ước lượng",
            "Cận dưới một phía 95%",
            "Ngưỡng quyết định",
            "Số tình huống ghép cặp",
            "Số nhóm nguồn",
            "Vượt ngưỡng",
        ),
        rows=tuple(values),
        draw=draw,
    )


def _adaptive_gap_spec(artifact: Mapping[str, object]) -> FigureAsset:
    effects = _mapping(artifact.get("paired_guided_minus_static"), "adaptive effects")
    values: list[tuple[str, float, float, float, str]] = []
    for arm in ARM_ORDER[1:]:
        effect = _mapping(effects.get(arm), f"adaptive effect {arm}")
        interval = _mapping(effect.get("confidence_interval_95"), f"adaptive interval {arm}")
        values.append(
            (
                ARM_LABELS[arm],
                _finite(effect.get("effect"), "adaptive effect"),
                _finite(interval.get("lower"), "adaptive lower bound"),
                _finite(interval.get("upper"), "adaptive upper bound"),
                _decision_text(effect.get("hypothesis_decision")),
            )
        )
    _validate_intervals([(item[1], item[2], item[3]) for item in values])
    base_count = _positive_int(
        artifact.get("paired_base_episode_count"), "paired base episode count"
    )
    family_count = _positive_int(artifact.get("family_count"), "adaptive family count")
    sample_size = (
        f"{base_count} tình huống gốc thuộc {family_count} nhóm; "
        "10 biến thể mỗi chiến lược"
    )

    def draw() -> Figure:
        figure = _new_figure((9.2, 5.0))
        axis = figure.add_subplot(111)
        y = np.arange(len(values))
        axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
        for index, (_, estimate, lower, upper, _) in enumerate(values):
            axis.hlines(index, lower, upper, color=PALETTE[index], linewidth=3)
            axis.scatter(estimate, index, s=65, color=PALETTE[index], zorder=3)
        axis.set_yticks(y, [item[0] for item in values])
        axis.set_xlabel("Chênh lệch xác suất tìm thấy thất bại: có phản hồi trừ lấy mẫu tĩnh")
        axis.set_title("Khoảng cách giữa hai chiến lược tìm kiếm tấn công")
        axis.grid(axis="x", alpha=0.22)
        axis.invert_yaxis()
        return figure

    return FigureAsset(
        number=8,
        slug="chenh_lech_tim_kiem_tan_cong",
        title="Khoảng cách giữa hai chiến lược tìm kiếm tấn công",
        caption=(
            "Hình 8. Chênh lệch xác suất tìm thấy ít nhất một thất bại trong cùng ngân sách "
            "mười biến thể giữa tìm kiếm có phản hồi và lấy mẫu tĩnh. Giá trị dương nghiêng về "
            f"tìm kiếm có phản hồi; đoạn ngang là khoảng tin cậy 95%. {sample_size}."
        ),
        recommended_section="Chương kết quả thực nghiệm",
        data_source="Phân tích ghép cặp theo tình huống gốc của tìm kiếm tấn công thích nghi",
        unit="Chênh lệch xác suất",
        sample_size=sample_size,
        headers=(
            "Cấu hình bảo vệ",
            "Ước lượng",
            "Cận dưới 95%",
            "Cận trên 95%",
            "Diễn giải quyết định",
        ),
        rows=tuple(values),
        draw=draw,
    )


def _calibration_spec(artifact: Mapping[str, object]) -> FigureAsset:
    order = _string_sequence(artifact.get("formal_condition_order"), "condition order")
    comparisons = _mapping(artifact.get("comparisons"), "diagnostic comparisons")
    values: list[tuple[str, float, float, float, float, float, float]] = []
    for condition in order:
        calibration = _mapping(
            _mapping(comparisons.get(condition), f"comparison {condition}").get(
                "calibration_degradation"
            ),
            f"calibration {condition}",
        )
        intervals = _mapping(calibration.get("confidence_interval_95"), "calibration intervals")
        brier_interval = _mapping(intervals.get("brier"), "brier interval")
        ece_interval = _mapping(intervals.get("ece_10_bins"), "calibration error interval")
        values.append(
            (
                CONDITION_LABELS[condition],
                _finite(calibration.get("brier_estimate"), "brier estimate"),
                _finite(brier_interval.get("lower_95"), "brier lower"),
                _finite(brier_interval.get("upper_95"), "brier upper"),
                _finite(calibration.get("ece_10_bins_estimate"), "calibration error estimate"),
                _finite(ece_interval.get("lower_95"), "calibration error lower"),
                _finite(ece_interval.get("upper_95"), "calibration error upper"),
            )
        )
    _validate_intervals([(item[1], item[2], item[3]) for item in values])
    _validate_intervals([(item[4], item[5], item[6]) for item in values])
    family_count = _positive_int(
        _mapping(artifact.get("source_family_paired_reference"), "paired reference").get(
            "expected_source_family_count"
        ),
        "source group count",
    )

    def draw() -> Figure:
        figure = _new_figure((12.0, 5.8))
        panels = (
            ("Mức tăng sai số Brier", 1, 2, 3),
            ("Mức tăng sai số hiệu chỉnh 10 khoảng", 4, 5, 6),
        )
        for panel_index, (title, estimate_index, lower_index, upper_index) in enumerate(
            panels, start=1
        ):
            axis = figure.add_subplot(1, 2, panel_index)
            y = np.arange(len(values))
            axis.axvline(0, color="#444444", linewidth=1, linestyle="--")
            for index, row in enumerate(values):
                axis.hlines(
                    index,
                    row[lower_index],
                    row[upper_index],
                    color=PALETTE[index],
                    linewidth=3,
                )
                axis.scatter(row[estimate_index], index, color=PALETTE[index], s=55, zorder=3)
            axis.set_yticks(y, [row[0] for row in values])
            axis.set_title(title)
            axis.set_xlabel("Chênh lệch so với điều kiện chuẩn")
            axis.grid(axis="x", alpha=0.22)
            axis.invert_yaxis()
        figure.suptitle("Suy giảm chất lượng hiệu chỉnh theo điều kiện gây nhiễu", fontsize=14)
        return figure

    return FigureAsset(
        number=9,
        slug="suy_giam_hieu_chinh",
        title="Suy giảm chất lượng hiệu chỉnh theo điều kiện gây nhiễu",
        caption=(
            "Hình 9. Mức tăng của hai sai số hiệu chỉnh trong từng điều kiện gây nhiễu so với "
            "điều kiện chuẩn. Giá trị dương biểu thị hiệu chỉnh kém hơn; đoạn ngang là khoảng "
            f"tin cậy 95%. Cỡ mẫu gồm {family_count} nhóm nguồn."
        ),
        recommended_section="Phụ lục phân tích chẩn đoán",
        data_source="Phân tích hiệu chỉnh được tính lại trong từng mẫu bootstrap",
        unit="Chênh lệch sai số",
        sample_size=f"{family_count} nhóm nguồn ghép cặp",
        headers=(
            "Điều kiện",
            "Chênh lệch sai số Brier",
            "Cận dưới Brier 95%",
            "Cận trên Brier 95%",
            "Chênh lệch sai số hiệu chỉnh",
            "Cận dưới hiệu chỉnh 95%",
            "Cận trên hiệu chỉnh 95%",
        ),
        rows=tuple(values),
        draw=draw,
    )


def _failure_taxonomy_spec(artifact: Mapping[str, object]) -> FigureAsset:
    taxonomy = _mapping(artifact.get("unique_failure_taxonomy"), "failure taxonomy")
    columns: list[tuple[str, str, str]] = []
    for strategy, strategy_label in (
        ("static_sampling", "Lấy mẫu tĩnh"),
        ("feedback_guided", "Có phản hồi"),
    ):
        strategy_data = _mapping(taxonomy.get(strategy), f"taxonomy {strategy}")
        for arm in ARM_ORDER[1:]:
            _mapping(strategy_data.get(arm), f"taxonomy {strategy} {arm}")
            columns.append((strategy, arm, f"{strategy_label}\n{ARM_LABELS[arm]}"))
    codes = tuple(VIOLATION_LABELS)
    matrix = np.zeros((len(codes), len(columns)), dtype=int)
    for column_index, (strategy, arm, _) in enumerate(columns):
        counts = _mapping(
            _mapping(_mapping(taxonomy.get(strategy), strategy).get(arm), arm).get(
                "violation_code_counts"
            ),
            "violation code counts",
        )
        unknown = set(counts).difference(VIOLATION_LABELS)
        if unknown:
            raise ValueError(f"unmapped violation codes: {sorted(unknown)}")
        for row_index, code in enumerate(codes):
            value = counts.get(code, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("violation count must be a non-negative integer")
            matrix[row_index, column_index] = value
    base_count = _positive_int(
        artifact.get("paired_base_episode_count"), "paired base episode count"
    )
    family_count = _positive_int(artifact.get("family_count"), "adaptive family count")
    candidate_count = base_count * 2 * 10

    def draw() -> Figure:
        figure = _new_figure((13.5, 8.0))
        axis = figure.add_subplot(111)
        image = axis.imshow(matrix, cmap="Blues", aspect="auto")
        axis.set_xticks(
            np.arange(len(columns)),
            [item[2] for item in columns],
            rotation=25,
            ha="right",
        )
        axis.set_yticks(np.arange(len(codes)), [VIOLATION_LABELS[code] for code in codes])
        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = int(matrix[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold and value > 0 else "#222222",
                    fontsize=8,
                )
        axis.set_title("Phân bố loại vi phạm theo chiến lược và cấu hình bảo vệ")
        figure.colorbar(image, ax=axis, label="Số lần ghi nhận")
        return figure

    table_rows = tuple(
        (
            VIOLATION_LABELS[code],
            columns[column_index][2].replace("\n", " — "),
            int(matrix[row_index, column_index]),
        )
        for row_index, code in enumerate(codes)
        for column_index in range(len(columns))
    )
    return FigureAsset(
        number=10,
        slug="phan_loai_vi_pham",
        title="Phân bố loại vi phạm theo chiến lược và cấu hình bảo vệ",
        caption=(
            "Hình 10. Số lần ghi nhận từng loại vi phạm trong các kết quả tấn công thành công, "
            "phân tách theo chiến lược tìm kiếm và cấu hình bảo vệ. Bảng màu chỉ mô tả tần suất "
            f"quan sát, không tự nó chứng minh nguyên nhân. Dữ liệu gồm {candidate_count} biến thể "
            f"từ {base_count} tình huống gốc thuộc {family_count} nhóm."
        ),
        recommended_section="Phụ lục phân tích lỗi",
        data_source="Phân loại vi phạm từ kết quả đánh giá tấn công thích nghi đã hậu kiểm",
        unit="Số lần ghi nhận",
        sample_size=f"{candidate_count} biến thể từ {base_count} tình huống gốc",
        headers=("Loại vi phạm", "Chiến lược và cấu hình", "Số lần ghi nhận"),
        rows=table_rows,
        draw=draw,
    )


def _runtime_spec(artifact: Mapping[str, object]) -> FigureAsset:
    if artifact.get("validation_status") != "PASS" or artifact.get("hardware_observed") is not True:
        raise ValueError("runtime telemetry is not an observed passing ledger")
    records = artifact.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("runtime telemetry records are missing")
    groups: dict[str, list[float]] = {}
    table_rows: list[tuple[str, str, float, bool]] = []
    for raw in records:
        record = _mapping(raw, "runtime record")
        stage_id = record.get("stage_id")
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError("runtime stage identifier is missing")
        elapsed_minutes = _finite(record.get("elapsed_seconds"), "elapsed seconds") / 60
        if elapsed_minutes <= 0:
            raise ValueError("runtime interval must be positive")
        accelerated = _boolean(record.get("accelerator_stage"), "accelerator stage")
        group = _runtime_group(stage_id)
        totals = groups.setdefault(group, [0.0, 0.0])
        totals[0 if accelerated else 1] += elapsed_minutes
        table_rows.append(
            (
                group,
                "Tăng tốc phần cứng" if accelerated else "Điều phối hoặc phân tích",
                elapsed_minutes,
                record.get("status") == "completed",
            )
        )
    interval_count = _positive_int(artifact.get("unique_interval_count"), "runtime interval count")
    compute_hours = _finite(artifact.get("compute_hours"), "accelerator compute hours")
    if compute_hours <= 0:
        raise ValueError("accelerator compute hours must be positive")
    ordered_groups = tuple(groups)

    def draw() -> Figure:
        figure = _new_figure((10.5, 6.0))
        axis = figure.add_subplot(111)
        y = np.arange(len(ordered_groups))
        accelerated = np.asarray([groups[group][0] for group in ordered_groups])
        host = np.asarray([groups[group][1] for group in ordered_groups])
        axis.barh(y, accelerated, color=PALETTE[0], label="Tăng tốc phần cứng")
        axis.barh(y, host, left=accelerated, color=PALETTE[1], label="Điều phối hoặc phân tích")
        axis.set_yticks(y, ordered_groups)
        axis.set_xlabel("Thời gian quan sát (phút)")
        axis.set_title("Phân bổ thời gian thực thi theo nhóm công việc")
        axis.grid(axis="x", alpha=0.22)
        axis.legend(loc="best")
        axis.invert_yaxis()
        return figure

    return FigureAsset(
        number=11,
        slug="phan_bo_thoi_gian_thuc_thi",
        title="Phân bổ thời gian thực thi theo nhóm công việc",
        caption=(
            "Hình 11. Tổng thời gian quan sát của các khoảng thực thi, gộp theo nhóm công việc "
            "và tách phần được ghi nhận là tăng tốc phần cứng khỏi phần điều phối hoặc phân tích. "
            f"Sổ đo gồm {interval_count} khoảng duy nhất; tổng giờ tăng tốc quan sát được là "
            f"{compute_hours:.3f} giờ."
        ),
        recommended_section="Phụ lục tài nguyên và khả năng tái lập",
        data_source="Sổ đo thời gian thực thi đã liên kết với biên nhận phần cứng",
        unit="Phút",
        sample_size=f"{interval_count} khoảng thực thi duy nhất",
        headers=("Nhóm công việc", "Loại thời gian", "Số phút", "Hoàn thành"),
        rows=tuple(table_rows),
        draw=draw,
    )


def _new_figure(size: tuple[float, float]) -> Figure:
    figure = Figure(figsize=size, dpi=100, constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure


def _draw_flow(
    axis: Any,
    nodes: Sequence[tuple[str, float, float]],
    edges: Sequence[tuple[int, int]],
    title: str,
) -> None:
    width = 0.20
    height = 0.19
    for index, (label, x, y) in enumerate(nodes):
        color = PALETTE[index % len(PALETTE)]
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=color,
            facecolor="#F7FAFC",
        )
        axis.add_patch(box)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center")
    for source_index, target_index in edges:
        _, source_x, source_y = nodes[source_index]
        _, target_x, target_y = nodes[target_index]
        source_center = (source_x + width / 2, source_y + height / 2)
        target_center = (target_x + width / 2, target_y + height / 2)
        dx = target_center[0] - source_center[0]
        dy = target_center[1] - source_center[1]
        norm = math.hypot(dx / width, dy / height)
        start = (
            source_center[0] + dx / max(norm, 1) * 0.52,
            source_center[1] + dy / max(norm, 1) * 0.52,
        )
        end = (
            target_center[0] - dx / max(norm, 1) * 0.52,
            target_center[1] - dy / max(norm, 1) * 0.52,
        )
        axis.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#333333")
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(title, pad=16)
    axis.axis("off")


def _save_figure(figure: Figure, stem: Path) -> list[str]:
    outputs: list[str] = []
    metadata: dict[str, dict[str, object]] = {
        "png": {"Software": "ViPIBench"},
        "svg": {"Creator": "ViPIBench", "Date": None},
        "pdf": {
            "Creator": "ViPIBench",
            "Producer": "ViPIBench",
            "CreationDate": None,
            "ModDate": None,
        },
    }
    for suffix in ("png", "svg", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(
            path,
            format=suffix,
            dpi=300 if suffix == "png" else None,
            metadata=metadata[suffix],
            bbox_inches="tight",
        )
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"figure output was not created: {path.name}")
        outputs.append(path.name)
    return outputs


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not headers or not rows:
        raise ValueError("figure data table cannot be empty")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def _artifact(
    artifacts: Mapping[str, Mapping[str, object]], name: str
) -> Mapping[str, object]:
    artifact = artifacts.get(name)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"required report artifact is missing: {name}")
    return artifact


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return dict(value)


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list | tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty string sequence")
    return tuple(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _probability(value: object, label: str) -> float:
    result = _finite(value, label)
    if not 0 <= result <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _validate_intervals(values: Sequence[tuple[float, float, float]]) -> None:
    for _estimate, lower, upper in values:
        if lower > upper:
            raise ValueError("reported confidence interval bounds are inconsistent")


def _paired_sample_summary(values: Sequence[tuple[str, float, float, float, int, int]]) -> str:
    pair_counts = sorted({item[4] for item in values})
    family_counts = sorted({item[5] for item in values})
    pairs = ", ".join(str(value) for value in pair_counts)
    families = ", ".join(str(value) for value in family_counts)
    return f"{pairs} cặp đối sánh thuộc {families} nhóm nguồn"


def _decision_text(value: object) -> str:
    mapping = {
        "SUPPORTED": "Được dữ liệu ủng hộ trong thiết kế đã khóa",
        "INCONCLUSIVE_SENSITIVITY_DISAGREEMENT": (
            "Chưa kết luận do phân tích độ nhạy không đồng thuận"
        ),
        "NOT_SUPPORTED_WITHIN_REGISTERED_BUDGET": (
            "Chưa được dữ liệu ủng hộ trong ngân sách đã khóa"
        ),
    }
    if not isinstance(value, str) or value not in mapping:
        raise ValueError("adaptive decision is not recognized")
    return mapping[value]


def _runtime_group(stage_id: str) -> str:
    normalized = stage_id.lower()
    preparation_tokens = ("preflight", "capacity", "compile", "audit-provenance")
    if any(token in normalized for token in preparation_tokens):
        return "Tiền kiểm và chuẩn bị dữ liệu"
    if any(token in normalized for token in ("encoder", "baseline", "detector")):
        return "Huấn luyện và đánh giá bộ phát hiện"
    if any(token in normalized for token in ("attack", "adaptive", "candidate")):
        return "Tìm kiếm và đánh giá tấn công"
    if any(token in normalized for token in ("target", "four-arm", "static-system")):
        return "Thực thi hệ thống cố định"
    return "Phân tích, hậu kiểm và đóng gói"

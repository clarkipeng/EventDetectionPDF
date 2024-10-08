"""Event Detection Average Precision

An average precision metric for event detection in time series and
video.

"""

from bisect import bisect_left
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    pass


# Set some placeholders for global parameters
series_id_column_name = "series_id"
time_column_name = "step"
event_column_name = "event"
score_column_name = "score"
use_scoring_intervals = False
tolerances = {
    "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}

def find_nearest_time_idx(times, target_time, excluded_indices, tolerance):
    """Find the index of the nearest time to the target_time
    that is not in excluded_indices."""
    idx = bisect_left(times, target_time)

    best_idx = None
    best_error = float("inf")

    offset_range = min(len(times), tolerance)
    for offset in range(
        -offset_range, offset_range
    ):  # Check the exact, one before, and one after
        check_idx = idx + offset
        if 0 <= check_idx < len(times) and check_idx not in excluded_indices:
            error = abs(times[check_idx] - target_time)
            if error < best_error:
                best_error = error
                best_idx = check_idx

    return best_idx, best_error


def match_detections(
    tolerance: float, ground_truths: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    detections_sorted = detections.sort_values(
        score_column_name, ascending=False
    ).dropna()
    is_matched = np.full_like(detections_sorted[event_column_name], False, dtype=bool)
    ground_truths_times = ground_truths.sort_values(time_column_name)[
        time_column_name
    ].tolist()
    matched_gt_indices: set[int] = set()

    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        det_time = getattr(det, time_column_name)

        best_idx, best_error = find_nearest_time_idx(
            ground_truths_times, det_time, matched_gt_indices, tolerance
        )

        if best_idx is not None and best_error < tolerance:
            is_matched[i] = True
            matched_gt_indices.add(best_idx)

    detections_sorted["matched"] = is_matched
    return detections_sorted


def precision_recall_curve(
    matches: np.ndarray, scores: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []  # type: ignore

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind="stable")[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = (
        tps / p
    )  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def aggregate_data_by_series_event(df, step_range):
    """
    :param df: DataFrame to aggregate.
    :param step_range: The range within which steps are considered close.
    :return: A single DataFrame with aggregated rows.
    """

    aggregated_data = []

    # Sorting the DataFrame by 'series_id' and 'step'
    sorted_df = df.sort_values(by=["series_id", "step"])

    for (series_id, event), group in sorted_df.groupby(["series_id", "event"]):
        start_step = group["step"].iloc[0]
        current_segment = []

        for _, row in group.iterrows():
            if row["step"] - start_step <= step_range:
                current_segment.append(row)
            else:
                if current_segment:
                    segment_df = pd.DataFrame(current_segment)
                    average_step = segment_df["step"].mean()
                    average_score = segment_df["score"].mean()
                    aggregated_data.append(
                        {
                            "series_id": series_id,
                            "event": event,
                            "step": average_step,
                            "score": average_score,
                        }
                    )
                start_step = row["step"]
                current_segment = [row]

        # Aggregating the last segment for each series_id and event
        if current_segment:
            segment_df = pd.DataFrame(current_segment)
            average_step = segment_df["step"].mean()
            average_score = segment_df["score"].mean()
            aggregated_data.append(
                {
                    "series_id": series_id,
                    "event": event,
                    "step": average_step,
                    "score": average_score,
                }
            )

    # Creating the final DataFrame, converting 'step' to int64, and adding 'row_id'
    final_df = pd.DataFrame(aggregated_data)
    final_df["step"] = final_df["step"].astype("int64")
    final_df = final_df.sort_values(by=["series_id", "step"]).reset_index(drop=True)
    final_df.reset_index(inplace=True)
    final_df.rename(columns={"index": "row_id"}, inplace=True)

    # Reordering the columns to match the desired structure
    final_df = final_df[["row_id", "series_id", "step", "event", "score"]]

    return final_df


"""Event Detection F1 score
An f1 metric for event detection in time series and video.
"""

def calculate_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]],
    series_id_column_name: str,
    time_column_name: str,
    event_column_name: str,
    score_column_name: str,
    metrics: List[str] = ['mAP'], 
    use_scoring_intervals: bool = False,
) -> float:
    # Validate metric parameters
    assert len(tolerances) > 0, "Events must have defined tolerances."
    #     print(set(solution[event_column_name]).difference(
    #         {"start", "end"}
    #     ))
    assert set(tolerances.keys()) == set(solution[event_column_name]).difference(
        {"start", "end"}
    ), (
        f"Solution column {event_column_name} must contain the same events "
        "as defined in tolerances."
    )
    assert pd.api.types.is_numeric_dtype(
        solution[time_column_name]
    ), f"Solution column {time_column_name} must be of numeric type."

    # Validate submission format
    for column_name in [
        series_id_column_name,
        time_column_name,
        event_column_name,
        score_column_name,
    ]:
        if column_name not in submission.columns:
            raise ParticipantVisibleError(
                f"Submission must have column '{column_name}'."
            )

    if not pd.api.types.is_numeric_dtype(submission[time_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{time_column_name}' must be of numeric type."
        )
    if not pd.api.types.is_numeric_dtype(submission[score_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{score_column_name}' must be of numeric type."
        )

    # Set these globally to avoid passing around a bunch of arguments
    globals()["series_id_column_name"] = series_id_column_name
    globals()["time_column_name"] = time_column_name
    globals()["event_column_name"] = event_column_name
    globals()["score_column_name"] = score_column_name
    globals()["use_scoring_intervals"] = use_scoring_intervals

    return event_detection_all(metrics, solution, submission, tolerances)


def event_detection_all(
    metrics,
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]],  # type: ignore
) -> float:
    # Ensure solution and submission are sorted properly
    solution = solution.sort_values([series_id_column_name, time_column_name])
    submission = submission.sort_values([series_id_column_name, time_column_name])

    # Extract scoring intervals.
    if use_scoring_intervals:
        # intervals = (
        #     solution.query("event in ['start', 'end']")
        #     .assign(
        #       interval=lambda x: x.groupby([series_id_column_name, event_column_name]).cumcount()
        #     )
        #     .pivot(
        #         index="interval",
        #         columns=[series_id_column_name, event_column_name],
        #         values=time_column_name,
        #     )
        #     .stack(series_id_column_name)
        #     .swaplevel()
        #     .sort_index()
        #     .loc[:, ["start", "end"]]
        #     .apply(lambda x: pd.Interval(*x, closed="both"), axis=1)
        # )
        pass

    # Extract ground-truth events.
    ground_truths = solution.query("event not in ['start', 'end']").reset_index(
        drop=True
    )

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts(event_column_name).to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched=False)

    # Remove detections outside of scoring intervals
    if use_scoring_intervals:
        # detections_filtered = []
        # for (det_group, dets), (int_group, ints) in zip(
        #     detections.groupby(series_id_column_name), intervals.groupby(series_id_column_name)
        # ):
        #     assert det_group == int_group
        #     detections_filtered.append(filter_detections(dets, ints))  # noqa: F821
        # detections_filtered = pd.concat(detections_filtered, ignore_index=True)
        pass
    else:
        detections_filtered = detections

    # Create table of event-class x tolerance x series_id values
    aggregation_keys = pd.DataFrame(
        [
            (ev, tol, vid)
            for ev in tolerances.keys()
            for tol in tolerances[ev]
            for vid in ground_truths[series_id_column_name].unique()
        ],
        columns=[event_column_name, "tolerance", series_id_column_name],
    )

    # Create match evaluation groups: event-class x tolerance x series_id
    detections_grouped = aggregation_keys.merge(
        detections_filtered, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])
    ground_truths_grouped = aggregation_keys.merge(
        ground_truths, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, "tolerance", series_id_column_name])

    # Match detections to ground truth events by evaluation group
    detections_matched = []
    for key in aggregation_keys.itertuples(index=False):
        dets = detections_grouped.get_group(key)
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(
            match_detections(dets["tolerance"].iloc[0], gts, dets)
        )
    detections_matched = pd.concat(detections_matched)

    # Compute score per event x tolerance group
    event_classes = ground_truths[event_column_name].unique()
    scores = {}
    
    if 'mf1' in metrics:
        f1_table = (
            detections_matched.query("event in @event_classes")  # type: ignore
            .groupby([event_column_name, "tolerance"])
            .apply(
                lambda group: f1_score(
                    group["matched"].to_numpy(),
                    group[score_column_name].to_numpy(),
                    class_counts[group[event_column_name].iat[0]],
                )
            )
        )
        # Average over tolerances, then over event classes
        mean_f1 = f1_table.groupby(event_column_name).mean().sum() / len(event_classes)
        scores["mf1"] =  mean_ap
        
        scores["mf1_tolerances"] = ap_table.groupby('tolerance').mean().values

    if 'mAP' in metrics:
        ap_table = (
            detections_matched.query("event in @event_classes")  # type: ignore
            .groupby([event_column_name, "tolerance"])
            .apply(
                lambda group: average_precision_score(
                    group["matched"].to_numpy(),
                    group[score_column_name].to_numpy(),
                    class_counts[group[event_column_name].iat[0]],
                )
            )
        )
        # Average over tolerances, then over event classes
        mean_ap = ap_table.groupby(event_column_name).mean().sum() / len(event_classes)
        scores["mAP"] =  mean_ap

        scores["mAP_tolerances"] = ap_table.groupby('tolerance').mean().values
    return scores


def f1_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = np.argmax(f1)
    return f1[idx]  # , precision[idx], recall[idx]


# def eventdetector_f1score(s_peaks, s_peaks_score, e_t, delta_with_time_unit =75):
#     delta_t: list = []

#     for thr in np.linspace(min(s_peaks_score),max(s_peaks_score),10):
#         peaks = s_peaks[s_peaks_score>thr]

#         e_matched = np.zeros(e_t.shape)
#         p_matched = np.zeros(s_peaks.shape)

#         for i, m_p in enumerate(s_peaks):
#             signed_delta = delta_with_time_unit

#             closest = None
#             for j, t_e in enumerate(e_t):
#                 m_t = t_e
#                 diff = m_p - m_t

#                 if abs(diff) <= delta_with_time_unit:
#                     if closest is None or abs(diff)< abs(m_p - e_t[closest]):
#                         closest = j

#             if closest:
#                 e_matched[closest] = 1
#                 p_matched[i] = 1
#         fn : int = len(e_matched) - np.sum(e_matched)
#         fp : int = len(s_peaks) - np.sum(p_matched)

#         tp : int = np.sum(e_matched)


#         # return tp, fp, fn, delta_t
#         if tp + fp == 0 or tp + fn == 0:
#             return 0.0, 0.0, 0.0
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#         if precision + recall == 0:
#             return 0.0, 0.0, 0.0
#         return (2.0 * precision * recall) / (precision + recall), precision, recall
# f1, precision, recall = eventdetector_f1score(submission[time_column_name],submission[score_column_name],solution[time_column_name])
# return {"mAP": mean_ap, "mf1": mean_f1, 'simple_f1': f1, 'simple_precision': precision, 'simple_recall': recall}

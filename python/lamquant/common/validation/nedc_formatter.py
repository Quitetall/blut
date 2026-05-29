#!/usr/bin/env python3
"""
NEDC-compatible CSV formatter for seizure detection predictions.

Converts LamQuant model outputs (timestamps + labels) to NEDC CSV format
for direct compatibility with nedc_eeg_eval evaluation toolkit.

NEDC CSV Format:
    channel,start_time,stop_time,label,confidence

    Example:
        TERM,10.5,12.3,seiz,0.95
        TERM,45.2,47.8,seiz,0.87

    Where:
    - channel: "TERM" for terminal (all-channel) annotations
    - start_time: Event start time in seconds
    - stop_time: Event end time in seconds
    - label: Annotation label (e.g., "seiz" for seizure, "bckg" for background)
    - confidence: Model confidence [0.0-1.0]
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class NedcEventFormatter:
    """
    Format seizure detection predictions as NEDC-compatible CSV.

    Usage:
        formatter = NedcEventFormatter()

        # Method 1: From event timestamps
        events = [
            (10.5, 12.3, "seiz", 0.95),  # start, stop, label, confidence
            (45.2, 47.8, "seiz", 0.87),
        ]
        formatter.write_events_csv("predictions.csv", events)

        # Method 2: From continuous predictions
        times = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 1, 0, 0])
        confidences = np.array([0.2, 0.9, 0.85, 0.1, 0.05])
        formatter.write_continuous_predictions(
            "predictions.csv", times, labels, confidences,
            sample_rate=1.0  # 1 sample per second
        )
    """

    def __init__(self, label_map: Optional[Dict[int, str]] = None):
        """
        Initialize formatter.

        Args:
            label_map: Mapping from numeric labels to string labels.
                      Default: {0: 'bckg', 1: 'seiz'}
        """
        if label_map is None:
            self.label_map = {0: 'bckg', 1: 'seiz'}
        else:
            self.label_map = label_map

    def write_events_csv(self, filepath: str,
                        events: List[Tuple[float, float, str, float]],
                        channel: str = 'TERM') -> None:
        """
        Write events to NEDC-format CSV file.

        Args:
            filepath: Output CSV file path
            events: List of (start_time, stop_time, label, confidence) tuples
            channel: Channel identifier (default 'TERM' for terminal/all-channel)
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            for start_time, stop_time, label, confidence in events:
                # Ensure label is a string
                if isinstance(label, int):
                    label = self.label_map.get(label, f'label_{label}')

                writer.writerow([
                    channel,
                    f'{start_time:.1f}',
                    f'{stop_time:.1f}',
                    label,
                    f'{confidence:.4f}'
                ])

    def write_continuous_predictions(
            self, filepath: str,
            sample_times: np.ndarray,
            predictions: np.ndarray,
            confidences: Optional[np.ndarray] = None,
            sample_rate: float = 1.0,
            min_confidence: float = 0.5,
            channel: str = 'TERM') -> None:
        """
        Convert continuous predictions to event-based CSV.

        Segments continuous predictions into events (contiguous regions with
        same label above threshold).

        Args:
            filepath: Output CSV file path
            sample_times: Array of sample timestamps (seconds)
            predictions: Array of predicted labels (0/1 or class indices)
            confidences: Array of confidence scores [0-1] (optional)
            sample_rate: Samples per second (for computing segment durations)
            min_confidence: Only include predictions with confidence >= this
            channel: Channel identifier (default 'TERM')
        """
        # Convert predictions to binary if needed
        if predictions.max() > 1:
            predictions = (predictions > 0.5).astype(int)

        # Default: flat confidences if not provided
        if confidences is None:
            confidences = np.ones_like(predictions, dtype=float)

        # Filter by confidence threshold
        mask = confidences >= min_confidence
        predictions = predictions[mask]
        sample_times = sample_times[mask]
        confidences = confidences[mask]

        # Segment into events
        events = self._segment_predictions(
            sample_times, predictions, confidences, sample_rate
        )

        # Write to CSV
        self.write_events_csv(filepath, events, channel=channel)

    def _segment_predictions(self, times: np.ndarray, labels: np.ndarray,
                            confidences: np.ndarray,
                            sample_rate: float
                            ) -> List[Tuple[float, float, str, float]]:
        """
        Segment continuous predictions into events.

        Creates an event for each contiguous region with the same label.

        Args:
            times: Sample times in seconds
            labels: Predicted labels (0, 1, or other)
            confidences: Confidence scores
            sample_rate: Samples per second

        Returns:
            List of (start_time, stop_time, label, avg_confidence) tuples
        """
        events = []

        if len(labels) == 0:
            return events

        # Track current segment
        current_label = labels[0]
        segment_start_idx = 0
        segment_confs = [confidences[0]]

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                # Segment ended, record it
                start_time = times[segment_start_idx]
                # End time is start + duration
                duration = (i - segment_start_idx) / sample_rate
                stop_time = start_time + duration

                avg_conf = np.mean(segment_confs)
                label_str = self.label_map.get(current_label,
                                              f'label_{current_label}')

                # Only record seizure events (non-background)
                if current_label != 0:
                    events.append((start_time, stop_time, label_str, avg_conf))

                # Start new segment
                current_label = labels[i]
                segment_start_idx = i
                segment_confs = [confidences[i]]
            else:
                segment_confs.append(confidences[i])

        # Record final segment
        if len(segment_confs) > 0 and current_label != 0:
            start_time = times[segment_start_idx]
            duration = (len(labels) - segment_start_idx) / sample_rate
            stop_time = start_time + duration
            avg_conf = np.mean(segment_confs)
            label_str = self.label_map.get(current_label,
                                          f'label_{current_label}')
            events.append((start_time, stop_time, label_str, avg_conf))

        return events

    def read_events_csv(self, filepath: str) -> List[Dict]:
        """
        Read NEDC-format CSV file (.csv or .csv_bi).

        Handles real NEDC csv_bi files which contain:
          - '# key = value' comment lines (metadata: version, bname, duration, ...)
          - a 'channel,start_time,stop_time,label,confidence' column header
          - data rows: 'TERM,0.0,49.0,bckg,1.0'

        Returns:
            List of dicts with keys: channel, start_time, stop_time, label, confidence
        """
        events = []

        with open(filepath, 'r') as f:
            # Strip comment lines and the column header row
            data_lines = []
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                # Skip the column header if present
                if line.lower().startswith('channel,'):
                    continue
                data_lines.append(line)

            reader = csv.DictReader(
                data_lines,
                fieldnames=['channel', 'start_time',
                            'stop_time', 'label', 'confidence']
            )
            for row in reader:
                row['start_time'] = float(row['start_time'])
                row['stop_time'] = float(row['stop_time'])
                row['confidence'] = float(row['confidence'])
                events.append(row)

        return events

    def merge_annotations(self, ref_file: str, hyp_file: str,
                         output_file: str) -> None:
        """
        Merge reference and hypothesis annotations into a single file.

        Useful for creating side-by-side comparison files.

        Args:
            ref_file: Reference (ground truth) CSV
            hyp_file: Hypothesis (predicted) CSV
            output_file: Output file (with _ref and _hyp suffixes added)
        """
        ref_events = self.read_events_csv(ref_file)
        hyp_events = self.read_events_csv(hyp_file)

        # Write both side-by-side
        output_path = Path(output_file)

        # Reference events
        with open(output_path.parent / (output_path.stem + '_ref.csv'), 'w') as f:
            writer = csv.writer(f)
            for event in ref_events:
                writer.writerow([
                    event['channel'],
                    f"{event['start_time']:.1f}",
                    f"{event['stop_time']:.1f}",
                    event['label'],
                    f"{event['confidence']:.4f}"
                ])

        # Hypothesis events
        with open(output_path.parent / (output_path.stem + '_hyp.csv'), 'w') as f:
            writer = csv.writer(f)
            for event in hyp_events:
                writer.writerow([
                    event['channel'],
                    f"{event['start_time']:.1f}",
                    f"{event['stop_time']:.1f}",
                    event['label'],
                    f"{event['confidence']:.4f}"
                ])


def format_predictions_for_nedc(output_dir: str, predictions_dict: Dict) -> str:
    """
    Convenience function: format model predictions for NEDC evaluation.

    Args:
        output_dir: Directory to write CSV files
        predictions_dict: Dictionary with keys:
            - 'sample_times': np.array of time points (seconds)
            - 'labels': np.array of predicted labels
            - 'confidences': np.array of confidence scores

    Returns:
        Path to written CSV file
    """
    formatter = NedcEventFormatter()

    output_path = Path(output_dir) / 'predictions_nedc.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    formatter.write_continuous_predictions(
        str(output_path),
        sample_times=predictions_dict['sample_times'],
        predictions=predictions_dict['labels'],
        confidences=predictions_dict.get('confidences', None),
        sample_rate=predictions_dict.get('sample_rate', 1.0),
        min_confidence=predictions_dict.get('min_confidence', 0.5)
    )

    return str(output_path)

"""Federated decoder learning — post-launch feature.

Each deployment site trains a local decoder fine-tune on patient-specific
data, then uploads model deltas to the central server. The server
aggregates deltas via FedAvg and pushes improved global weights.

This preserves patient privacy: raw EEG never leaves the site. Only
model weight updates are transmitted (differential privacy noise optional).

Architecture:
    Central server: hosts global decoder checkpoint
    Sites: each runs local fine-tuning on patient data
    Protocol: FedAvg with optional DP-SGD noise injection

Usage:
    # Server side
    server = FederatedServer(global_decoder_path)
    server.receive_update(site_id, delta_state_dict)
    server.aggregate()  # FedAvg
    server.broadcast()  # push updated weights

    # Client side
    client = FederatedClient(local_decoder, server_url)
    client.fine_tune(local_data, epochs=10)
    client.upload_delta()
"""

import torch
import copy
from typing import Dict, List, Optional


class FederatedServer:
    """Central aggregation server for federated decoder learning."""

    def __init__(self, global_model_state: dict):
        self.global_state = copy.deepcopy(global_model_state)
        self.pending_deltas: List[dict] = []
        self.round = 0

    def receive_update(self, site_id: str, delta: dict):
        """Receive a weight delta from a client site."""
        self.pending_deltas.append({
            'site_id': site_id,
            'delta': delta,
        })

    def aggregate(self, min_clients: int = 2) -> bool:
        """FedAvg: average all pending deltas into global model.

        Returns True if aggregation happened, False if not enough clients.
        """
        if len(self.pending_deltas) < min_clients:
            return False

        n = len(self.pending_deltas)
        for key in self.global_state:
            if self.global_state[key].dtype.is_floating_point:
                avg_delta = sum(d['delta'].get(key, torch.zeros_like(self.global_state[key]))
                                for d in self.pending_deltas) / n
                self.global_state[key] += avg_delta

        self.pending_deltas.clear()
        self.round += 1
        return True

    def get_global_state(self) -> dict:
        return copy.deepcopy(self.global_state)


class FederatedClient:
    """Client-side federated fine-tuning."""

    def __init__(self, model, dp_noise_scale: float = 0.0):
        self.model = model
        self.dp_noise_scale = dp_noise_scale
        self.pre_finetune_state = None

    def snapshot(self):
        """Save pre-fine-tune state for delta computation."""
        self.pre_finetune_state = copy.deepcopy(self.model.state_dict())

    def compute_delta(self) -> dict:
        """Compute weight delta (post - pre fine-tuning).

        Optionally adds differential privacy noise.
        """
        if self.pre_finetune_state is None:
            raise RuntimeError("Call snapshot() before fine-tuning")

        delta = {}
        current = self.model.state_dict()
        for key in current:
            if current[key].dtype.is_floating_point:
                d = current[key] - self.pre_finetune_state[key]
                if self.dp_noise_scale > 0:
                    # Clip + noise for differential privacy
                    norm = d.norm()
                    clip_bound = 1.0
                    if norm > clip_bound:
                        d = d * clip_bound / norm
                    d = d + torch.randn_like(d) * self.dp_noise_scale
                delta[key] = d

        return delta

    def apply_global_state(self, global_state: dict):
        """Load global weights from server."""
        self.model.load_state_dict(global_state, strict=False)

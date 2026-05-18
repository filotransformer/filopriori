"""
TCP Baselines from Literature

This module provides implementations of state-of-the-art baselines
for Test Case Prioritization comparison.

Available baselines:
    - RETECS: Reinforcement Learning for TCP (Spieker et al., ISSTA 2017)
    - DeepOrder: Deep Learning for TCP (Chen et al., ICSME 2021)
    - NodeRank: Mutation-based Ensemble for TCP (Li et al., IEEE TSE 2024)
    - FailRank-BB: SBERT + LogisticRegression for TCP (Hernandes et al., 2024)
    - Heuristic baselines: Random, FailureRate, Recency, etc.
"""

from .retecs import RETECSPrioritizer, NetworkAgent, TableauAgent
from .deeporder import DeepOrderModel, run_deeporder_experiment
from .tcpnet import TCPNetPrioritizer, TCPNetModel
from .noderank import NodeRankModel, run_noderank_experiment, compare_with_baselines
from .failrank_bb import FailRankBBModel

__all__ = [
    'RETECSPrioritizer',
    'NetworkAgent',
    'TableauAgent',
    'DeepOrderModel',
    'run_deeporder_experiment',
    'TCPNetPrioritizer',
    'TCPNetModel',
    'NodeRankModel',
    'run_noderank_experiment',
    'compare_with_baselines',
    'FailRankBBModel',
]

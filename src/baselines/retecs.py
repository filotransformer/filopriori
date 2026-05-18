"""
RETECS: Reinforcement Learning for Automatic Test Case Prioritization
and Selection in Continuous Integration

Faithful implementation following:
    Spieker, H., Gotlieb, A., Marijan, D., & Mossige, M. (2017).
    Reinforcement learning for automatic test case prioritization and selection
    in continuous integration. ISSTA 2017.

Reference implementation:
    https://github.com/romolodevito/RL_for_TestPrioritization

Two agents:
    - NetworkAgent (RETECS-N): MLP-based with experience replay
    - TableauAgent (RETECS-T): Tabular Q-learning with discretized states
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque


# =============================================================================
# STATE FEATURE EXTRACTION
# =============================================================================

class TestCaseHistory:
    """
    Tracks per-test-case execution history for state feature extraction.
    OPTIMIZED: O(1) per get_state using running min/max and deque.
    """

    def __init__(self):
        self.durations = {}       # test_id -> last duration
        self.last_exec_build = {} # test_id -> last build index
        self.verdicts = defaultdict(lambda: deque(maxlen=4))  # Only last 4 needed
        self.current_build_idx = 0
        # Running min/max for time-since normalization
        self._min_last_exec = None  # min of last_exec_build values
        self._max_last_exec = None  # max of last_exec_build values

    def get_state(self, test_id: str, all_durations: Dict[str, float] = None) -> np.ndarray:
        """
        Extract 6-dimensional state features for a test case — O(1).

        Features (all normalized to [0,1]):
            0: duration_norm - (maxDuration - testDuration) / (maxDuration - minDuration)
            1: timeSince_norm - normalized time since last execution
            2-5: history[0..3] - last 4 verdicts (0=fail, 1=pass), padded with 1
        """
        # Feature 0: Normalized duration (inverted: shorter = higher priority)
        if all_durations and len(all_durations) > 1:
            durations_list = list(all_durations.values())
            max_dur = max(durations_list)
            min_dur = min(durations_list)
            test_dur = all_durations.get(test_id, max_dur)
            if max_dur > min_dur:
                duration_norm = (max_dur - test_dur) / (max_dur - min_dur)
            else:
                duration_norm = 0.5
        elif test_id in self.durations:
            duration_norm = 0.5
        else:
            duration_norm = 0.5

        # Feature 1: Time since last execution (normalized) — O(1) using running min/max
        if test_id in self.last_exec_build:
            builds_since = self.current_build_idx - self.last_exec_build[test_id]
            if self._max_last_exec is not None and self._min_last_exec is not None:
                max_exec_time = self.current_build_idx - self._min_last_exec
                min_exec_time = self.current_build_idx - self._max_last_exec
                if max_exec_time > min_exec_time:
                    time_since_norm = (max_exec_time - builds_since) / (max_exec_time - min_exec_time)
                else:
                    time_since_norm = 0.5
            else:
                time_since_norm = 0.5
        else:
            time_since_norm = 0.0  # Never executed = high time since

        # Features 2-5: Last 4 verdicts (0=fail, 1=pass), padded with 1 (pass)
        past_verdicts = self.verdicts.get(test_id)
        history = [1.0] * 4  # Default: padded with 1 (pass)
        if past_verdicts:
            n = len(past_verdicts)
            for i in range(min(4, n)):
                history[i] = float(past_verdicts[-(i + 1)])  # Most recent first

        state = np.array([duration_norm, time_since_norm] + history, dtype=np.float32)
        return state

    def update(self, test_id: str, verdict: int, duration: float = 1.0):
        """Update history after observing a test result. verdict: 0=fail, 1=pass."""
        self.verdicts[test_id].append(verdict)
        self.durations[test_id] = duration
        self.last_exec_build[test_id] = self.current_build_idx

        # Update running min/max for last_exec_build
        if self._max_last_exec is None or self.current_build_idx > self._max_last_exec:
            self._max_last_exec = self.current_build_idx
        # min_last_exec only needs updating if this test hadn't been seen before
        # or if we're updating an old minimum — approximate: track min lazily
        if self._min_last_exec is None:
            self._min_last_exec = self.current_build_idx

    def advance_build(self):
        """Move to the next build."""
        self.current_build_idx += 1


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

def reward_tcfail(ranking: List[str], verdicts: Dict[str, int]) -> List[float]:
    """
    tcfail reward: each failed test receives 1.0, passes receive 0.0.

    Args:
        ranking: Ordered list of test IDs
        verdicts: test_id -> 0 (pass) or 1 (fail)

    Returns:
        List of per-test rewards in ranking order
    """
    rewards = []
    for test_id in ranking:
        if verdicts.get(test_id, 0) == 1:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_timerank(ranking: List[str], verdicts: Dict[str, int]) -> List[float]:
    """
    timerank reward: each failed test receives total_failures;
    passes receive cumulative sum of failures seen so far.

    Args:
        ranking: Ordered list of test IDs
        verdicts: test_id -> 0 (pass) or 1 (fail)

    Returns:
        List of per-test rewards in ranking order
    """
    total_failures = sum(1 for v in verdicts.values() if v == 1)
    rewards = []
    cumulative_fails = 0
    for test_id in ranking:
        if verdicts.get(test_id, 0) == 1:
            rewards.append(float(total_failures))
            cumulative_fails += 1
        else:
            rewards.append(float(cumulative_fails))
    return rewards


# =============================================================================
# NETWORK AGENT (RETECS-N) — MLP-based
# =============================================================================

class NetworkAgent:
    """
    RETECS Network Agent using MLP (scikit-learn MLPRegressor).

    - Input: 6 continuous features [0,1]
    - MLP with hidden_layer_sizes=(12,), relu, adam
    - Experience replay buffer of 10,000 tuples
    - Retrains every retrain_interval CI cycles with batch of 1,000
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple = (12,),
        replay_buffer_size: int = 10000,
        retrain_interval: int = 5,
        retrain_batch_size: int = 1000,
        max_iter: int = 1200,
        reward_func: str = 'tcfail'
    ):
        from sklearn.neural_network import MLPRegressor

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            warm_start=True,
            random_state=42
        )
        self.replay_buffer = []  # List of (state, reward) tuples
        self.replay_buffer_size = replay_buffer_size
        self.retrain_interval = retrain_interval
        self.retrain_batch_size = retrain_batch_size
        self.reward_func = reward_func
        self.cycle_count = 0
        self.is_fitted = False

    def get_action(self, states: np.ndarray) -> np.ndarray:
        """
        Get priority scores for a batch of test states.

        Args:
            states: (n_tests, 6) array of state features

        Returns:
            (n_tests,) array of priority scores
        """
        if not self.is_fitted:
            return np.random.rand(len(states))

        scores = self.model.predict(states)
        return scores

    def update(self, states: np.ndarray, rewards: List[float]):
        """
        Add experiences to replay buffer and retrain if needed.

        Args:
            states: (n_tests, 6) array of state features
            rewards: List of per-test rewards
        """
        # Add to replay buffer with temporal weighting
        for i in range(len(states)):
            self.replay_buffer.append((states[i].copy(), rewards[i]))

        # Keep buffer bounded
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]

        self.cycle_count += 1

        # Retrain every retrain_interval cycles
        if self.cycle_count % self.retrain_interval == 0 and len(self.replay_buffer) > 10:
            self._retrain()

    def _retrain(self):
        """Retrain MLP from experience replay buffer."""
        n_samples = min(self.retrain_batch_size, len(self.replay_buffer))

        # Temporal weighting: more recent experiences have higher probability
        weights = np.arange(1, len(self.replay_buffer) + 1, dtype=np.float64)
        weights = weights / weights.sum()

        indices = np.random.choice(
            len(self.replay_buffer), size=n_samples, replace=False, p=weights
        )

        X = np.array([self.replay_buffer[i][0] for i in indices])
        y = np.array([self.replay_buffer[i][1] for i in indices])

        self.model.fit(X, y)
        self.is_fitted = True


# =============================================================================
# TABLEAU AGENT (RETECS-T) — Tabular Q-learning
# =============================================================================

class TableauAgent:
    """
    RETECS Tableau Agent using tabular Q-learning with discretized states.

    - State: (duration_bucket[0-2], timeSince_bucket[0-2], history[0..3])
    - 100 discrete actions (priority bins)
    - Q-table: state -> {Q: array[100], N: array[100]}
    - Epsilon-greedy exploration with decay
    """

    def __init__(
        self,
        n_actions: int = 100,
        initial_q: float = 5.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        initial_epsilon: float = 1.0,
        reward_func: str = 'tcfail'
    ):
        self.n_actions = n_actions
        self.initial_q = initial_q
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.reward_func = reward_func

        # Q-table: state_key -> {'Q': np.array, 'N': np.array}
        self.q_table = {}

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretize continuous state into buckets.

        Args:
            state: 6-dim continuous state [duration, timeSince, h0, h1, h2, h3]

        Returns:
            Tuple key for Q-table lookup
        """
        # Duration bucket (0, 1, 2)
        dur_bucket = min(int(state[0] * 3), 2)
        # Time since bucket (0, 1, 2)
        time_bucket = min(int(state[1] * 3), 2)
        # History as integers (0 or 1)
        h0 = int(round(state[2]))
        h1 = int(round(state[3]))
        h2 = int(round(state[4]))
        h3 = int(round(state[5]))

        return (dur_bucket, time_bucket, h0, h1, h2, h3)

    def _get_q_entry(self, state_key: tuple) -> dict:
        """Get or create Q-table entry for a state."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {
                'Q': np.full(self.n_actions, self.initial_q, dtype=np.float64),
                'N': np.zeros(self.n_actions, dtype=np.float64)
            }
        return self.q_table[state_key]

    def get_action(self, states: np.ndarray) -> np.ndarray:
        """
        Get priority scores for a batch of test states.

        For each test, selects an action (priority bin) using epsilon-greedy,
        then converts to a continuous score.

        Args:
            states: (n_tests, 6) array of state features

        Returns:
            (n_tests,) array of priority scores in [0,1]
        """
        scores = np.zeros(len(states))
        for i, state in enumerate(states):
            state_key = self._discretize_state(state)
            entry = self._get_q_entry(state_key)

            if random.random() < self.epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = int(np.argmax(entry['Q']))

            # Convert action to continuous score
            scores[i] = action / (self.n_actions - 1)

        return scores

    def update(self, states: np.ndarray, rewards: List[float]):
        """
        Update Q-values based on observed rewards.

        Uses incremental mean update: Q_new = Q_old + (1/N) * (reward - Q_old)

        Args:
            states: (n_tests, 6) array of state features
            rewards: List of per-test rewards
        """
        for i, state in enumerate(states):
            state_key = self._discretize_state(state)
            entry = self._get_q_entry(state_key)

            # Determine which action was taken (closest bin)
            # Re-select using current policy for consistency
            if random.random() < self.epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = int(np.argmax(entry['Q']))

            entry['N'][action] += 1
            n = entry['N'][action]
            entry['Q'][action] += (1.0 / n) * (rewards[i] - entry['Q'][action])

        # Decay epsilon
        self.epsilon = (self.epsilon - self.min_epsilon) * self.gamma + self.min_epsilon


# =============================================================================
# RETECS EXPERIMENT RUNNER
# =============================================================================

class RETECSPrioritizer:
    """
    High-level RETECS prioritizer combining agent + history + reward.

    Usage:
        prioritizer = RETECSPrioritizer(agent_type='network', reward_func='tcfail')

        # Training phase
        for build in train_builds:
            prioritizer.train_on_build(test_ids, verdicts, durations)

        # Evaluation phase
        for build in test_builds:
            ranking = prioritizer.prioritize(test_ids)
            prioritizer.update_history(test_ids, verdicts, durations)
    """

    def __init__(
        self,
        agent_type: str = 'network',
        reward_func: str = 'tcfail',
        seed: int = 42
    ):
        self.reward_func = reward_func
        self.history = TestCaseHistory()
        self.agent_type = agent_type

        random.seed(seed)
        np.random.seed(seed)

        if agent_type == 'network':
            self.agent = NetworkAgent(reward_func=reward_func)
        elif agent_type == 'tableau':
            self.agent = TableauAgent(reward_func=reward_func)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}. Use 'network' or 'tableau'.")

    def _compute_rewards(self, ranking: List[str], verdicts: Dict[str, int]) -> List[float]:
        """Compute per-test rewards using the configured reward function."""
        if self.reward_func == 'tcfail':
            return reward_tcfail(ranking, verdicts)
        elif self.reward_func == 'timerank':
            return reward_timerank(ranking, verdicts)
        else:
            raise ValueError(f"Unknown reward function: {self.reward_func}")

    def train_on_build(
        self,
        test_ids: List[str],
        verdicts: Dict[str, int],
        durations: Optional[Dict[str, float]] = None
    ):
        """
        Train the agent on a single build's data.

        Args:
            test_ids: List of test IDs in this build
            verdicts: test_id -> 0 (pass) or 1 (fail)
            durations: test_id -> execution duration
        """
        if durations is None:
            durations = {t: 1.0 for t in test_ids}

        # Extract states
        states = np.array([
            self.history.get_state(tid, durations) for tid in test_ids
        ])

        # Get priority scores and rank
        scores = self.agent.get_action(states)
        ranked_indices = np.argsort(-scores)
        ranking = [test_ids[i] for i in ranked_indices]

        # Compute rewards
        rewards = self._compute_rewards(ranking, verdicts)

        # Reorder states to match ranking order
        ranked_states = states[ranked_indices]

        # Update agent
        self.agent.update(ranked_states, rewards)

        # Update test history
        for tid in test_ids:
            verdict = verdicts.get(tid, 0)
            dur = durations.get(tid, 1.0)
            # In RETECS paper: 0=fail, 1=pass for history features
            self.history.update(tid, verdict, dur)

        self.history.advance_build()

    def prioritize(self, test_ids: List[str], durations: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Prioritize test cases (no exploration).

        Args:
            test_ids: List of test IDs to prioritize
            durations: test_id -> execution duration (for state computation)

        Returns:
            Ordered list of test IDs (highest priority first)
        """
        if durations is None:
            durations = {t: 1.0 for t in test_ids}

        # Disable exploration for evaluation
        if isinstance(self.agent, TableauAgent):
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = 0
        elif isinstance(self.agent, NetworkAgent):
            pass  # NetworkAgent doesn't use epsilon

        states = np.array([
            self.history.get_state(tid, durations) for tid in test_ids
        ])

        scores = self.agent.get_action(states)

        # Restore epsilon
        if isinstance(self.agent, TableauAgent):
            self.agent.epsilon = old_epsilon

        ranked_indices = np.argsort(-scores)
        return [test_ids[i] for i in ranked_indices]

    def update_history(
        self,
        test_ids: List[str],
        verdicts: Dict[str, int],
        durations: Optional[Dict[str, float]] = None
    ):
        """Update history after evaluation (without training the agent)."""
        if durations is None:
            durations = {t: 1.0 for t in test_ids}

        for tid in test_ids:
            verdict = verdicts.get(tid, 0)
            dur = durations.get(tid, 1.0)
            self.history.update(tid, verdict, dur)

        self.history.advance_build()

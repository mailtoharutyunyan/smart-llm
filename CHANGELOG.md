# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-04-10

### Added
- Autonomous evolution orchestrator (`evolve.py`) for single-command self-improvement loops.
- Adversarial quality filter with configurable evaluator floor (0.60 bootstrap / 0.65 production).
- Anti-overfitting dataset builder with dependency-aware shuffling, noise injection, and centroid compression.
- Evaluator drift detection to monitor model stability across training iterations.
- Model registry with version tracking, promotion, and rollback support.
- Background trace generator for hands-free data collection.
- MCP server integration for IDE tool exposure.
- Tier 1 (deterministic) and Tier 2 (reasoning) evaluation benchmarks.

### Changed
- Difficulty score coefficients recalibrated to prevent verbosity gaming.
- Dataset builder shuffle rate increased to 45% for better structural diversity.

### Fixed
- Cold-start score inflation where high novelty/difficulty boosted weak evaluator scores above threshold.

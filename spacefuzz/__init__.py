"""SpaceFuzz: stress-test an astronaut's day plan before it starts.

readers/  data sources -> standard tables and the environment trace
core/     plan, astronaut state, one-day model, crash rules, data quality gate
fuzzing/  mutation operators, two-track engine, plan adjustment, midday check, advice
config.py thresholds, magnitudes, sampling profiles, seed (part of the cache key)
"""

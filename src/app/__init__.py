"""Application layer for MLflow Evaluation Agent.

Provides Streamlit chat interface and async-to-sync streaming utilities.
"""

from .streaming import async_to_sync_generator

__all__ = ["async_to_sync_generator"]

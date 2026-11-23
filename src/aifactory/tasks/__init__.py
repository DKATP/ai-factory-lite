"""Thin Databricks task entrypoints.

Each module here is a ``python -m aifactory.tasks.<name>`` entrypoint with the same shape:
parse arguments into a :class:`~aifactory.config.RunConfig`, call domain code, write a table,
print a one-line summary, exit non-zero on failure.

They contain no business logic. That is the point: the logic lives in modules that a test can
import without a workspace, and these wrappers exist only so the Jobs DAG has something to
call. If an entrypoint here grows an ``if``, it belongs somewhere else.
"""

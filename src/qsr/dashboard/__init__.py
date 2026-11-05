"""UI-independent logic for the Streamlit dashboard (Milestone 4).

This package never imports Streamlit, so it can be tested in any environment,
and it never imports Qiskit at module level, so the classical parts work
without the ``quantum`` extra. All scientific calculations are delegated to the
existing modules (``qsr.models``, ``qsr.simulation``, ``qsr.quantum``,
``qsr.visualization``); nothing is re-implemented here.
"""

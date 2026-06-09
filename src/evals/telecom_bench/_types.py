"""Shared record shapes for telecom_bench loaders."""

from __future__ import annotations

from typing import TypedDict


class MCQRecord(TypedDict, total=False):
    id: int
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str


class QARecord(TypedDict, total=False):
    question: str
    answer: str
    best_answer: str

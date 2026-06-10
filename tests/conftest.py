from __future__ import annotations

import pytest


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "NASA launched a new space telescope.",
        "The hockey team won the final match.",
        "Computer graphics rendering uses pixels.",
        "The political debate focused on policy.",
    ]

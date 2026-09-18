"""Report routes placeholder.

Forensic report generation is intentionally NOT implemented in Step 3; it
belongs to a later phase. The router is registered so the route organization
is already in place, but it exposes no endpoints yet.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["reports"])

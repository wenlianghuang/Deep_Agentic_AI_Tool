"""
Parlant SDK 整合的指南管理系統
使用官方 Parlant SDK 管理指南和客戶旅程
"""

from .parlant_manager import (
    get_guideline,
    get_customer_journey,
    initialize_parlant_sync
)

__all__ = [
    "get_guideline",
    "get_customer_journey",
    "initialize_parlant_sync"
]


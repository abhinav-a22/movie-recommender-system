"""
Version 3: Collaborative Filtering modules.
Includes Item-Based CF, User-Based CF, and unified CollaborativeRecommender facade.
"""

from .item_based import ItemBasedCF
from .user_based import UserBasedCF
from .collaborative import CollaborativeRecommender

__all__ = ["ItemBasedCF", "UserBasedCF", "CollaborativeRecommender"]


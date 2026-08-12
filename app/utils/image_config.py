"""
Centralized Image Configuration and Safe Resolution Utility for EstateRAG AI.
Provides default placeholder image, URL validation, and crash-proof image resolution.
"""

import os
from typing import Optional, List, Any

# Centralized default fallback placeholder image
DEFAULT_PLACEHOLDER_IMAGE = "https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=600&q=80"


def validate_single_image_url(url: str) -> bool:
    """Checks if a string is a non-empty valid image URL or asset path."""
    if not url or not isinstance(url, str):
        return False
    clean = url.strip()
    if len(clean) < 10:
        return False
    return clean.startswith(("http://", "https://", "assets/", "data/"))


def get_property_image(property_item: Any) -> str:
    """
    Safely resolves the primary image URL for a property item.
    Guarantees a valid string return and NEVER raises an exception.
    """
    try:
        if not property_item or not hasattr(property_item, "image_urls"):
            return DEFAULT_PLACEHOLDER_IMAGE

        urls = property_item.image_urls
        if isinstance(urls, str):
            urls = [u.strip() for u in urls.split(",") if u.strip()]

        if not urls or not isinstance(urls, list):
            return DEFAULT_PLACEHOLDER_IMAGE

        for url in urls:
            if isinstance(url, str) and validate_single_image_url(url):
                return url.strip()

        return DEFAULT_PLACEHOLDER_IMAGE
    except Exception:
        return DEFAULT_PLACEHOLDER_IMAGE

"""
Centralized Image Configuration and Safe Resolution Utility for EstateRAG AI.
Provides default placeholder images, URL validation, and crash-proof deterministic image resolution.
"""

import hashlib
from typing import Optional, List, Any

# Centralized default fallback placeholder image
DEFAULT_PLACEHOLDER_IMAGE = "https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=600&q=80"

# Curated architectural image pools by property type to guarantee visual diversity & zero duplicate lookups
CATEGORY_IMAGE_POOLS = {
    "Apartment": [
        "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1502005229762-fc1b2d82d883?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1493809842364-78817add7ffb?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=600&q=80",
    ],
    "Villa": [
        "https://images.unsplash.com/photo-1613977257363-707ba9348227?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1580587771525-78b9dba3b914?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1513694203232-719a280e022f?auto=format&fit=crop&w=600&q=80",
    ],
    "Independent House": [
        "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1570129477492-45c003edd2be?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1572120360610-d971b9d7767c?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1583608205776-bfd35f0d9f83?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1576941089067-2de3c901e126?auto=format&fit=crop&w=600&q=80",
    ],
    "Penthouse": [
        "https://images.unsplash.com/photo-1600607687920-4e2a09cf159d?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1512915922686-57c11dde9b6b?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1549517045-bc93de075e53?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1513584684374-8bab748fbf90?auto=format&fit=crop&w=600&q=80",
    ],
    "Land": [
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&w=600&q=80",
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=600&q=80",
    ],
}


def validate_single_image_url(url: str) -> bool:
    """Checks if a string is a non-empty valid image URL or asset path."""
    if not url or not isinstance(url, str):
        return False
    clean = url.strip()
    if len(clean) < 10:
        return False
    return clean.startswith(("http://", "https://", "assets/", "data/"))


def get_deterministic_fallback_image(property_id: str, property_type: str = "Apartment") -> str:
    """
    Computes a stable, deterministic fallback image based on stable MD5 hash of property_id.
    Ensures identical image selection across server restarts and reruns.
    """
    category = property_type if property_type in CATEGORY_IMAGE_POOLS else "Apartment"
    pool = CATEGORY_IMAGE_POOLS[category]
    
    # Stable hash of property ID
    hash_num = int(hashlib.md5(property_id.encode("utf-8")).hexdigest(), 16)
    idx = hash_num % len(pool)
    return pool[idx]


def get_property_image(property_item: Any) -> str:
    """
    Safely resolves the primary image URL for a property item.
    Prefers original URL if valid; falls back to deterministic category image.
    Guarantees a valid string return and NEVER raises an exception.
    """
    try:
        if not property_item:
            return DEFAULT_PLACEHOLDER_IMAGE

        prop_id = getattr(property_item, "property_id", "PROP-1001")
        prop_type = getattr(property_item, "property_type", "Apartment")

        urls = getattr(property_item, "image_urls", None)
        if isinstance(urls, str):
            urls = [u.strip() for u in urls.split(",") if u.strip()]

        if urls and isinstance(urls, list):
            for url in urls:
                if isinstance(url, str) and validate_single_image_url(url):
                    return url.strip()

        return get_deterministic_fallback_image(str(prop_id), str(prop_type))
    except Exception:
        return DEFAULT_PLACEHOLDER_IMAGE

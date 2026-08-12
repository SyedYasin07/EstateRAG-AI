"""
Image Validation & Health Utility for EstateRAG AI.
Provides image format verification, duplicate detection, placeholder fallback, and data health reporting.
"""

from typing import List, Dict, Set, Any
import urllib.parse
from app.data.schema import PropertyItem
from app.utils.image_config import DEFAULT_PLACEHOLDER_IMAGE, get_property_image


class ImageValidator:
    """Validates image URLs, checks dataset uniqueness, and reports image health metrics."""

    @staticmethod
    def validate_image_url(url: str) -> bool:
        """Verifies if input string is a valid non-empty image URL or asset path."""
        if not url or not isinstance(url, str):
            return False
        clean_url = url.strip()
        if len(clean_url) < 10:
            return False
        return clean_url.startswith(("http://", "https://", "assets/", "data/"))

    @classmethod
    def get_first_valid_image(cls, image_urls: List[str]) -> str:
        """Returns the first valid image URL or fallback placeholder."""
        if not image_urls:
            return DEFAULT_PLACEHOLDER_IMAGE
        for url in image_urls:
            if cls.validate_image_url(url):
                return url.strip()
        return DEFAULT_PLACEHOLDER_IMAGE

    @classmethod
    def check_duplicate_images(cls, properties: List[PropertyItem]) -> Dict[str, List[str]]:
        """Finds image URLs assigned to more than one property_id."""
        url_map: Dict[str, List[str]] = {}
        for p in properties:
            for url in p.image_urls:
                clean_url = url.strip()
                if clean_url and clean_url != DEFAULT_PLACEHOLDER_IMAGE:
                    if clean_url not in url_map:
                        url_map[clean_url] = []
                    url_map[clean_url].append(p.property_id)
        
        # Filter only duplicates
        duplicates = {url: pids for url, pids in url_map.items() if len(pids) > 1}
        return duplicates

    @classmethod
    def dataset_image_report(cls, properties: List[PropertyItem]) -> Dict[str, Any]:
        """Generates comprehensive data quality & image health metrics."""
        total_properties = len(properties)
        props_with_images = 0
        all_urls: Set[str] = set()
        duplicate_map = cls.check_duplicate_images(properties)
        duplicate_count = sum(len(pids) - 1 for pids in duplicate_map.values())

        for p in properties:
            valid_urls = [u for u in p.image_urls if cls.validate_image_url(u)]
            if valid_urls:
                props_with_images += 1
                all_urls.update(valid_urls)

        return {
            "total_properties": total_properties,
            "properties_with_images": props_with_images,
            "properties_without_images": total_properties - props_with_images,
            "unique_image_urls": len(all_urls),
            "duplicate_image_assignments": duplicate_count,
            "duplicate_details": duplicate_map,
            "image_coverage_pct": round((props_with_images / total_properties * 100), 1) if total_properties > 0 else 0.0,
        }

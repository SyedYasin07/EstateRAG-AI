"""
Dataset Validation Script for EstateRAG AI.
Audits properties.csv for duplicate IDs, duplicate image URLs, missing location fields, and data health.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data.dataset_loader import DatasetLoader
from app.utils.image_validator import ImageValidator


def validate():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data_path = os.environ.get("DATA_PATH", "data/properties.csv")
    print(f"==================================================")
    print(f" ESTATE RAG AI — DATASET HEALTH & VALIDATION AUDIT")
    print(f"==================================================")
    print(f"Target Dataset: {data_path}\n")

    loader = DatasetLoader(data_path=data_path)
    properties = loader.load_data()
    df = loader.clean_df

    total_properties = len(properties)
    cities = sorted(list(df["city"].unique()))
    localities = sorted(list(df["locality"].unique()))

    # 1. Duplicate Property ID check
    pids = [p.property_id for p in properties]
    duplicate_pids = set([x for x in pids if pids.count(x) > 1])

    # 2. Missing Locations Check
    missing_cities = df[df["city"].str.strip() == ""].shape[0]
    missing_localities = df[df["locality"].str.strip() == ""].shape[0]
    missing_prices = df[df["price_lakhs"] <= 0].shape[0]

    # 3. Image Health Check
    img_report = ImageValidator.dataset_image_report(properties)

    print("DATASET AUDIT SUMMARY")
    print("---------------------")
    print(f"Total Listings       : {total_properties}")
    print(f"Cities Covered       : {len(cities)} ({', '.join(cities[:6])}...)")
    print(f"Localities Covered   : {len(localities)}")
    print(f"Unique Image URLs    : {img_report['unique_image_urls']}")
    print(f"Props With Images    : {img_report['properties_with_images']} ({img_report['image_coverage_pct']}%)")
    print(f"Duplicate IDs        : {len(duplicate_pids)}")
    print(f"Duplicate Image URLs : {img_report['duplicate_image_assignments']}")
    print(f"Missing Cities       : {missing_cities}")
    print(f"Missing Localities   : {missing_localities}")
    print(f"Missing/Invalid Price: {missing_prices}")
    print("---------------------\n")

    if duplicate_pids:
        print(f"⚠️ WARNING: Duplicate Property IDs found: {duplicate_pids}")
    if img_report["duplicate_details"]:
        print(f"⚠️ WARNING: Found {len(img_report['duplicate_details'])} image URLs assigned to multiple listings.")

    if not duplicate_pids and img_report["duplicate_image_assignments"] == 0:
        print("✅ DATASET HEALTH PASSED CLEANLY! No duplicate IDs or duplicate image assignments found.")


if __name__ == "__main__":
    validate()

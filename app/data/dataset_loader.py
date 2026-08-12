"""
Dataset loader for loading, cleaning, normalizing and inspecting real estate property data.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from app.data.schema import PropertyItem, DatasetSummary


COLUMN_MAPPINGS = {
    "id": "property_id",
    "prop_id": "property_id",
    "propertyid": "property_id",
    "code": "property_id",
    "name": "title",
    "property_name": "title",
    "price": "price_lakhs",
    "price_in_lakhs": "price_lakhs",
    "cost": "price_lakhs",
    "sqft": "area_sqft",
    "carpet_area": "area_sqft",
    "bhk": "bedrooms",
    "bedroom": "bedrooms",
    "baths": "bathrooms",
    "bathroom": "bathrooms",
    "type": "property_type",
}


class DatasetLoader:
    def __init__(self, data_path: str = "data/properties.csv"):
        self.data_path = data_path
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.properties: List[PropertyItem] = []

    def load_data(self) -> List[PropertyItem]:
        """Loads CSV or Excel data, cleans and validates property records."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at path: {self.data_path}")

        if self.data_path.endswith(".csv"):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format for dataset: {self.data_path}")

        self.raw_df = df.copy()
        self.clean_df, self.properties = self._clean_and_transform(df)
        return self.properties

    def _clean_and_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[PropertyItem]]:
        # Normalize column names to lowercase with underscores
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Apply column mapping dictionary
        rename_dict = {}
        for col in df.columns:
            if col in COLUMN_MAPPINGS:
                rename_dict[col] = COLUMN_MAPPINGS[col]
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)

        # Essential default fallback handling
        if "property_id" not in df.columns:
            df["property_id"] = [f"PROP-{1000 + i}" for i in range(1, len(df) + 1)]

        if "title" not in df.columns:
            df["title"] = df.apply(
                lambda r: f"{r.get('bedrooms', 2)} BHK {r.get('property_type', 'Property')} in {r.get('location', 'Prime Location')}",
                axis=1,
            )

        # Standard missing value fill
        df["location"] = df.get("location", pd.Series(["City Center"] * len(df))).fillna("City Center").astype(str)
        df["city"] = df.get("city", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)
        df["price_lakhs"] = pd.to_numeric(df.get("price_lakhs", 0), errors="coerce").fillna(0.0)
        df["area_sqft"] = pd.to_numeric(df.get("area_sqft", 0), errors="coerce").fillna(0).astype(int)
        df["bedrooms"] = pd.to_numeric(df.get("bedrooms", 1), errors="coerce").fillna(1).astype(int)
        df["bathrooms"] = pd.to_numeric(df.get("bathrooms", 1), errors="coerce").fillna(1).astype(int)
        df["property_type"] = df.get("property_type", pd.Series(["Apartment"] * len(df))).fillna("Apartment").astype(str)
        df["amenities"] = df.get("amenities", pd.Series([""] * len(df))).fillna("").astype(str)
        df["furnishing"] = df.get("furnishing", pd.Series(["Unfurnished"] * len(df))).fillna("Unfurnished").astype(str)
        df["age_years"] = pd.to_numeric(df.get("age_years", 0), errors="coerce").fillna(0).astype(int)
        df["description"] = df.get("description", pd.Series([""] * len(df))).fillna("").astype(str)
        df["state"] = df.get("state", pd.Series([""] * len(df))).fillna("").astype(str)
        df["district"] = df.get("district", pd.Series([""] * len(df))).fillna("").astype(str)
        df["locality"] = df.get("locality", pd.Series([""] * len(df))).fillna("").astype(str)
        df["area"] = df.get("area", pd.Series([""] * len(df))).fillna("").astype(str)
        df["landmark"] = df.get("landmark", pd.Series([""] * len(df))).fillna("").astype(str)
        df["pincode"] = df.get("pincode", pd.Series([""] * len(df))).fillna("").astype(str)
        df["image_urls"] = df.get("image_urls", pd.Series([""] * len(df))).fillna("").astype(str)

        # Deduplicate on property_id
        df.drop_duplicates(subset=["property_id"], keep="first", inplace=True)

        properties: List[PropertyItem] = []
        for _, row in df.iterrows():
            item = PropertyItem(
                property_id=str(row["property_id"]),
                title=str(row["title"]),
                location=str(row["location"]),
                city=str(row["city"]),
                price_lakhs=float(row["price_lakhs"]),
                area_sqft=int(row["area_sqft"]),
                bedrooms=int(row["bedrooms"]),
                bathrooms=int(row["bathrooms"]),
                property_type=str(row["property_type"]),
                amenities=str(row["amenities"]),
                furnishing=str(row["furnishing"]),
                age_years=int(row["age_years"]),
                description=str(row["description"]),
                state=str(row["state"]),
                district=str(row["district"]),
                locality=str(row["locality"]),
                area=str(row["area"]),
                landmark=str(row["landmark"]),
                pincode=str(row["pincode"]),
                image_urls=str(row["image_urls"]),
            )
            properties.append(item)

        return df, properties

    def get_summary(self) -> DatasetSummary:
        """Returns structured summary statistics of loaded property dataset."""
        if not self.properties:
            self.load_data()

        df = self.clean_df
        all_amenities = set()
        for p in self.properties:
            all_amenities.update(p.amenities)

        return DatasetSummary(
            total_properties=len(self.properties),
            cities=sorted(list(df["city"].unique())),
            property_types=sorted(list(df["property_type"].unique())),
            min_price_lakhs=float(df["price_lakhs"].min()) if not df.empty else 0.0,
            max_price_lakhs=float(df["price_lakhs"].max()) if not df.empty else 0.0,
            avg_price_lakhs=round(float(df["price_lakhs"].mean()), 2) if not df.empty else 0.0,
            min_area_sqft=int(df["area_sqft"].min()) if not df.empty else 0,
            max_area_sqft=int(df["area_sqft"].max()) if not df.empty else 0,
            available_amenities=sorted(list(all_amenities)),
        )

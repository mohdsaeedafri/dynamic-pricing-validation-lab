"""Shared schema and display constants."""

REQUIRED_COLUMNS = (
    "PRICE_RETAIL",
    "PRICE_CURRENT",
    "RunDate",
    "DEPARTMENT",
    "CATEGORY",
)

OPTIONAL_COLUMNS = (
    "SKU",
    "PRODUCT_NAME",
    "BRAND",
    "PROMOTION",
    "SHIPPING_LOCATION",
)

# Keys are normalized by removing spaces and punctuation and using lowercase.
COLUMN_ALIASES = {
    "priceretail": "PRICE_RETAIL",
    "retailprice": "PRICE_RETAIL",
    "listprice": "PRICE_RETAIL",
    "msrp": "PRICE_RETAIL",
    "pricecurrent": "PRICE_CURRENT",
    "currentprice": "PRICE_CURRENT",
    "sellingprice": "PRICE_CURRENT",
    "rundate": "RunDate",
    "date": "RunDate",
    "effectivedate": "RunDate",
    "department": "DEPARTMENT",
    "category": "CATEGORY",
    "sku": "SKU",
    "productid": "SKU",
    "productname": "PRODUCT_NAME",
    "name": "PRODUCT_NAME",
    "brand": "BRAND",
    "promotion": "PROMOTION",
    "promo": "PROMOTION",
    "shippinglocation": "SHIPPING_LOCATION",
    "location": "SHIPPING_LOCATION",
    "zipcode": "SHIPPING_LOCATION",
    "zip": "SHIPPING_LOCATION",
}

SEASON_BY_MONTH = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
    12: "Winter",
}

MAX_UPLOAD_ROWS = 100_000


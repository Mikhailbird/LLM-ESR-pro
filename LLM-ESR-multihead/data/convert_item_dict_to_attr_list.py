import json

def convert_to_attr_list_format(input_path, output_path):
    raw = json.load(open(input_path))
    item2attr = {}

    for asin, data in raw.items():
        attrs = []

        # brand
        brand = data.get('brand')
        if brand:
            attrs.append(f"brand:{brand.strip()}")

        # category: pick the last one from the first path
        categories = data.get('categories', [])
        if isinstance(categories, list) and categories and isinstance(categories[0], list):
            cat_leaf = categories[0][-1]
            attrs.append(f"category:{cat_leaf.strip()}")

        # price bucket
        price = data.get('price')
        if isinstance(price, (int, float)):
            if price < 20:
                price_bucket = "<20"
            elif price <= 50:
                price_bucket = "20-50"
            else:
                price_bucket = ">50"
            attrs.append(f"price:{price_bucket}")

        if attrs:
            item2attr[asin] = attrs

    with open(output_path, "w") as f:
        json.dump(item2attr, f, indent=2)
    print(f"Converted {len(item2attr)} items to attribute list format.")

if __name__ == "__main__":
    convert_to_attr_list_format("data/beauty/handled/item2attributes.json", "data/beauty/handled/item2attributes_flat.json")
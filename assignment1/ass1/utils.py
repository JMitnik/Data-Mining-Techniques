def alias_item(item, alias_map):
    for (key, values) in alias_map.items():
        for value in values:
            if value.lower() in item.lower():
                return str(key).lower()

    return item.lower()

def range_list_expanded(list_of_ranges: list[str]) -> list[int]:
    """
    Expand ranges specified in interface. They should be formatted as [starts_index:end_index].

    Args
    ----
        list_of_ranges (List[str]): List of indexes or ranges.

    Returns
    -------
        A list of all indexes covered in ranges or individual indexes
    """
    indexes: list[int] = []

    for val in list_of_ranges:
        if ":" in val:
            indexes.extend(range(*[int(num) for num in val.split(":")]))
        else:
            indexes.append(int(val))

    return indexes

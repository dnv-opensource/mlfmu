def to_camel(string: str) -> str:
    """Change casing of string to CamelCase."""
    words = string.split("_")
    camel_string: str = words[0] + "".join(word.capitalize() for word in words[1:])
    return camel_string

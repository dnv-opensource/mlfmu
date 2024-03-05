
# Style Guide

All code shall be [black](https://pypi.org/project/black/) formatted.

References, details as well as examples of bad/good styles and their respective reasoning can be found below.

## References

* [PEP-8](https://www.python.org/dev/peps/pep-0008/) (see also [pep8.org](https://pep8.org/))
* [PEP-257](https://www.python.org/dev/peps/pep-0257/)
* Python style guide by [theluminousmen.com](https://luminousmen.com/post/the-ultimate-python-style-guidelines)
* [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code)
* [Jupyter](https://jupyter.readthedocs.io/en/latest/contributing/ipython-dev-guide/coding_style.html) style guide
* Python style guide on [learnpython.com](https://learnpython.com/blog/python-coding-best-practices-and-style-guidelines/)
* [flake8](https://flake8.pycqa.org/en/latest/)
* [black](https://pypi.org/project/black/)

## Code Layout

* Use 4 spaces instead of tabs
* Maximum line length is 88 characters (not 79 as proposed in [PEP-8](https://www.python.org/dev/peps/pep-0008/))
* 2 blank lines between classes and functions
* 1 blank line within class, between class methods
* Use blank lines for logic separation of functionality within functions/methods wherever it is justified
* No whitespace adjacent to parentheses, brackets, or braces

```py
    # Bad
    spam( items[ 1 ], { key1 : arg1, key2 : arg2 }, )

    # Good
    spam(items[1], {key1: arg1, key2: arg2}, [])
```

* Surround operators with single whitespace on either side.

```py
    # Bad
    x<1

    # Good
    x == 1
```

* Never end your lines with a semicolon, and do not use a semicolon to put two statements on the same line
* When branching, always start a new block on a new line

```py
    # Bad
    if flag: return None

    # Good
    if flag:
        return None
```

* Similarly to branching, do not write methods on one line in any case:

```py
    # Bad
    def do_something(self): print("Something")

    # Good
    def do_something(self):
        print("Something")
```

* Place a class's `__init__` function (the constructor) always at the beginning of the class

## Line Breaks

* If function arguments do not fit into the specified line length, move them to a new line with indentation

```py
    # Bad
    def long_function_name(var_one, var_two, var_three,
        var_four):
        print(var_one)

    # Bad
    def long_function_name(var_one, var_two, var_three,
            var_four):
        print(var_one)

    # Better (but not preferred)
    def long_function_name(var_one,
                           var_two,
                           var_three,
                           var_four):
        print(var_one)

    # Good (and preferred)
    def long_function_name(
        var_one,
        var_two,
        var_three,
        var_four,
    ):
        print(var_one)
```

* Move concatenated logical conditions to new lines if the line does not fit the maximum line size. This will help you understand the condition by looking from top to bottom. Poor formatting makes it difficult to read and understand complex predicates.

```py
    # Good
    if (
        this_is_one_thing
        and that_is_another_thing
        or that_is_third_thing
        or that_is_yet_another_thing
        and one_more_thing
    ):
        do_something()
```

* Where binary operations stretch multiple lines, break lines before the binary operators, not thereafter

```py
    # Bad
    GDP = (
        private_consumption +
        gross_investment +
        government_investment +
        government_spending +
        (exports - imports)
    )

    # Good
    GDP = (
        private_consumption
        + gross_investment
        + government_investment
        + government_spending
        + (exports - imports)
    )
```

* Chaining methods should be broken up on multiple lines for better readability

```py
    (
        df.write.format("jdbc")
        .option("url", "jdbc:postgresql:dbserver")
        .option("dbtable", "schema.tablename")
        .option("user", "username")
        .option("password", "password")
        .save()
    )
```

* Add a trailing comma to sequences of items when the closing container token ], ), or } does not appear on the same line as the final element

```py
    # Bad
    y = [
        0,
        1,
        4,
        6
    ]
    z = {
        'a': 1,
        'b': 2
    }

    # Good
    x = [1, 2, 3]

    # Good
    y = [
        0,
        1,
        4,
        6,         <- note the trailing comma
    ]
    z = {
        'a': 1,
        'b': 2,    <- note the trailing comma
    }
```

## String Formatting

* When quoting string literals, use double-quoted strings. When the string itself contains single or double quote characters, however, use the respective other one to avoid backslashes in the string. It improves readability.
* Use f-strings to format strings:

```py
    # Bad
    print("Hello, %s. You are %s years old. You are a %s." % (name, age, profession))

    # Good
    print(f"Hello, {name}. You are {age} years old. You are a {profession}.")
```

* Use multiline strings,  not \ , since it gets much more readable.

```py
    raise AttributeError(
        "Here is a multiline error message with a very long first line "
        "and a shorter second line."
    )
```

## Naming Conventions

* For module names: `lowercase` .
Long module names can have words separated by underscores (`really_long_module_name.py`), but this is not required. Try to use the convention of nearby files.
* For class names: `CamelCase`
* For methods, functions, variables and attributes: `lowercase_with_underscores`
* For constants: `UPPERCASE` or `UPPERCASE_WITH_UNDERSCORES`
(Python does not differentiate between variables and constants. Using UPPERCASE for constants is just a convention, but helps a lot to quickly identify variables meant to serve as constants.)
* Implementation-specific private methods and variables will use `_single_underscore_prefix`
* Don't include the type of a variable in its name.
      E.g. use `senders` instead of `sender_list`
* Names shall be clear about what a variable, class, or function contains or does. If you struggle to come up with a clear name, rethink your architecture: Often, the difficulty in finding a crisp name for something is a hint that separation of responsibilities can be improved. The solution then is less to agree on a name, but to start a round of refactoring: The name you're seeking often comes naturally then with refactoring to an improved architecture with clear responsibilities.
(see [SRP](https://en.wikipedia.org/wiki/Single-responsibility_principle), Single-Responsibilty Principle by Robert C. Martin)

## Named Arguments

* Use named arguments to improve readability and avoid mistakes introduced with future code maintenance

```py
    # Bad
    urlget("[http://google.com](http://google.com/)", 20)

    # Good
    urlget("[http://google.com](http://google.com/)", timeout=20)
```

* Never use mutable objects as default arguments in Python. If an attribute in a class or a named parameter in a function is of a mutable data type (e.g. a list or dict), never set its default value in the declaration of an object but always set it to None first, and then only later assign the default value in the class's constructor, or the functions body, respectively. Sounds complicated? If you prefer the shortcut, the examples below are your friend.
If your are interested in the long story including the why‘s, read these discussions on [Reddit](https://old.reddit.com/r/Python/comments/opb7hm/do_not_use_mutable_objects_as_default_arguments/) and [Twitter](https://twitter.com/willmcgugan/status/1419616480971399171).

```py
    # Bad
    class Foo:
        items = []

    # Good
    class Foo:
        items = None
        def __init__(self):
            self.items = []


    # Bad
    class Foo:
        def __init__(self, items=[]):
            self.items = items

    # Good
    class Foo:
        def __init__(self, items=None):
            self.items = items or []


    # Bad
    def some_function(x, y, items=[]):
        ...

    # Good
    def some_function(x, y, items=None):
        items = items or []
        ...
```

## Commenting

* First of all, if the code needs comments to clarify its work, you should think about refactoring it. The best comment to code is the code itself.
* Describe complex, possibly incomprehensible points and side effects in the comments
* Separate `#` and the comment with one whitespace

```py
    #bad comment
    # good comment
```

* Use inline comments sparsely
* Where used, inline comments shall have 2 whitespaces before the `#` and one whitespace thereafter

```py
    x = y + z  # inline comment
    str1 = str2 + str3  # another inline comment
```

* If a piece of code is poorly understood, mark the piece with a `@TODO:` tag and your name to support future refactoring:

```py
    def get_ancestors_ids(self):
        # @TODO: Do a cache reset while saving the category tree. CLAROS, YYYY-MM-DD
        cache_name = f"{self._meta.model_name}_ancestors_{self.pk}"
        cached_ids = cache.get(cache_name)
        if cached_ids:
            return cached_ids

        ids = [c.pk for c in self.get_ancestors(include_self=True)]
        cache.set(cache_name, ids, timeout=3600)

        return ids
```

## Type hints

* Use type hints in function signatures and module-scope variables. This is good documentation and can be used with linters for type checking and error checking. Use them whenever possible.
* Use pyi files to type annotate third-party or extension modules.

## Docstrings

* All Docstrings should be written in [Numpy](https://numpydoc.readthedocs.io/en/latest/format.html) format. For a good tutorial on Docstrings, see [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code)
* In a Docstring, summarize function/method behavior and document its arguments, return value(s), side effects, exceptions raised, and restrictions
* Wrap Docstrings with triple double quotes (""")
* The description of the arguments must be indented

```py
    def some_method(name, print=False):
        """This function does something

        Parameters
        ----------
        name : str
            The name to use
        print: bool, optional
            A flag used to print the name to the console, by default False

        Raises
        ------
        KeyError
            If name is not found

        Returns
        -------
        int
            The return code
        """
        ...
        return 0
```

## Exceptions

* Raise specific exceptions and catch specific exceptions, such as KeyError, ValueError, etc.
* Do not raise or catch just Exception, except in rare cases where this is unavoidable, such as a try/except block on the top-level loop of some long-running process. For a good tutorial on why this matters, see [The Most Diabolical Python Antipattern](https://realpython.com/the-most-diabolical-python-antipattern/).
* Minimize the amount of code in a try/except block. The larger the body of the try,
      the more likely that an exception will be raised by a line of code that you didn’t expect to raise an exception.

## Imports

* Avoid creating circular imports by importing modules more specialized than the one you are editing
* Relative imports are forbidden ([PEP-8](https://www.python.org/dev/peps/pep-0008/) only “highly discourages” them). Where absolutely needed, the `from future import absolute_import` syntax should be used (see [PEP-328](https://www.python.org/dev/peps/pep-0328/))
* Never use wildcard imports (`from <module> import *`). Always be explicit about what you're importing. Namespaces make code easier to read, so use them.
* Break long imports using parentheses and indent by 4 spaces. Include the trailing comma after the last import and place the closing bracket on a separate line

```py
    from my_pkg.utils import (
        some_utility_method_1,
        some_utility_method_2,
        some_utility_method_3,
        some_utility_method_4,
        some_utility_method_5,
    )
```

* Imports should be written in the following order, separated by a blank line:
    1. build-in modules
    2. third-party modules
    3. local application/library specific imports

```py
    import logging
    import os
    import typing as T

    import pandas as pd
    import numpy as np

    import my_package
    import my_package.my_module
    from my_package.my_module import my_function, MyClass
```

* Even if a Python file is intended to be used as executable / script file only, it shall still be importable as a module, and its import should not have any side effects. Its main functionality shall hence be in a `main()` function, so that the code can be imported as a module for testing or being reused in the future:

```py
    def main():
        ...

    if __name__ == "__main__":
        main()
```

## Unit-tests

* Use pytest as the preferred testing framework.
* The name of a test shall clearly express what is being tested.
* Each test should preferably check only one specific aspect.

```py
    # Bad
    def test_smth():
        result = f()
        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3
        assert result[3] == 4

    # Good
    def test_smth_type():
        result = f()
        assert isinstance(result, list), "Result should be list"

    def test_smth_values():
        result = f()
        assert set(result) == set(expected), f"Result should be {set(expected)}"
```

## And finally: It is a bad idea to use

* global variables.
* iterators where they can be replaced by vectorized operations.
* lambda where it is not required.
* map and lambda where it can be replaced by a simple list comprehension.
* multiple nested maps and lambdas.
* nested functions. They are hard to test and debug.

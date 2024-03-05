# my-package
my-package is an example package

my-package supports
* ..


## Installation

```sh
pip install my-package
```

## Usage Example

API:

```py
from my_package import ...
```

CLI:

```sh
my-package ...
```

_For more examples and usage, please refer to my-package's [documentation][my_package_docs]._

## Development Setup

1. Install Python 3.9 or higher, i.e. [Python 3.10](https://www.python.org/downloads/release/python-3104/) or [Python 3.11](https://www.python.org/downloads/release/python-3114/)

2. Update pip and setuptools:

    ```sh
    python -m pip install --upgrade pip setuptools
    ```

3. git clone the my-package repository into your local development directory:

    ```sh
    git clone https://github.com/dnv-innersource/my-package path/to/your/dev/my-package
    ```

4. In the my-package root folder:

    Create a Python virtual environment:

    ```sh
    python -m venv .venv
    ```

    Activate the virtual environment:

    ..on Windows:

    ```sh
    > .venv\Scripts\activate.bat
    ```

    ..on Linux:

    ```sh
    source .venv/bin/activate
    ```

    Update pip and setuptools:

    ```sh
    (.venv) $ python -m pip install --upgrade pip setuptools
    ```

    (Optional) If you want PyTorch cuda support on your local machine
    (i.e. to use your GPU for torch operations), you should preferably install PyTorch with cuda support first, before installing all other dependendencies.
    On the official [PyTorch website](https://pytorch.org/get-started/locally/)
    you can generate a pip install command matching your local machine's operating system, using a wizard.
    If you are on Windows, the resulting pip install command will most likely look something like this:

    ```sh
    (.venv) $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    _Hint:_ If you are unsure which cuda version to indicate in above `pip install .. /cuXXX` command, you can use the shell command `nvidia-smi` on your local system to find out the cuda version supported by the current graphics driver installed on your system. When then generating the `pip install` command with the wizard from the [PyTorch website](https://pytorch.org/get-started/locally/), select the cuda version that matches the major version of what your graphics driver supports (major version must match, minor version may deviate).

    Install my-package's dependencies. <br>

    ```sh
    (.venv) $ pip install -r requirements-dev.txt
    ```
    This should return without errors.

    Finally, install my-package itself, yet not as a regular package but as an _editable_ package instead, using the pip install option -e:
    ```sh
    (.venv) $ pip install -e .
    ```

5. Test that the installation works (in the my-package root folder):

    ```sh
    (.venv) $ pytest .
    ```

## Meta

All code in my-package is DNV intellectual property and for DNV internal use only.

Copyright (c) 2024 [DNV](https://www.dnv.com) AS. All rights reserved.

Author One - [@LinkedIn](https://www.linkedin.com/in/authorone) - author.one@dnv.com

Author Two - [@LinkedIn](https://www.linkedin.com/in/authortwo) - author.two@dnv.com

Author Three - [@LinkedIn](https://www.linkedin.com/in/authorthree) - author.three@dnv.com

@TODO: (1) Adapt to chosen license (or delete if no license is applied). <br>
@TODO: (2) Adapt or delete the license file (LICENSE.md) <br>
@TODO: (3) Adapt or delete the license entry in setup.cfg <br>
Distributed under the XYZ license. See [LICENSE](LICENSE.md) for more information.

[https://github.com/dnv-innersource/my-package](https://github.com/dnv-innersource/my-package)

## Contributing

1. Fork it (<https://github.com/dnv-innersource/my-package/fork>) (Note: this is currently disabled for this repo. For DNV internal development, continue with the next step.)
2. Create an issue in your GitHub repo
3. Create your branch based on the issue number and type (`git checkout -b issue-name`)
4. Evaluate and stage the changes you want to commit (`git add -i`)
5. Commit your changes (`git commit -am 'place a descriptive commit message here'`)
6. Push to the branch (`git push origin issue-name`)
7. Create a new Pull Request in GitHub

For your contribution, please make sure you follow the [STYLEGUIDE](STYLEGUIDE.md) before creating the Pull Request.

<!-- Markdown link & img dfn's -->
[my_package_docs]: https://dnv-innersource.github.io/my-package/README.html

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dictIO.utils.path import relative_path
from json_schema_for_humans.generate import (
    SchemaToRender,
    TemplateRenderer,
    generate_schemas_doc,
)
from json_schema_for_humans.generation_configuration import GenerationConfiguration

from mlfmu.types.fmu_component import ModelComponent

if TYPE_CHECKING:
    import os

    from pydantic import BaseModel
    from pydantic._internal._model_construction import ModelMetaclass

__ALL__ = ["publish_interface_schema"]


def generate_interface_schema(
    model: BaseModel | ModelMetaclass,
    schema_dir: str | os.PathLike[str] | None = None,
) -> None:
    """
    Generate a JSON interface schema file for the given model.

    Args
    ----
        model (BaseModel | ModelMetaclass): The pydantic model for which to generate the schema.
        schema_dir (str | os.PathLike[str], optional):
            The directory where the schema file will be saved. Defaults to None.
    """
    schema_dir_default = Path.cwd() / "docs/schema"
    schema_dir = schema_dir or schema_dir_default
    # Make sure schema_dir argument is of type Path. If not, cast it to Path type.
    schema_dir = schema_dir if isinstance(schema_dir, Path) else Path(schema_dir)

    # Assert model argument is a pydantic BaseModel
    # Background: ModelMetaClass is added just to please static type checking,
    #             which would otherwise complain.
    #             Behind the scenes in pdyantic, models always inherit the attributes of BaseModel.
    if not hasattr(model, "model_json_schema"):
        raise ValueError("model argument must be a pydantic BaseModel")
    model = cast("BaseModel", model)

    # Create schema_dir if it does not exist
    schema_dir.mkdir(parents=True, exist_ok=True)

    json_file: Path = schema_dir / "schema.json"
    schema = json.dumps(model.model_json_schema(by_alias=True), indent=4)

    with Path.open(json_file, "w", encoding="utf-8") as f:
        _ = f.write(schema)

    return


def generate_interface_docs(
    schema_dir: str | os.PathLike[str] | None = None,
    docs_dir: str | os.PathLike[str] | None = None,
) -> None:
    """
    Generate HTML documentation for the JSON interface schema files in the schema directory.

    Args
    ----
        schema_dir (str | os.PathLike[str], optional):
            The directory where the schema files are located. Defaults to None.
        docs_dir (str | os.PathLike[str], optional):
            The directory where the documentation files will be saved. Defaults to None.
    """
    schema_dir_default = Path.cwd() / "docs/schema"
    schema_dir = schema_dir or schema_dir_default

    docs_dir_default = Path.cwd() / "docs/interface"
    docs_dir = docs_dir or docs_dir_default

    # Make sure schema_dir and docs_dir are of type Path. If not, cast it to Path type.
    schema_dir = schema_dir if isinstance(schema_dir, Path) else Path(schema_dir)
    docs_dir = docs_dir if isinstance(docs_dir, Path) else Path(docs_dir)

    # Create dirs in case they don't exist
    schema_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Collect all schemata in schema dir
    pattern: str = "**.json"
    schemata: list[Path] = list(schema_dir.glob(pattern))

    # Generate html documentation for schemata
    config: GenerationConfiguration = GenerationConfiguration(
        template_name="js",
        expand_buttons=True,
        link_to_reused_ref=False,
        show_breadcrumbs=False,
    )

    schemas_to_render: list[SchemaToRender] = []

    for schema in schemata:
        rel_path: Path = relative_path(from_path=schema_dir, to_path=schema.parent)
        name: str = schema.stem
        html_file: Path = docs_dir / rel_path / f"{name}.html"
        schema_to_render: SchemaToRender = SchemaToRender(
            schema_file=schema,
            result_file=html_file,
            output_dir=None,
        )
        schemas_to_render.append(schema_to_render)

    _ = generate_schemas_doc(
        schemas_to_render=schemas_to_render,
        template_renderer=TemplateRenderer(config),
    )


def publish_interface_schema(
    schema_dir: str | os.PathLike[str] | None = None,
    docs_dir: str | os.PathLike[str] | None = None,
) -> None:
    """
    Publish the JSON schema and HTML documentation for the interface.

    Args
    ----
        schema_dir (str | os.PathLike[str], optional):
            The directory where the schema file will be saved. Defaults to None.
        docs_dir (str | os.PathLike[str], optional):
            The directory where the documentation files will be saved. Defaults to None.
    """
    # Generate JSON schema
    generate_interface_schema(model=ModelComponent, schema_dir=schema_dir)

    # Generate documentation HTML
    generate_interface_docs(schema_dir=schema_dir, docs_dir=docs_dir)

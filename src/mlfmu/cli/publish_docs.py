import logging

from mlfmu.utils.interface import publish_interface_schema

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the publishing process for the mlfmu interface docs, by calling the publish_interface_schema function."""
    logger.info("Start publish-interface-docs.py")
    publish_interface_schema()

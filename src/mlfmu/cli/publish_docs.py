import logging

from mlfmu.utils.interface import publish_interface_schema

logger = logging.getLogger(__name__)


def main():
    logger.info("Start publish-interface-docs.py")
    publish_interface_schema()

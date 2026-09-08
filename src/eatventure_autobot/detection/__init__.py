from eatventure_autobot.detection.opencv_matcher import OpenCvTemplateMatcher
from eatventure_autobot.domain.protocols import TemplateMatcher


def create_template_matcher() -> TemplateMatcher:
    return OpenCvTemplateMatcher()

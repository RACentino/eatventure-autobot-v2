from eatventure_autobot.detection.opencv_matcher import OpenCvTemplateMatcher
from eatventure_autobot.domain.protocols import TemplateMatcher


def create_template_matcher(threads: int = 1) -> TemplateMatcher:
    return OpenCvTemplateMatcher(threads)

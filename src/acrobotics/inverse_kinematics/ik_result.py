from typing import List


class IKResult:
    def __init__(self, success: bool, solutions: List = None):
        self.success = success
        if self.success:
            assert solutions is not None
            self.solutions = solutions

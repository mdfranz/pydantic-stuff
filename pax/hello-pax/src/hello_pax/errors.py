class HelloPaxError(Exception):
    """Base error carrying an optional structured rule code."""

    def __init__(self, message: str, rule_code: str | None = None) -> None:
        super().__init__(message)
        self.rule_code = rule_code

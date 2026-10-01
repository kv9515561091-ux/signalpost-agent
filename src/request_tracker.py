class RequestTracker:
    def __init__(self, limit: int = 2000):
        self.limit = limit
        self.count = 0

    def record(self, requests: int = 1):
        if self.count + requests > self.limit:
            raise RuntimeError(
                f"Outbound request limit exceeded: "
                f"{self.limit}"
            )

        self.count += requests

    def remaining(self) -> int:
        return self.limit - self.count

    def summary(self) -> dict:
        return {
            "requests_used": self.count,
            "request_limit": self.limit,
            "requests_remaining": self.remaining(),
        }


request_tracker = RequestTracker()

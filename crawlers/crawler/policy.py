from rotating_proxies.policy import BanDetectionPolicy


class ArxivBanPolicy(BanDetectionPolicy):
    def response_is_ban(self, request, response):
        if request.method == "HEAD" and response.status == 200:
            return False

        return super().response_is_ban(request, response)


class BiorxivBanPolicy(BanDetectionPolicy):
    def response_is_ban(self, request, response):
        if request.method == "HEAD" and response.status == 200:
            return False

        return super().response_is_ban(request, response)

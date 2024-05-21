import requests


class ProxyManager:
    def __init__(self):
        pass

    def get_random_proxy(self):
        return None


# Test your proxies
if __name__ == '__main__':
    proxy_manager = ProxyManager()

    for _ in range(10):
        random_proxy = proxy_manager.get_random_proxy()

        print(f"Random proxy: {random_proxy}")
        try:
            response = requests.get('https://api.ipify.org?format=json', proxies=random_proxy)
            # response = requests.get('http://example.com', proxies=random_proxy)
            print(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

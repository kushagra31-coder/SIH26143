import os, requests, json

token = os.environ.get("GFW_API_TOKEN")
headers = {"Authorization": f"Bearer {token}"}
resp = requests.get("https://gateway.api.globalfishingwatch.org/v3/vessels/search", headers=headers, params={"query": "imo:9337119", "datasets[0]": "public-global-vessel-identity:latest"})
print(json.dumps(resp.json(), indent=2))

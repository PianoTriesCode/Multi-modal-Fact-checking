from langchain_community.utilities import SearxSearchWrapper

searx = SearxSearchWrapper(
    searx_host="http://127.0.0.1:8089",
    unsecure=True,
    headers={"User-Agent": "Mozilla/5.0"},
)

print(searx.run("When was the Eiffel Tower built?"))

import requests

URL = "https://repo.anaconda.com/pkgs/main/linux-64/jupyter_core-4.10.0-py37h06a4308_0.conda"
# 2. download the data behind the URL
caminho_relativo = "libs/tar"
# 3. Open the response into a new file called instagram.ico
with open("{}/{}".format(caminho_relativo, "urls.txt")) as file:
    for url in file:
        url = url[:-1]
        print(url)
        if ".conda" in url[-6:]:
            url = "{}{}".format(url[:-6], ".tar.bz2")
            response = requests.get(url)
            nome_arquivo = (url.split("/"))[-1]
            open("{}/{}".format(caminho_relativo, nome_arquivo), "wb").write(response.content)
        else:
            response = requests.get(url)
            nome_arquivo = (url.split("/"))[-1]
            open("{}/{}".format(caminho_relativo, nome_arquivo), "wb").write(response.content)

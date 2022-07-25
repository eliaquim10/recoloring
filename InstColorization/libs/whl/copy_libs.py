import subprocess

caminho_relativo = "libs/whl"
# 3. Open the response into a new file called instagram.ico
with open("{}/{}".format(caminho_relativo, "finds.txt")) as file:
    for url in file:
        url = url[:-1]
        # print(url)
        nome_arquivo = (url.split("/"))[-1]
        subprocess.run(["cp", url, "{}/{}".format(caminho_relativo, nome_arquivo)], capture_output=True)
        # print(subprocess.run(["cp", url, "{}/{}".format(caminho_relativo, nome_arquivo)], capture_output=True))

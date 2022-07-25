import os
import subprocess


caminho_relativo = "libs/tar"
# print(os.listdir(caminho_relativo))

# caminho_relativo = "libs/whl"
# 3. Open the response into a new file called instagram.ico

# subprocess.run(["conda", "activate", "color"], capture_output=True)
"""
for url in os.listdir(caminho_relativo):
    
    # subprocess.run(["conda", "install", "{}/{}".format(caminho_relativo, url)], capture_output=True)

    s = subprocess.run(["conda", "install", "{}/{}".format(caminho_relativo, url)], capture_output=True)
    if "Error" in s:
        print(s)

caminho_relativo = "libs/whl"
for url in os.listdir(caminho_relativo):    
    # subprocess.run(["conda", "install", "{}/{}".format(caminho_relativo, url)], capture_output=True)
    print("="*50)
    print("Error" in str(subprocess.run(["pip", "install", "{}/{}".format(caminho_relativo, url)], capture_output=True)))

subprocess.run(["conda", "deactivate"], capture_output=True)
"""

def install(env, path_libs):
    tentativas = {url: 0 for url in os.listdir(path_libs)}

    while (any(map(lambda x: x > -1 and x < 3,tentativas.values()))):
        for url in tentativas.keys():            
            # subprocess.run([env, "install", "{}/{}".format(path_libs, url)], capture_output=True)
            if tentativas[url] >= 0:
                s = subprocess.run([env, "install", "{}/{}".format(path_libs, url)], capture_output=True)
                print(s)
                if "Error" in str(s):
                    tentativas[url] += 1
                else:
                    tentativas[url] = -1

install("pip", "libs/whl")
install("/opt/conda/bin/conda", "libs/tar")
# detectron2cu101
# libs/whl/fvcore-0.1.dev200506-py3-none-any.whl
# libs/whl/portalocker-2.4.0-py2.py3-none-any.whl 

# iopath
# detectron2cu110

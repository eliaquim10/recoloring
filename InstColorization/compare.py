def compare(s1, s2):
    s1 = s1.split("@")[0] if "@" in s1 else  s1.split("==")[0]
    if s1 not in s2:
        print("{} vs {}".format(s1, s2))
with open("env_colab.txt") as colab1:
    with open("env_colab2.txt") as colab2:
        for lib1, lib2 in zip(colab1, colab2):
            print("{} vs {}".format(lib1[:-1], lib2[:-1]))
            compare(lib2, lib1)
            # else:
            #     pass

import glob
import shutil
# liste_ocr=["kraken4.3.13.dev25","lectaurep-kraken4.3.13.dev25","tesseract0.3.10","REF"]
n=1
path_corpus = "../ELTeC-fra/salve%s/*"%n
# for ocr in liste_ocr:

for file in glob.glob(path_corpus):
    if ".txt" in file:
        print("file -->",file)
        src=file

    path_dest = file.split("/")
    print(path_dest)
    rep=path_dest[-1].split("_")
    rep_auteur= "_".join([rep[0],rep[1]])
    rep_ocr = rep_auteur+"_REF"

    path_destination="../archi/salve%s/%s/%s/%s"%(n,rep_auteur,rep_ocr,rep_ocr+".txt")
    print("path_destination --> ",path_destination)
    shutil.copy2(src, path_destination)
    #dest = '/archi/log.txt'

        # if rep_ocr in path_dest[-1]:
        #
        #     shutil.copy2(src, path_destination)